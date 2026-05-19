

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any, Coroutine

from dotenv import load_dotenv
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

import importlib

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(ROOT))

_brain = importlib.import_module("4_agent_brain")
CHROMA_PATH = _brain.CHROMA_PATH
GeminiQuotaExceeded = _brain.GeminiQuotaExceeded
QUOTA_BUSY_MSG = _brain.QUOTA_BUSY_MSG

from conversation import ConversationEngine, ConversationState
from tts_edge import DEFAULT_VOICE, SAMPLE_RATE, synthesize_pcm

from deepgram import DeepgramClient
from deepgram.clients.listen.enums import LiveTranscriptionEvents
from deepgram.clients.listen.v1.websocket.options import LiveOptions

load_dotenv(ROOT / ".env")

TTS_PCM_FORMAT = "pcm_16000"
TTS_PCM_CHUNK_BYTES = 4096
DEEPGRAM_UTTERANCE_END_MS = int(os.getenv("DEEPGRAM_UTTERANCE_END_MS", "2000"))
UTTERANCE_COMMIT_PAUSE_SEC = float(os.getenv("UTTERANCE_COMMIT_PAUSE_SEC", "1.2"))
UTTERANCE_PAUSE_AFTER_SPEECH_FINAL_SEC = float(
    os.getenv("UTTERANCE_PAUSE_AFTER_SPEECH_FINAL_SEC", "0.35")
)
QUOTA_BUSY_PCM_PATH = ROOT / "assets" / "quota_busy.pcm"

templates = Jinja2Templates(directory=str(ROOT / "templates"))


async def iter_pcm_chunks(pcm: bytes, chunk_size: int = TTS_PCM_CHUNK_BYTES):
    for offset in range(0, len(pcm), chunk_size):
        yield pcm[offset : offset + chunk_size]


def require_keys() -> None:
    if not (os.getenv("DEEPGRAM_API_KEY") or "").strip():
        raise RuntimeError("Missing required environment variable: DEEPGRAM_API_KEY")
    if not (os.getenv("GOOGLE_API_KEY") or "").strip():
        raise RuntimeError("Missing GOOGLE_API_KEY (Gemini field extraction)")


def validated_deepgram_key() -> str:
    key = (os.getenv("DEEPGRAM_API_KEY") or "").strip()
    if not key or any(c.isspace() for c in key):
        raise ValueError("DEEPGRAM_API_KEY missing or invalid")
    return key


def dg_listen_websocket_v1(client: DeepgramClient):
    return client.listen.websocket.v("1")


def extract_transcript_from_result(result: Any) -> str:
    channel = getattr(result, "channel", None)
    if not channel:
        return ""
    alts = getattr(channel, "alternatives", None) or []
    if not alts:
        return ""
    return (getattr(alts[0], "transcript", None) or "").strip()


class VoiceWebSession:
    def __init__(self, websocket: WebSocket, loop: asyncio.AbstractEventLoop) -> None:
        self.ws = websocket
        self.loop = loop
        self.engine = ConversationEngine()
        print(f"[TTS] Edge TTS only — voice {DEFAULT_VOICE}")
        self.dg_client = DeepgramClient(os.environ["DEEPGRAM_API_KEY"])
        self._dg_conn: Any = None
        self._dg_alive = False
        self._dg_reconnect_used = False
        self._mic_on = False
        self._speaking = False
        self._lock = asyncio.Lock()
        self._utterance_lock = asyncio.Lock()
        self._closed = asyncio.Event()
        self._phrase_parts: list[str] = []
        self._commit_task: asyncio.Task[None] | None = None
        self._turn_in_progress = False

    def _cancel_commit_task(self) -> None:
        task = self._commit_task
        if task and not task.done():
            task.cancel()
        self._commit_task = None

    def _deepgram_connection_open(self) -> bool:
        conn = self._dg_conn
        if conn is None:
            return False
        if getattr(conn, "_exit_event", None) is not None and conn._exit_event.is_set():
            return False
        return self._dg_alive

    def _on_deepgram_open(self, _client: Any, **_kwargs: Any) -> None:
        self._dg_alive = True

    def _on_deepgram_close(self, _client: Any, **_kwargs: Any) -> None:
        self._dg_alive = False
        self._schedule(self._handle_deepgram_drop(1011))

    def _on_deepgram_error(self, _client: Any, **_kwargs: Any) -> None:
        self._dg_alive = False
        self._schedule(self._handle_deepgram_drop(None))

    async def _handle_deepgram_drop(self, code: int | None) -> None:
        if self._closed.is_set() or not self._mic_on or self._speaking:
            return
        if self._dg_reconnect_used:
            return
        self._dg_reconnect_used = True
        print(
            f"[deepgram] WARNING: connection closed (code={code}); "
            "attempting one reconnect..."
        )
        await self._tear_down_deepgram()
        await self.ensure_deepgram_started()

    async def _tear_down_deepgram(self) -> None:
        self._cancel_commit_task()
        self._dg_alive = False
        conn = self._dg_conn
        self._dg_conn = None
        if conn:
            try:
                await asyncio.to_thread(conn.finish)
            except Exception:
                pass

    async def stream_pcm(self, pcm: bytes) -> bool:
        if not pcm or self._closed.is_set():
            return False
        self._speaking = True
        try:
            return await self._stream_pcm_inner(pcm)
        finally:
            self._speaking = False

    async def _stream_pcm_inner(self, pcm: bytes) -> bool:
        await self.send_json_safe(
            {
                "type": "tts_start",
                "sample_rate": SAMPLE_RATE,
                "format": TTS_PCM_FORMAT,
                "voice": DEFAULT_VOICE,
            }
        )
        await self.send_status("speaking", "[Speaking...]")
        async for chunk in iter_pcm_chunks(pcm):
            await self.ws.send_bytes(chunk)
        return True

    async def stream_tts(self, text: str) -> bool:
        spoken = (text or "").strip()
        if not spoken or self._closed.is_set():
            return False
        self._speaking = True
        try:
            pcm = await asyncio.to_thread(synthesize_pcm, spoken)
            if not pcm:
                return False
            return await self._stream_pcm_inner(pcm)
        except Exception as exc:
            print(f"[WARN] TTS failed: {exc}")
            return False
        finally:
            self._speaking = False

    async def send_json_safe(self, payload: dict[str, Any]) -> None:
        await self.ws.send_text(json.dumps(payload, ensure_ascii=False))

    async def send_status(self, state: str, text: str) -> None:
        await self.send_json_safe({"type": "status", "state": state, "text": text})

    def _schedule(self, coro: Coroutine[Any, Any, Any]) -> None:
        asyncio.run_coroutine_threadsafe(coro, self.loop)

    async def _on_deepgram_transcript(self, *args: Any, **kwargs: Any) -> None:
        result = kwargs.get("result") or (args[0] if args else None)
        if result is None or self._closed.is_set():
            return
        if not getattr(result, "is_final", False):
            return
        transcript = extract_transcript_from_result(result)
        if not transcript:
            return
        speech_final = bool(getattr(result, "speech_final", False))
        pause_sec = (
            UTTERANCE_PAUSE_AFTER_SPEECH_FINAL_SEC
            if speech_final
            else UTTERANCE_COMMIT_PAUSE_SEC
        )
        async with self._utterance_lock:
            self._phrase_parts.append(transcript)
        if self._commit_task and not self._commit_task.done():
            self._commit_task.cancel()
        self._commit_task = asyncio.create_task(
            self._commit_utterance_after_pause(pause_sec)
        )

    async def _commit_utterance_after_pause(self, pause_sec: float) -> None:
        try:
            await asyncio.sleep(pause_sec)
        except asyncio.CancelledError:
            return
        if self._closed.is_set():
            return
        async with self._utterance_lock:
            if self._turn_in_progress or not self._phrase_parts or self._closed.is_set():
                return
            combined = " ".join(self._phrase_parts).strip()
            self._phrase_parts.clear()
            self._turn_in_progress = True
        if len(combined) < 2:
            async with self._utterance_lock:
                self._turn_in_progress = False
            return
        try:
            await self.process_final_transcript(combined)
        finally:
            async with self._utterance_lock:
                self._turn_in_progress = False

    async def ensure_deepgram_started(self) -> None:
        if self._dg_conn is not None:
            return
        try:
            dg_key = validated_deepgram_key()
        except ValueError as exc:
            await self.send_json_safe({"type": "error", "detail": str(exc)})
            return
        self.dg_client = DeepgramClient(dg_key)
        conn = dg_listen_websocket_v1(self.dg_client)
        conn.on(
            LiveTranscriptionEvents.Open,
            lambda _dg_client, *a, **kw: self._on_deepgram_open(
                *(a if a else (_dg_client,)), **kw
            ),
        )
        conn.on(
            LiveTranscriptionEvents.Close,
            lambda _dg_client, *a, **kw: self._on_deepgram_close(
                *(a if a else (_dg_client,)), **kw
            ),
        )
        conn.on(
            LiveTranscriptionEvents.Error,
            lambda _dg_client, *a, **kw: self._on_deepgram_error(
                *(a if a else (_dg_client,)), **kw
            ),
        )
        conn.on(
            LiveTranscriptionEvents.Transcript,
            lambda _dg_client, *a, **kw: self._schedule(
                self._on_deepgram_transcript(*a, **kw)
            ),
        )
        opts = LiveOptions(
            model="nova-3-general",
            language="ar",
            encoding="linear16",
            channels=1,
            sample_rate=SAMPLE_RATE,
            punctuate=True,
            interim_results=True,
            utterance_end_ms=str(DEEPGRAM_UTTERANCE_END_MS),
        )
        ok = await asyncio.to_thread(conn.start, opts)
        if not ok:
            await self.send_json_safe(
                {"type": "error", "detail": "Deepgram rejected the WebSocket."}
            )
            return
        self._dg_conn = conn
        self._dg_alive = True

    async def deepgram_shutdown(self) -> None:
        await self._tear_down_deepgram()

    async def forward_pcm(self, data: bytes) -> None:
        if (
            not data
            or not self._mic_on
            or self._closed.is_set()
            or self._speaking
            or not self._deepgram_connection_open()
        ):
            return
        conn = self._dg_conn
        if conn is None:
            return
        try:
            ok = await asyncio.to_thread(conn.send, data)
            if ok is False:
                self._dg_alive = False
        except Exception:
            self._dg_alive = False

    async def handle_quota_busy(self) -> None:
        await self.send_json_safe(
            {"type": "assistant", "text": QUOTA_BUSY_MSG, "quota_busy": True}
        )
        if QUOTA_BUSY_PCM_PATH.is_file():
            await self.stream_pcm(QUOTA_BUSY_PCM_PATH.read_bytes())
        else:
            await self.stream_tts(QUOTA_BUSY_MSG)

    async def process_final_transcript(self, transcript: str) -> None:
        if self._closed.is_set():
            return
        async with self._lock:
            if self._closed.is_set():
                return
            await self.send_json_safe({"type": "transcript", "role": "user", "text": transcript})
            st = self.engine.session.conversation_state
            await self.send_json_safe({"type": "funnel_state", "step": st.value})
            await self.send_status("thinking", "[Thinking...]")

            try:
                turn = self.engine.handle_turn(transcript)
            except GeminiQuotaExceeded as exc:
                print(f"[warn] Gemini quota exhausted: {exc}")
                try:
                    await self.handle_quota_busy()
                finally:
                    await self.send_json_safe({"type": "tts_complete"})
                await self.send_status(
                    "idle",
                    "[Listening...]" if self._mic_on else "Ready",
                )
                return
            except Exception as exc:
                await self.send_json_safe({"type": "error", "detail": str(exc)})
                await self.send_status(
                    "idle",
                    "[Listening...]" if self._mic_on else "Ready",
                )
                return

            await self.send_json_safe(
                {
                    "type": "funnel_state",
                    "step": turn.state.value,
                    "intake_complete": self.engine.session.intake_complete(),
                }
            )

            if turn.properties:
                await self.send_json_safe({"type": "properties", "items": turn.properties})

            if turn.speech and not self._closed.is_set():
                await self.send_json_safe({"type": "assistant", "text": turn.speech})
                try:
                    await self.stream_tts(turn.speech)
                except Exception as exc:
                    if not self._closed.is_set():
                        await self.send_json_safe({"type": "error", "detail": f"TTS: {exc}"})
                finally:
                    if not self._closed.is_set():
                        await self.send_json_safe({"type": "tts_complete"})

            await self.send_status(
                "idle",
                "[Listening...]" if self._mic_on else "Ready",
            )

    async def handle_control(self, mic_on: bool) -> None:
        self._mic_on = mic_on
        if mic_on:
            await self.send_status("listening", "[Listening...]")
            await self.ensure_deepgram_started()
        else:
            await self.deepgram_shutdown()
            await self.send_status("idle", "Ready")

    async def run_receive_loop(self) -> None:
        try:
            while True:
                msg = await self.ws.receive()
                if msg.get("type") == "websocket.disconnect":
                    break
                if msg.get("type") != "websocket.receive":
                    continue
                blob = msg.get("bytes")
                if blob is not None:
                    await self.forward_pcm(blob)
                    continue
                raw_t = msg.get("text")
                if raw_t:
                    try:
                        data = json.loads(raw_t)
                    except json.JSONDecodeError:
                        continue
                    if data.get("type") == "control":
                        await self.handle_control(bool(data.get("mic")))
        except WebSocketDisconnect:
            pass
        finally:
            self._closed.set()
            self._cancel_commit_task()
            await self.deepgram_shutdown()


def create_app() -> FastAPI:
    app = FastAPI(title="Real Estate Voice Agent")

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request):
        return templates.TemplateResponse("index.html", {"request": request})

    @app.websocket("/ws/stream")
    async def stream_ws(ws: WebSocket):
        await ws.accept()
        try:
            require_keys()
        except RuntimeError as exc:
            await ws.send_text(json.dumps({"type": "error", "detail": str(exc)}))
            await ws.close()
            return
        if not CHROMA_PATH.is_dir():
            await ws.send_text(
                json.dumps(
                    {
                        "type": "error",
                        "detail": f"ChromaDB missing at {CHROMA_PATH}. Run fast_index.py first.",
                    }
                )
            )
            await ws.close()
            return
        loop = asyncio.get_running_loop()
        await VoiceWebSession(ws, loop).run_receive_loop()

    return app


app = create_app()


def main():
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("WEB_PORT", "8000")))


if __name__ == "__main__":
    main()
