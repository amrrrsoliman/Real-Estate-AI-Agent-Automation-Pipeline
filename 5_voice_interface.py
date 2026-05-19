"""
Real-time voice loop: microphone → Deepgram STT → LangGraph agent → Edge TTS (PCM playback).
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from typing import Any

import pyaudio
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage

from tts_edge import synthesize_pcm

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import importlib

_brain = importlib.import_module("4_agent_brain")
CHROMA_PATH = _brain.CHROMA_PATH
GraphState = _brain.GraphState
build_graph = _brain.build_graph

from deepgram import DeepgramClient
from deepgram.clients.listen.enums import LiveTranscriptionEvents
from deepgram.clients.listen.v1.websocket.options import LiveOptions

# ---------------------------------------------------------------------------
# Audio / model constants
# ---------------------------------------------------------------------------
SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_SIZE = 1024
FORMAT = pyaudio.paInt16
THREAD_ID = "voice-session-1"
PCM_CHUNK_BYTES = CHUNK_SIZE * 2  # int16 @ mono

QUIT_PHRASES = frozenset({"quit", "exit", "q", "bye", "مع السلامة", "سلام"})


def load_config() -> dict[str, str]:
    load_dotenv()
    keys = {
        "DEEPGRAM_API_KEY": os.getenv("DEEPGRAM_API_KEY", "").strip(),
    }
    missing = [name for name, value in keys.items() if not value]
    if missing:
        raise EnvironmentError(
            f"Missing environment variables: {', '.join(missing)}. "
            "Set them in .env or the system environment."
        )
    return keys


def open_deepgram_listen(dg: DeepgramClient):
    """Deepgram SDK v3.4+: websocket client (listen.live is deprecated)."""
    return dg.listen.websocket.v("1")


def extract_transcript(result: Any) -> str:
    channel = getattr(result, "channel", None)
    if not channel:
        return ""
    alternatives = getattr(channel, "alternatives", None) or []
    if not alternatives:
        return ""
    return (getattr(alternatives[0], "transcript", None) or "").strip()


def extract_agent_reply(state: GraphState) -> str:
    messages = state.get("messages") or []
    last_human = -1
    for i, msg in enumerate(messages):
        if isinstance(msg, HumanMessage):
            last_human = i
    parts: list[str] = []
    for msg in messages[last_human + 1 :]:
        if isinstance(msg, AIMessage):
            text = str(msg.content or "").strip()
            if text:
                parts.append(text)
    return "\n".join(parts).strip()


class VoiceInterface:
    def __init__(self) -> None:
        cfg = load_config()
        self.deepgram_key = cfg["DEEPGRAM_API_KEY"]

        self.loop: asyncio.AbstractEventLoop | None = None
        self.running = False
        self._processing = asyncio.Lock()
        self._speaking = False

        self.graph = build_graph()
        self.graph_config = {"configurable": {"thread_id": THREAD_ID}}
        self.deepgram = DeepgramClient(self.deepgram_key)

        self._pa: pyaudio.PyAudio | None = None
        self._mic_stream: Any = None
        self._out_stream: Any = None
        self._dg_connection: Any = None

    def _status(self, label: str) -> None:
        print(label, flush=True)

    async def handle_transcript(self, transcript: str) -> None:
        if not transcript or self._speaking:
            return
        if transcript.lower().strip() in QUIT_PHRASES:
            self.running = False
            return

        async with self._processing:
            print(f"[User]: {transcript}", flush=True)
            self._status("[🧠 Thinking...]")
            try:
                state = await self.graph.ainvoke(
                    {"messages": [HumanMessage(content=transcript)]},
                    config=self.graph_config,
                )
            except Exception as exc:
                print(f"[Error] Agent: {exc}", flush=True)
                self._status("[🎙️ Listening...]")
                return

            reply = extract_agent_reply(state)
            if not reply:
                self._status("[🎙️ Listening...]")
                return
            await self.speak(reply)

    # SDK emits handler(ws_client, result=payload); bound method ⇒ (self, ws_client, result=…).
    def _on_transcript_received(self, _connection: Any, result: Any, **_kwargs: Any) -> None:
        """Deepgram Transcript event (Results); user spec: result.is_final."""
        if not getattr(result, "is_final", False):
            return
        transcript = extract_transcript(result)
        if not transcript or not self.loop or not self.running:
            return
        asyncio.run_coroutine_threadsafe(
            self.handle_transcript(transcript),
            self.loop,
        )

    async def speak(self, text: str) -> None:
        self._speaking = True
        self._status("[🗣️ Speaking...]")
        try:
            if self._pa is None:
                self._pa = pyaudio.PyAudio()
            if self._out_stream is None:
                self._out_stream = self._pa.open(
                    format=FORMAT,
                    channels=CHANNELS,
                    rate=SAMPLE_RATE,
                    output=True,
                    frames_per_buffer=CHUNK_SIZE,
                )

            pcm = await asyncio.to_thread(synthesize_pcm, text)
            for offset in range(0, len(pcm), PCM_CHUNK_BYTES):
                chunk = pcm[offset : offset + PCM_CHUNK_BYTES]
                if chunk and self._out_stream:
                    await asyncio.to_thread(self._out_stream.write, chunk)
        except Exception as exc:
            print(f"[Error] TTS: {exc}", flush=True)
        finally:
            self._speaking = False
            if self.running:
                self._status("[🎙️ Listening...]")

    async def _mic_pump(self) -> None:
        assert self._mic_stream is not None
        assert self._dg_connection is not None
        while self.running:
            if self._speaking:
                await asyncio.sleep(0.05)
                continue
            data = await asyncio.to_thread(
                self._mic_stream.read,
                CHUNK_SIZE,
                False,
            )
            if self._dg_connection:
                await asyncio.to_thread(self._dg_connection.send, data)

    async def _run_deepgram_session(self) -> None:
        options = LiveOptions(
            model="nova-3-general",
            language="ar",
            punctuate=True,
            encoding="linear16",
            channels=CHANNELS,
            sample_rate=SAMPLE_RATE,
            interim_results=True,
        )
        connection = open_deepgram_listen(self.deepgram)
        self._dg_connection = connection

        connection.on(
            LiveTranscriptionEvents.Transcript,
            self._on_transcript_received,
        )

        started = await asyncio.to_thread(connection.start, options)
        if not started:
            raise RuntimeError("Deepgram live connection failed to start.")

        self._pa = pyaudio.PyAudio()
        self._mic_stream = self._pa.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=SAMPLE_RATE,
            input=True,
            frames_per_buffer=CHUNK_SIZE,
        )

        self._status("[🎙️ Listening...]")
        mic_task = asyncio.create_task(self._mic_pump())
        try:
            while self.running:
                await asyncio.sleep(0.1)
        finally:
            mic_task.cancel()
            try:
                await mic_task
            except asyncio.CancelledError:
                pass
            await asyncio.to_thread(connection.finish)

    def _cleanup_audio(self) -> None:
        for stream, close_fn in (
            (self._mic_stream, "stop_stream"),
            (self._out_stream, "stop_stream"),
        ):
            if stream is not None:
                try:
                    getattr(stream, close_fn)()
                    stream.close()
                except Exception:
                    pass
        self._mic_stream = None
        self._out_stream = None
        if self._pa is not None:
            try:
                self._pa.terminate()
            except Exception:
                pass
            self._pa = None
        self._dg_connection = None

    async def run(self) -> None:
        if not CHROMA_PATH.is_dir():
            raise FileNotFoundError(
                f"ChromaDB not found at {CHROMA_PATH}. Run fast_index.py first."
            )

        self.loop = asyncio.get_running_loop()
        self.running = True

        print("=" * 52)
        print("  Voice Real Estate Agent (Deepgram + Edge TTS)")
        print("  Speak in Arabic. Press Ctrl+C to exit.")
        print("=" * 52)

        try:
            await self._run_deepgram_session()
        finally:
            self.running = False
            self._cleanup_audio()
            print("\nBye!", flush=True)


async def main() -> int:
    try:
        await VoiceInterface().run()
        return 0
    except KeyboardInterrupt:
        print("\nBye!", flush=True)
        return 0
    except Exception as exc:
        print(f"\n[Fatal] {exc}", flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
