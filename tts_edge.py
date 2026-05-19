"""
Edge TTS — sole speech engine. Sync speak() for CLI; PCM synthesis for WebSocket UI.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import edge_tts
import miniaudio

# Egyptian Arabic neural voice
DEFAULT_VOICE = "ar-EG-SalmaNeural"
SAMPLE_RATE = 16000


async def _save_mp3_async(text: str, path: Path, voice: str = DEFAULT_VOICE) -> None:
    communicate = edge_tts.Communicate(text.strip(), voice)
    await communicate.save(str(path))


def _mp3_bytes_async(text: str, voice: str = DEFAULT_VOICE) -> bytes:
    async def _collect() -> bytes:
        communicate = edge_tts.Communicate(text.strip(), voice)
        parts: list[bytes] = []
        async for chunk in communicate.stream():
            if chunk.get("type") == "audio":
                parts.append(chunk["data"])
        return b"".join(parts)

    return asyncio.run(_collect())


def mp3_to_pcm_s16le_16k(mp3_bytes: bytes) -> bytes:
    if not mp3_bytes:
        return b""
    decoded = miniaudio.decode(
        mp3_bytes,
        output_format=miniaudio.SampleFormat.SIGNED16,
        nchannels=1,
        sample_rate=SAMPLE_RATE,
    )
    return bytes(decoded.samples)


def synthesize_pcm(text: str, voice: str = DEFAULT_VOICE) -> bytes:
    """edge-tts → PCM @ 16 kHz for browser WebSocket playback."""
    mp3 = _mp3_bytes_async(text, voice)
    return mp3_to_pcm_s16le_16k(mp3)


def _play_audio_file(path: Path) -> None:
    """Platform default audio player."""
    path_str = str(path.resolve())
    if sys.platform == "win32":
        os.startfile(path_str)  # type: ignore[attr-defined]
    elif sys.platform == "darwin":
        subprocess.run(["afplay", path_str], check=False)
    else:
        subprocess.run(["xdg-open", path_str], check=False)


def speak(text: str, voice: str = DEFAULT_VOICE) -> None:
    """
    Synchronous TTS: generate temp MP3 via edge-tts, play, delete file.
    Every spoken response must call this (or synthesize_pcm for streaming UI).
    """
    spoken = (text or "").strip()
    if not spoken:
        return
    tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
    tmp_path = Path(tmp.name)
    tmp.close()
    try:
        asyncio.run(_save_mp3_async(spoken, tmp_path, voice))
        _play_audio_file(tmp_path)
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
