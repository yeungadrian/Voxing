"""Audio recording and playback via sounddevice."""

import asyncio
from collections.abc import Iterable
from functools import partial

import numpy as np
import sounddevice as sd

from vox.config import settings


def _rms(audio: np.ndarray) -> float:
    """Calculate root mean square of audio signal."""
    return float(np.sqrt(np.mean(audio**2)))


def _normalize_audio(audio: np.ndarray) -> np.ndarray:
    """Normalize audio to prevent clipping."""
    audio = audio.flatten().astype(np.float32)
    max_val = np.abs(audio).max()
    if max_val > 0:
        audio = audio / max_val * 0.9
    return audio.reshape(-1, 1)


def _record_blocking(
    silence_duration: float = 0.7,
    max_duration: float = 10.0,
) -> np.ndarray | None:
    """Record audio until silence is detected."""
    chunk_duration = 0.1
    chunk_samples = int(settings.audio_sample_rate * chunk_duration)
    silence_chunks_needed = int(silence_duration / chunk_duration)
    max_chunks = int(max_duration / chunk_duration)

    audio_chunks: list[np.ndarray] = []
    silence_count = 0
    recording = False

    with sd.InputStream(
        samplerate=settings.audio_sample_rate, channels=1, dtype=np.float32
    ) as stream:
        while len(audio_chunks) < max_chunks:
            chunk, _ = stream.read(chunk_samples)
            has_voice = _rms(chunk) > settings.silence_threshold

            if not recording:
                if has_voice:
                    recording = True
                    audio_chunks.append(chunk)
                    silence_count = 0
                continue

            audio_chunks.append(chunk)

            if has_voice:
                silence_count = 0
            else:
                silence_count += 1
                if silence_count >= silence_chunks_needed:
                    break

    if not audio_chunks:
        return None
    return np.concatenate(audio_chunks, axis=0).flatten()


def _play_stream(chunks: Iterable[np.ndarray], sample_rate: int) -> None:
    """Play an iterable of audio chunks through the speakers."""
    with sd.OutputStream(
        samplerate=sample_rate,
        channels=1,
        dtype=np.float32,
        blocksize=4096,
    ) as stream:
        for chunk in chunks:
            audio = np.array(chunk, dtype=np.float32)
            stream.write(_normalize_audio(audio))


async def record() -> np.ndarray | None:
    """Record a short voice command."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None, partial(_record_blocking, silence_duration=0.7, max_duration=10.0)
    )


async def record_long() -> np.ndarray | None:
    """Record extended speech for transcription."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None, partial(_record_blocking, silence_duration=3.0, max_duration=180.0)
    )


async def play_stream(chunks: Iterable[np.ndarray], sample_rate: int) -> None:
    """Play audio chunks asynchronously."""
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, partial(_play_stream, chunks, sample_rate))
