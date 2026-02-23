import queue
import threading
from collections.abc import Iterator

import mlx.core as mx
import numpy as np
import sounddevice as sd

from voxing.parakeet import ParakeetTDT

SAMPLE_RATE = 16_000
MIN_AUDIO_SECS = 1.0
SILENCE_DURATION = 1.5
SILENCE_THRESHOLD = 0.01
MAX_BUFFER_SECS = 30
CHUNK_DURATION = 0.1
SPECULATIVE_INTERVAL_SECS = 0.7

CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_DURATION)
MIN_AUDIO_SAMPLES = int(SAMPLE_RATE * MIN_AUDIO_SECS)
MAX_BUFFER_SAMPLES = SAMPLE_RATE * MAX_BUFFER_SECS
SILENCE_SAMPLES = int(SAMPLE_RATE * SILENCE_DURATION)
SPECULATIVE_INTERVAL_CHUNKS = int(SPECULATIVE_INTERVAL_SECS / CHUNK_DURATION)

MODEL_NAME = "mlx-community/parakeet-tdt-0.6b-v3"


def _rms(audio: np.ndarray) -> float:
    """Root-mean-square energy of an audio array."""
    return float(np.sqrt(np.mean(audio**2)))


def _transcribe(model: ParakeetTDT, audio: np.ndarray) -> str:
    """Transcribe audio to text."""
    result = model.generate(mx.array(audio))
    mx.clear_cache()
    return result.text.strip()


def should_commit(audio: np.ndarray) -> bool:
    """True if silence detected after sufficient speech, or buffer cap exceeded."""
    if len(audio) >= MAX_BUFFER_SAMPLES:
        return True
    if len(audio) < MIN_AUDIO_SAMPLES:
        return False
    tail = audio[-SILENCE_SAMPLES:]
    return _rms(tail) < SILENCE_THRESHOLD


def _capture_loop(
    chunk_q: queue.Queue[np.ndarray | None], stop: threading.Event
) -> None:
    """Read audio from mic; put None sentinel when stop is set."""
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype=np.float32) as stream:
        while not stop.is_set():
            chunk, _ = stream.read(CHUNK_SAMPLES)
            chunk_q.put(chunk[:, 0])
    chunk_q.put(None)


def run_realtime(
    model: ParakeetTDT,
    *,
    stop_event: threading.Event,
) -> Iterator[str]:
    """Record and transcribe in real time; yields current text on each update."""
    chunk_q: queue.Queue[np.ndarray | None] = queue.Queue()
    threading.Thread(
        target=_capture_loop, args=(chunk_q, stop_event), daemon=True
    ).start()

    committed = ""
    speculative = ""
    buffer: list[np.ndarray] = []
    buffer_samples = 0
    chunks_since_decode = 0

    while (chunk := chunk_q.get()) is not None:
        buffer.append(chunk)
        buffer_samples += len(chunk)
        chunks_since_decode += 1

        # Accumulate until minimum audio available
        if buffer_samples < MIN_AUDIO_SAMPLES:
            continue

        # Decode when interval elapsed or buffer maxed
        buffer_maxed = buffer_samples >= MAX_BUFFER_SAMPLES
        if not buffer_maxed and chunks_since_decode < SPECULATIVE_INTERVAL_CHUNKS:
            continue

        audio = np.concatenate(buffer)
        chunks_since_decode = 0

        if should_commit(audio):
            # Commit: finalize text, reset buffer
            # TODO: scan backward for last silence window to avoid mid-word splits
            text = _transcribe(model, audio)
            if text:
                committed = f"{committed} {text}".strip()
            buffer = []
            buffer_samples = 0
            speculative = ""
        else:
            # Speculative: keep buffer, update working text
            speculative = _transcribe(model, audio)

        yield f"{committed} {speculative}".strip()

    # Flush remaining audio on stop
    if buffer:
        audio = np.concatenate(buffer)
        if len(audio) >= MIN_AUDIO_SAMPLES:
            text = _transcribe(model, audio)
            if text:
                committed = f"{committed} {text}".strip()
    if committed:
        yield committed
