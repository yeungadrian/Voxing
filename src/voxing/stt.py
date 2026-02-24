import queue
import threading
from collections.abc import Iterator

import mlx.core as mx
import numpy as np
import sounddevice as sd

from voxing.parakeet import ParakeetTDT

SAMPLE_RATE = 16_000
MIN_AUDIO_SECS = 1.5
SILENCE_DURATION = 3.0
SILENCE_THRESHOLD = 0.01
MAX_BUFFER_SECS = 45
CHUNK_DURATION = 0.1
DRAFT_INTERVAL_SECS = 0.5

CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_DURATION)
MIN_AUDIO_SAMPLES = int(SAMPLE_RATE * MIN_AUDIO_SECS)
MAX_BUFFER_SAMPLES = SAMPLE_RATE * MAX_BUFFER_SECS
SILENCE_SAMPLES = int(SAMPLE_RATE * SILENCE_DURATION)
DRAFT_INTERVAL_CHUNKS = int(DRAFT_INTERVAL_SECS / CHUNK_DURATION)

MODEL_NAME = "mlx-community/parakeet-tdt-0.6b-v3"


def _detect_silence(audio: np.ndarray) -> bool:
    """Root-mean-square energy of an audio array."""
    return np.sqrt(np.mean(audio**2)) < SILENCE_THRESHOLD


def _transcribe(model: ParakeetTDT, audio: np.ndarray) -> str:
    """Transcribe audio to text."""
    result = model.generate(mx.array(audio))
    mx.clear_cache()
    return result.text.strip()


def is_utterance_complete(audio: np.ndarray) -> bool:
    """True if silence detected after sufficient speech, or buffer cap exceeded."""
    if len(audio) >= MAX_BUFFER_SAMPLES:
        return True
    if len(audio) < MIN_AUDIO_SAMPLES:
        return False
    if len(audio) >= SILENCE_SAMPLES:
        tail = audio[-SILENCE_SAMPLES:]
        return _detect_silence(tail)
    return False


def _stream_mic_chunks(
    chunk_queue: queue.Queue[np.ndarray | None], stop_event: threading.Event
) -> None:
    """Read audio from mic; put None sentinel when stop is set."""
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype=np.float32) as stream:
        while not stop_event.is_set():
            chunk, _ = stream.read(CHUNK_SAMPLES)
            chunk_queue.put(chunk[:, 0])
    chunk_queue.put(None)


class RealtimeTranscriber:
    def __init__(self, model: ParakeetTDT) -> None:
        self._model = model
        self._chunk_queue: queue.Queue[np.ndarray | None] = queue.Queue()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def stop(self) -> None:
        """Signal the capture thread to stop."""
        self._stop_event.set()

    def __enter__(self) -> "RealtimeTranscriber":
        self._thread = threading.Thread(
            target=_stream_mic_chunks, args=(self._chunk_queue, self._stop_event)
        )
        self._thread.start()
        return self

    def __exit__(self, *args: object) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join()

    def __iter__(self) -> Iterator[str]:
        confirmed = ""
        draft = ""
        buffer: list[np.ndarray] = []
        buffer_samples = 0
        chunks_since_decode = 0

        while (chunk := self._chunk_queue.get()) is not None:
            buffer.append(chunk)
            buffer_samples += len(chunk)
            chunks_since_decode += 1

            # Accumulate until minimum audio available
            if buffer_samples < MIN_AUDIO_SAMPLES:
                continue

            # Decode when interval elapsed or buffer full
            buffer_full = buffer_samples >= MAX_BUFFER_SAMPLES
            if not buffer_full and chunks_since_decode < DRAFT_INTERVAL_CHUNKS:
                continue

            audio = np.concatenate(buffer)
            chunks_since_decode = 0

            if is_utterance_complete(audio):
                # Confirm: finalize text, reset buffer
                # TODO: scan backward for best silence window to avoid mid-word splits
                text = _transcribe(self._model, audio)
                if text:
                    confirmed = f"{confirmed} {text}".strip()
                buffer = []
                buffer_samples = 0
                draft = ""
            else:
                # Draft: keep buffer, update working text
                draft = _transcribe(self._model, audio)

            yield f"{confirmed} {draft}".strip()

        # Flush remaining audio on stop
        if buffer:
            audio = np.concatenate(buffer)
            if len(audio) >= MIN_AUDIO_SAMPLES:
                text = _transcribe(self._model, audio)
                if text:
                    confirmed = f"{confirmed} {text}".strip()
        if confirmed:
            yield confirmed
