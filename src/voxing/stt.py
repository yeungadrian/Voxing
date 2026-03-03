import queue
import threading
from collections.abc import Callable, Iterator

import mlx.core as mx
import numpy as np
import sounddevice as sd

from voxing.config import Settings
from voxing.parakeet import ParakeetTDT


def _transcribe(model: ParakeetTDT, audio: np.ndarray) -> str:
    """Transcribe audio to text."""
    result = model.generate(mx.array(audio))
    mx.clear_cache()
    return result.text.strip()


def _stream_mic_chunks(
    chunk_queue: queue.Queue[np.ndarray | None],
    stop_event: threading.Event,
    sample_rate: int,
    chunk_samples: int,
    on_chunk: Callable[[np.ndarray], None] | None = None,
) -> None:
    """Read audio from mic; put None sentinel when stop is set or on error."""
    try:
        with sd.InputStream(
            samplerate=sample_rate, channels=1, dtype=np.float32
        ) as stream:
            while not stop_event.is_set():
                chunk, _ = stream.read(chunk_samples)
                mono = chunk[:, 0]
                chunk_queue.put(mono)
                if on_chunk is not None:
                    on_chunk(mono)
    finally:
        # Always send sentinel so __iter__ is never left blocking indefinitely
        chunk_queue.put(None)


class RealtimeTranscriber:
    def __init__(
        self,
        model: ParakeetTDT,
        settings: Settings,
        on_chunk: Callable[[np.ndarray], None] | None = None,
    ) -> None:
        self._model = model
        self._settings = settings
        self._on_chunk = on_chunk
        self._chunk_samples = int(settings.sample_rate * settings.chunk_duration)
        self._min_audio_samples = int(settings.sample_rate * settings.min_audio_secs)
        self._max_buffer_samples = int(settings.sample_rate * settings.max_buffer_secs)
        self._silence_samples = int(settings.sample_rate * settings.silence_duration)
        self._draft_interval_chunks = int(
            settings.draft_interval_secs / settings.chunk_duration
        )
        self._chunk_queue: queue.Queue[np.ndarray | None] = queue.Queue()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def stop(self) -> None:
        """Signal the capture thread to stop."""
        self._stop_event.set()

    def _detect_silence(self, audio: np.ndarray) -> bool:
        """Return True if RMS energy of audio is below silence threshold."""
        return bool(np.sqrt(np.mean(audio**2)) < self._settings.silence_threshold)

    def _is_utterance_complete(self, audio: np.ndarray) -> bool:
        """True if silence detected after sufficient speech, or buffer cap exceeded."""
        if len(audio) >= self._max_buffer_samples:
            return True
        if len(audio) < self._min_audio_samples:
            return False
        if len(audio) >= self._silence_samples:
            tail = audio[-self._silence_samples :]
            return self._detect_silence(tail)
        return False

    def __enter__(self) -> "RealtimeTranscriber":
        self._thread = threading.Thread(
            target=_stream_mic_chunks,
            args=(
                self._chunk_queue,
                self._stop_event,
                self._settings.sample_rate,
                self._chunk_samples,
                self._on_chunk,
            ),
        )
        self._thread.start()
        return self

    def __exit__(self, *args: object) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join()

    def __iter__(self) -> Iterator[str]:
        # TODO: Separate draft vs confirmed so we can display difference to user.
        confirmed = ""
        draft = ""
        buffer: list[np.ndarray] = []
        buffer_samples = 0
        chunks_since_decode = 0

        while True:
            chunk = self._chunk_queue.get()
            if chunk is None:
                break
            buffer.append(chunk)
            buffer_samples += len(chunk)
            chunks_since_decode += 1

            # Stop is set: drain remaining queued chunks then flush once at the end
            if self._stop_event.is_set():
                continue

            # Accumulate until minimum audio available
            if buffer_samples < self._min_audio_samples:
                continue

            # Decode when interval elapsed or buffer full
            buffer_full = buffer_samples >= self._max_buffer_samples
            if not buffer_full and chunks_since_decode < self._draft_interval_chunks:
                continue

            audio = np.concatenate(buffer)
            chunks_since_decode = 0

            if self._is_utterance_complete(audio):
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
            if len(audio) >= self._min_audio_samples:
                text = _transcribe(self._model, audio)
                if text:
                    confirmed = f"{confirmed} {text}".strip()
        if confirmed:
            yield confirmed
