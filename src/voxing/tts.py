import queue
import threading
from collections.abc import Callable

import mlx.core as mx
import numpy as np
import sounddevice as sd

from voxing.chatterbox import ChatterboxTurboTTS
from voxing.chatterbox import load_model as _load_chatterbox

CHUNK_SAMPLES = 2400  # ~100ms at 24kHz

_DONE: object = object()


def load_tts(model_id: str) -> ChatterboxTurboTTS:
    """Download (if needed) and load a TTS model from HuggingFace Hub."""
    return _load_chatterbox(model_id)


def synthesize(model: ChatterboxTurboTTS, texts: str | list[str]) -> np.ndarray:
    """Generate speech audio from text, returning a numpy array at 24 kHz."""
    if isinstance(texts, str):
        texts = [texts]
    segments: list[np.ndarray] = []
    for text in texts:
        for result in model.generate(text):
            segments.append(np.array(result.audio))
    mx.clear_cache()
    return np.concatenate(segments) if segments else np.array([], dtype=np.float32)


def _drain_queue(audio_queue: queue.Queue[np.ndarray | object]) -> None:
    """Discard all remaining items from the queue."""
    while True:
        try:
            audio_queue.get_nowait()
        except queue.Empty:
            break


def _playback_consumer(
    audio_queue: queue.Queue[np.ndarray | object],
    stream: sd.OutputStream,
    on_chunk: Callable[[np.ndarray], None] | None,
    on_first_chunk: Callable[[], None] | None,
    error_bucket: list[BaseException],
    stop_event: threading.Event | None,
) -> None:
    """Drain audio chunks from the queue and write them to the sound stream."""
    first_chunk_fired = False
    try:
        while True:
            try:
                item = audio_queue.get(timeout=0.1)
            except queue.Empty:
                if stop_event is not None and stop_event.is_set():
                    break
                continue
            if item is _DONE:
                break
            if stop_event is not None and stop_event.is_set():
                break
            sub_chunk: np.ndarray = item  # type: ignore[assignment]
            stream.write(sub_chunk)
            if not first_chunk_fired:
                first_chunk_fired = True
                if on_first_chunk is not None:
                    on_first_chunk()
            if on_chunk is not None:
                on_chunk(sub_chunk[:, 0] if sub_chunk.ndim == 2 else sub_chunk)
    except Exception as exc:
        error_bucket.append(exc)


def synthesize_and_play(
    model: ChatterboxTurboTTS,
    texts: str | list[str],
    on_chunk: Callable[[np.ndarray], None] | None = None,
    on_first_chunk: Callable[[], None] | None = None,
    stop_event: threading.Event | None = None,
) -> None:
    """Generate speech and stream it to the audio device in real-time."""
    if isinstance(texts, str):
        texts = [texts]

    stream = sd.OutputStream(
        samplerate=model.sample_rate,
        channels=1,
        dtype="float32",
        blocksize=CHUNK_SAMPLES,
    )
    stream.start()

    audio_queue: queue.Queue[np.ndarray | object] = queue.Queue()
    error_bucket: list[BaseException] = []
    stopped = False

    consumer = threading.Thread(
        target=_playback_consumer,
        args=(audio_queue, stream, on_chunk, on_first_chunk, error_bucket, stop_event),
        daemon=True,
    )
    consumer.start()

    try:
        pending_chunk: np.ndarray | None = None
        for text in texts:
            if stop_event is not None and stop_event.is_set():
                stopped = True
                return
            for result in model.generate(text):
                if stop_event is not None and stop_event.is_set():
                    stopped = True
                    return
                audio = np.array(result.audio).astype(np.float32)
                if audio.ndim == 1:
                    audio = audio[:, np.newaxis]
                for i in range(0, len(audio), CHUNK_SAMPLES):
                    if stop_event is not None and stop_event.is_set():
                        stopped = True
                        return
                    if error_bucket:
                        return
                    if pending_chunk is not None:
                        audio_queue.put(pending_chunk)
                    pending_chunk = audio[i : i + CHUNK_SAMPLES]
        if pending_chunk is not None:
            if len(pending_chunk) < CHUNK_SAMPLES:
                pad_length = CHUNK_SAMPLES - len(pending_chunk)
                pending_chunk = np.pad(pending_chunk, ((0, pad_length), (0, 0)))
            audio_queue.put(pending_chunk)
    finally:
        if stopped:
            _drain_queue(audio_queue)
        audio_queue.put(_DONE)
        consumer.join(timeout=10.0)
        if stopped:
            stream.abort()
        else:
            stream.stop()
        stream.close()
        if error_bucket:
            raise error_bucket[0]
