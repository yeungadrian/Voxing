import queue
import threading
from collections.abc import Callable
from dataclasses import dataclass, field

import mlx.core as mx
import numpy as np
import sounddevice as sd

from voxing.chatterbox import ChatterboxTurboTTS, load_tts_model

_VIS_CHUNK_SECONDS = 0.1  # 100 ms per visualisation chunk


@dataclass(slots=True)
class _PlaybackState:
    buf: np.ndarray | None = None
    offset: int = 0
    done: threading.Event = field(default_factory=threading.Event)
    samples_played: int = 0


def load_tts(model_id: str) -> ChatterboxTurboTTS:
    """Download (if needed) and load a TTS model from HuggingFace Hub."""
    return load_tts_model(model_id)


def synthesize(model: ChatterboxTurboTTS, text: str) -> np.ndarray:
    """Generate speech audio from text, returning a numpy array at 24 kHz."""
    segments: list[np.ndarray] = []
    for result in model.generate(text):
        segments.append(np.array(result.audio))
    mx.clear_cache()
    return np.concatenate(segments) if segments else np.array([], dtype=np.float32)


def synthesize_and_play(
    model: ChatterboxTurboTTS,
    text: str,
    on_chunk: Callable[[np.ndarray], None] | None = None,
    on_first_chunk: Callable[[], None] | None = None,
    stop_event: threading.Event | None = None,
) -> None:
    """Generate speech and stream it to the audio device in real-time."""
    audio_queue: queue.Queue[np.ndarray | None] = queue.Queue()
    state = _PlaybackState()
    vis_chunk_samples = int(model.sample_rate * _VIS_CHUNK_SECONDS)
    viz_buf = np.empty(0, dtype=np.float32)
    cursor = 0

    def _is_stopped() -> bool:
        """Check whether synthesis has been cancelled."""
        return stop_event is not None and stop_event.is_set()

    def _audio_callback(
        outdata: np.ndarray,
        frames: int,
        _time: object,
        _status: object,
    ) -> None:
        """Fill output buffer from queue, padding with silence."""
        filled = 0
        while filled < frames:
            if state.buf is not None and state.offset < len(state.buf):
                take = min(len(state.buf) - state.offset, frames - filled)
                outdata[filled : filled + take] = state.buf[
                    state.offset : state.offset + take
                ]
                state.offset += take
                filled += take
                continue
            # Current buf exhausted, try next item
            state.buf = None
            state.offset = 0
            try:
                item = audio_queue.get_nowait()
            except queue.Empty:
                break
            if item is None:
                state.done.set()
                break
            state.buf = item
            state.offset = 0
        state.samples_played += filled
        if filled < frames:
            outdata[filled:] = 0.0

    def _dispatch_viz() -> None:
        """Send fixed-size chunks up to the playback cursor."""
        nonlocal cursor
        if on_chunk is None:
            return
        target = min(state.samples_played, len(viz_buf))
        while cursor + vis_chunk_samples <= target:
            on_chunk(viz_buf[cursor : cursor + vis_chunk_samples])
            cursor += vis_chunk_samples

    first_chunk_sent = False
    try:
        with sd.OutputStream(
            samplerate=model.sample_rate,
            channels=1,
            dtype="float32",
            blocksize=vis_chunk_samples,
            callback=_audio_callback,
        ):
            if _is_stopped():
                return
            for result in model.generate(text):
                if _is_stopped():
                    return
                audio = np.array(result.audio).astype(np.float32)
                if audio.ndim == 1:
                    audio = audio[:, np.newaxis]
                if not first_chunk_sent:
                    first_chunk_sent = True
                    if on_first_chunk is not None:
                        on_first_chunk()
                audio_queue.put(audio)
                # Accumulate 1-D audio for visualisation dispatch
                viz_buf = np.append(viz_buf, audio[:, 0] if audio.ndim > 1 else audio)
                _dispatch_viz()
            audio_queue.put(None)
            # Wait for callback to finish playing
            while not state.done.is_set():
                if _is_stopped():
                    return
                _dispatch_viz()
                state.done.wait(timeout=0.05)
            # Flush remaining samples
            if on_chunk is not None and cursor < len(viz_buf):
                on_chunk(viz_buf[cursor:])
                cursor = len(viz_buf)
    finally:
        mx.clear_cache()
