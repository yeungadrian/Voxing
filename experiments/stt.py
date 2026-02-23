# STT — recording, VAD, and real-time transcription
#
# Loop pattern:
#   1. Capture audio continuously into a buffer (CHUNK_DURATION chunks).
#   2. Every SPECULATIVE_INTERVAL_CHUNKS chunks, speculatively transcribe for display.
#   3. When silence is detected after MIN_COMMIT_SECS, or the buffer hits
#      MAX_BUFFER_SECS, commit: finalize text, reset buffer, and continue.
#
# Run: uv run experiments/stt.py

import dataclasses
import threading

import mlx.core as mx
import numpy as np
import sounddevice as sd
from parakeet import Model, load_model
from rich.console import Console
from rich.live import Live
from rich.text import Text

SAMPLE_RATE = 16_000
MIN_AUDIO_SECS = 0.5      # minimum buffer length before attempting any transcription
MIN_COMMIT_SECS = 7.5     # minimum audio length before silence can trigger a commit
SILENCE_DURATION = 0.3     # seconds of tail silence that triggers a commit
MIN_SILENCE_DURATION = 0.05  # minimum silence window near the buffer cap
SILENCE_THRESHOLD = 0.06   # RMS below this is considered silence
MAX_BUFFER_SECS = 30      # hard cap — commit regardless of silence
CHUNK_DURATION = 0.1      # capture granularity (seconds)
SPECULATIVE_INTERVAL_CHUNKS = 7  # run a speculative pass every N chunks (≈ 0.7 s)
MODEL_NAME = "mlx-community/parakeet-tdt-0.6b-v3"

console = Console()


def rms(audio: np.ndarray) -> float:
    """Root-mean-square energy of an audio array."""
    return float(np.sqrt(np.mean(audio**2)))


def _infer(model: Model, audio: np.ndarray) -> str:
    """Run model inference and return stripped text."""
    result = model.generate(mx.array(audio))
    mx.metal.clear_cache()
    return result.text.strip()


def adaptive_silence_duration(fill: int) -> float:
    """Linearly shrink the silence window as the buffer approaches MAX_BUFFER_SECS.

    Returns SILENCE_DURATION before MIN_COMMIT_SECS, then interpolates down to
    MIN_SILENCE_DURATION at MAX_BUFFER_SECS so that commits happen more eagerly
    near the hard cap.
    """
    min_fill = int(SAMPLE_RATE * MIN_COMMIT_SECS)
    max_fill = int(SAMPLE_RATE * MAX_BUFFER_SECS)
    if fill <= min_fill:
        return SILENCE_DURATION
    t = min((fill - min_fill) / (max_fill - min_fill), 1.0)
    return SILENCE_DURATION + t * (MIN_SILENCE_DURATION - SILENCE_DURATION)


def should_commit(audio: np.ndarray) -> bool:
    """True if silence detected after sufficient speech, or buffer cap exceeded."""
    if len(audio) >= SAMPLE_RATE * MAX_BUFFER_SECS:
        return True  # hard cap — commit regardless of silence
    if len(audio) < SAMPLE_RATE * MIN_COMMIT_SECS:
        return False  # too short; any breath pause would cause a false commit
    silence_secs = adaptive_silence_duration(len(audio))
    tail = audio[-int(SAMPLE_RATE * silence_secs):]
    return rms(tail) < SILENCE_THRESHOLD


def display(live: Live, committed: str, speculative: str) -> None:
    """Update the live display: committed text in green, speculative text dimmed."""
    output = Text()
    output.append(committed, style="green")
    if committed and speculative:
        output.append(" ")
    output.append(speculative, style="dim")
    live.update(output)


class _AudioBuffer:
    """Thread-safe pre-allocated audio buffer shared between capture and decode threads.

    Pre-allocated flat array (MAX_BUFFER_SECS x SAMPLE_RATE) -- zero allocation
    per recording iteration vs a growing list.  A ring buffer would work too,
    but is only needed for always-on "last N seconds" patterns like wake-word
    detection.

    A Condition is used in place of a plain Lock so that append() can notify
    the decode thread after each write, replacing polling with event-driven
    wakeups.  The condition is held only during buffer copies and resets --
    never during model inference -- so _capture_loop is blocked for at most
    a few microseconds per call.
    """

    def __init__(self) -> None:
        self._data = np.zeros(MAX_BUFFER_SECS * SAMPLE_RATE, dtype=np.float32)
        self._cond = threading.Condition()
        self._fill = 0    # samples written so far
        self._chunks = 0  # chunks written since last commit

    @property
    def fill(self) -> int:
        """Samples written so far; GIL-atomic read -- one chunk behind is fine."""
        return self._fill

    @property
    def chunks(self) -> int:
        """Chunks written since last commit; GIL-atomic read."""
        return self._chunks

    def append(self, chunk: np.ndarray) -> None:
        """Write a mono chunk into the buffer; silently drops if buffer is full."""
        n = chunk.shape[0]
        with self._cond:
            end = self._fill + n
            if end <= len(self._data):
                self._data[self._fill:end] = chunk[:, 0]
                self._fill = end
                self._chunks += 1
                self._cond.notify_all()

    def snapshot(self) -> np.ndarray:
        """Return a copy of the filled region for speculative transcription."""
        with self._cond:
            return self._data[: self._fill].copy()

    def commit(self) -> np.ndarray:
        """Return filled audio and reset buffer for reuse; empty array if unfilled."""
        with self._cond:
            if self._fill == 0:
                return np.empty(0, dtype=np.float32)
            audio = self._data[: self._fill].copy()
            self._fill = 0
            self._chunks = 0
        return audio

    def wait(self) -> None:
        """Block until a chunk is appended or wake() is called."""
        with self._cond:
            self._cond.wait()

    def wake(self) -> None:
        """Unblock any thread waiting in wait()."""
        with self._cond:
            self._cond.notify_all()


@dataclasses.dataclass
class _State:
    committed: str = ""
    speculative: str = ""


# --- run_realtime() ----------------------------------------------------------
# Threading model: two threads, one shared _AudioBuffer.
#   _capture_loop  -- writes CHUNK_DURATION chunks via buf.append(), which
#                     notifies _decode_loop after each write.
#   _decode_loop   -- blocks in buf.wait() between passes; woken by each
#                     append() or by buf.wake() on shutdown.
def _capture_loop(buf: _AudioBuffer, stop: threading.Event) -> None:
    """Open audio stream and write fixed-size chunks into the audio buffer."""
    chunk_samples = int(SAMPLE_RATE * CHUNK_DURATION)
    with sd.InputStream(
        samplerate=SAMPLE_RATE, channels=1, dtype=np.float32
    ) as stream:
        while not stop.is_set():
            # Blocking read releases the GIL while waiting, so the
            # transcription thread runs freely -- no busy-polling.
            chunk, _ = stream.read(chunk_samples)
            buf.append(chunk)


def _decode_loop(
    buf: _AudioBuffer,
    state: _State,
    live: Live,
    model: Model,
    stop: threading.Event,
) -> None:
    """Speculative-then-commit transcription loop.

    Every SPECULATIVE_INTERVAL_CHUNKS chunks:
      - Transcribe the growing buffer (speculative).
      - If silence or buffer cap detected, commit and reset instead.
    """
    next_pass_chunk = SPECULATIVE_INTERVAL_CHUNKS

    while not stop.is_set():
        buf.wait()
        if buf.fill < SAMPLE_RATE * MIN_AUDIO_SECS or buf.chunks < next_pass_chunk:
            continue

        audio = buf.snapshot()

        if should_commit(audio):
            # --- Commit path ---
            audio = buf.commit()  # resets buf.chunks to 0
            if len(audio) == 0:
                continue
            text = _infer(model, audio)
            if text:
                state.committed = f"{state.committed} {text}".strip()
            state.speculative = ""
        else:
            # --- Speculative path ---
            state.speculative = _infer(model, audio)

        next_pass_chunk = buf.chunks + SPECULATIVE_INTERVAL_CHUNKS
        display(live, state.committed, state.speculative)


def run_realtime(model: Model) -> str:
    """Record and transcribe in real time; return full text when Enter is pressed."""
    buf = _AudioBuffer()
    state = _State()
    stop_recording = threading.Event()
    stop_transcribing = threading.Event()

    with Live(Text(""), console=console, refresh_per_second=10) as live:
        record_t = threading.Thread(
            target=_capture_loop, args=(buf, stop_recording), daemon=True
        )
        transcribe_t = threading.Thread(
            target=_decode_loop,
            args=(buf, state, live, model, stop_transcribing),
            daemon=True,
        )
        record_t.start()
        transcribe_t.start()

        console.print("Recording... (press Enter to stop)")
        input()

        # Stop capturing first so the buffer has a stable final state.
        stop_recording.set()
        record_t.join()

        # Wake the decode loop in case it's blocked in wait() (no more
        # chunks will arrive), then wait for any in-flight inference to finish.
        stop_transcribing.set()
        buf.wake()
        transcribe_t.join()

    # Final pass: transcribe any audio that was captured but not yet committed
    # (e.g. a short trailing segment that never triggered should_commit).
    remaining = buf.snapshot()
    if len(remaining) >= SAMPLE_RATE * MIN_AUDIO_SECS:
        text = _infer(model, remaining)
        if text:
            state.speculative = text

    return f"{state.committed} {state.speculative}".strip()


def main() -> None:
    with console.status("Loading model..."):
        model = load_model(MODEL_NAME)
    console.print("Model loaded.")

    try:
        while True:
            console.input("\nPress Enter to start real-time recording... ")
            text = run_realtime(model)
            console.print(f"\nFinal: [bold]{text}[/bold]")
    except KeyboardInterrupt:
        console.print("\nExiting.")


if __name__ == "__main__":
    main()
