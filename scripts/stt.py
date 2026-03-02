"""Test the STT pipeline with a live spectrogram visualisation.

Usage: uv run scripts/stt.py
"""

import threading
from collections import deque

import numpy as np
from rich.console import Group
from rich.live import Live
from rich.panel import Panel
from rich.text import Text

from voxing.config import Settings
from voxing.parakeet import load_model
from voxing.stt import RealtimeTranscriber
from voxing.tui.theme import FOREGROUND

_REFRESH_HZ = 30

# Spectrogram parameters
_SPEC_BANDS = 12  # frequency rows
_FREQ_LO = 80.0  # Hz — lowest band edge
_FREQ_HZ = 8000.0  # Hz — highest band edge (speech tops out ~4 kHz)
_DECAY = 0.92  # per-frame exponential decay for the rolling max normaliser
_BLOCKS = " ▁▂▃▄▅▆▇█"  # 9 levels: index 0 = silence, 8 = full


def _build_spec_state(sample_rate: int, chunk_samples: int) -> dict:
    """Pre-compute all fixed FFT artefacts so the hot path is allocation-free."""
    window = np.hanning(chunk_samples).astype(np.float32)
    freqs = np.fft.rfftfreq(chunk_samples, 1.0 / sample_rate)
    edges = np.logspace(np.log10(_FREQ_LO), np.log10(_FREQ_HZ), _SPEC_BANDS + 1)
    bin_edges = [int(np.searchsorted(freqs, e)) for e in edges]
    return {"window": window, "bin_edges": bin_edges, "rolling_max": 1e-6}


def _compute_column(chunk: np.ndarray, state: dict) -> np.ndarray:
    """Return a (_SPEC_BANDS,) float32 array of log-compressed band energies.

    Runs entirely before the lock is acquired — no allocation inside the lock.
    """
    mag = np.abs(np.fft.rfft(chunk * state["window"]))
    edges = state["bin_edges"]
    bands = np.array(
        [float(np.mean(mag[edges[i] : edges[i + 1]])) for i in range(_SPEC_BANDS)],
        dtype=np.float32,
    )
    # Update rolling max with exponential decay so display stays responsive
    peak = float(bands.max())
    state["rolling_max"] = max(state["rolling_max"] * _DECAY, peak, 1e-6)
    bands /= state["rolling_max"]
    # Log-compress for perceptual linearity
    bands = np.sqrt(np.log1p(bands * 20.0) / np.log1p(20.0))
    return np.clip(bands, 0.0, 1.0)


def _render_spectrogram(columns: list[np.ndarray], height: int) -> str:
    """Return a Rich-markup multi-line string for the rolling spectrogram.

    Rows: frequency (high at top, low at bottom).
    Columns: time (oldest at left, newest at right).
    Each cell is one of _BLOCKS coloured by magnitude.
    """
    if not columns:
        return "\n".join(" " for _ in range(height))
    lines: list[str] = []
    for row in range(height - 1, -1, -1):
        chars: list[str] = []
        for col in columns:
            v = float(col[row]) if row < len(col) else 0.0
            level = min(int(v * 8), 8)
            if level == 0:
                chars.append(" ")
            else:
                # Interpolate colour brightness: dim PRIMARY at level 1, full at 8
                brightness = int(55 + level * 25)  # 80..255
                chars.append(f"[#{brightness:02x}b4fa]{_BLOCKS[level]}[/]")
        lines.append("".join(chars))
    return "\n".join(lines)


def _make_renderable(
    columns: deque[np.ndarray],
    columns_lock: threading.Lock,
    transcription: list[str],
    width: int,
) -> Group:
    """Build the Rich renderable for one Live refresh."""
    with columns_lock:
        snap = list(columns)
    spec_str = _render_spectrogram(snap, _SPEC_BANDS)
    spec_text = Text.from_markup(spec_str)
    if _paused.is_set():
        text = "[dim]Paused — press Enter to resume[/dim]"
    elif transcription[0]:
        text = transcription[0]
    else:
        text = "[dim]Listening...[/dim]"
    return Group(
        spec_text,
        Panel(
            Text.from_markup(text),
            title="transcription",
            border_style=f"dim {FOREGROUND}",
        ),
    )


settings = Settings()

print("Loading model...")
model = load_model(settings.model_id)
print("Model loaded.\n")

input("Press Enter to start recording... ")

# One spectral column per audio chunk; keep enough to fill the terminal width.
_MAX_COLUMNS = 300
columns: deque[np.ndarray] = deque(maxlen=_MAX_COLUMNS)
columns_lock = threading.Lock()
transcription: list[str] = [""]
_stop_event = threading.Event()
_paused = threading.Event()

# Build FFT artefacts once — chunk_samples = chunk_duration * sample_rate = 0.1 * 16000
_chunk_samples = int(settings.chunk_duration * settings.sample_rate)
_spec_state = _build_spec_state(settings.sample_rate, _chunk_samples)


def _on_chunk(chunk: np.ndarray) -> None:
    """Compute the spectrum column for this chunk and append it to the ring buffer."""
    if _paused.is_set():
        return
    col = _compute_column(chunk, _spec_state)  # FFT outside the lock
    with columns_lock:
        columns.append(col)


def _refresh_loop(live: Live) -> None:
    """Push spectrogram redraws at a fixed rate, independent of transcription."""
    interval = 1.0 / _REFRESH_HZ
    while not _stop_event.is_set():
        width = max(10, live.console.width)
        live.update(_make_renderable(columns, columns_lock, transcription, width=width))
        _stop_event.wait(interval)


def _stdin_loop() -> None:
    """Toggle pause/resume on each Enter keypress."""
    while not _stop_event.is_set():
        try:
            input()
        except EOFError:
            break
        if _paused.is_set():
            _paused.clear()
        else:
            _paused.set()
            with columns_lock:
                columns.clear()


print("Press Enter to pause/resume, Ctrl+C to stop\n")

with (
    Live(refresh_per_second=_REFRESH_HZ, screen=False) as live,
    RealtimeTranscriber(model, settings, on_chunk=_on_chunk) as transcriber,
):
    threading.Thread(target=_refresh_loop, args=(live,), daemon=True).start()
    threading.Thread(target=_stdin_loop, daemon=True).start()

    try:
        for text in transcriber:
            transcription[0] = text
    except KeyboardInterrupt:
        transcriber.stop()

    _stop_event.set()
print(f"\nFinal: {transcription[0]}")
