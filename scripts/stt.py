"""Test the STT pipeline with a live audio visualisation.

Usage: uv run scripts/stt.py

Set _VIZ_MODE to "waveform" or "spectrogram" to switch visualisations.
"""

import threading
from collections import deque
from typing import Literal

import numpy as np
from rich.console import Group
from rich.live import Live
from rich.panel import Panel
from rich.text import Text

from voxing.config import Settings
from voxing.parakeet import load_model
from voxing.stt import RealtimeTranscriber
from voxing.tui.theme import FOREGROUND, PRIMARY
from voxing.viz import (
    BLOCKS,
    BRAILLE_BASE,
    SPEC_BANDS,
    SUB_ROW_BITS,
    VIZ_HEIGHT,
    SpectrogramState,
    bar_columns,
    build_spec_state,
    compute_column,
    peaks,
)

_VIZ_MODE: Literal["waveform", "spectrogram"] = "spectrogram"
_REFRESH_HZ = 30


def _render_waveform(samples: deque[float], width: int, height: int) -> str:
    """Return a Rich-markup multi-line string for a braille waveform."""
    amps = peaks(np.array(samples, dtype=np.float32))
    columns = bar_columns(width)
    total_dots = height * 4
    center = total_dots / 2.0
    lines: list[str] = []
    for y in range(height):
        row_chars: list[str] = []
        for bar_idx in columns:
            if bar_idx is None:
                row_chars.append(" ")
                continue
            amp = amps[bar_idx]
            half_extent = amp * center
            top = center - half_extent
            bottom = center + half_extent
            bits = 0
            for sub in range(4):
                dot_pos = y * 4 + sub
                if top <= dot_pos < bottom:
                    bits |= SUB_ROW_BITS[sub]
            if bits:
                row_chars.append(f"[{PRIMARY}]{chr(BRAILLE_BASE + bits)}[/]")
            else:
                row_chars.append(" ")
        lines.append("".join(row_chars))
    return "\n".join(lines)


def _render_spectrogram(columns: list[np.ndarray], height: int) -> str:
    """Return a Rich-markup multi-line string for the rolling spectrogram.

    Rows: frequency (high at top, low at bottom).
    Columns: time (oldest at left, newest at right).
    Each cell is one of BLOCKS coloured by magnitude.
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
                brightness = int(55 + level * 25)  # dim → bright blue
                chars.append(f"[#{brightness:02x}b4fa]{BLOCKS[level]}[/]")
        lines.append("".join(chars))
    return "\n".join(lines)


def _make_renderable(
    width: int,
    transcription: list[str],
    samples: deque[float],
    samples_lock: threading.Lock,
    spec_columns: deque[np.ndarray],
    spec_columns_lock: threading.Lock,
) -> Group:
    """Build the Rich renderable for one Live refresh."""
    if _VIZ_MODE == "waveform":
        with samples_lock:
            snap = deque(samples)
        viz_str = _render_waveform(snap, width, VIZ_HEIGHT)
    else:
        with spec_columns_lock:
            snap_cols = list(spec_columns)
        snap_cols = snap_cols[-width:]
        viz_str = _render_spectrogram(snap_cols, SPEC_BANDS)

    if _paused.is_set():
        text = "[dim]Paused — press Enter to resume[/dim]"
    elif transcription[0]:
        text = transcription[0]
    else:
        text = "[dim]Listening...[/dim]"
    return Group(
        Text.from_markup(viz_str),
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

_MAX_SAMPLES = int(3.0 * settings.sample_rate)
samples: deque[float] = deque(maxlen=_MAX_SAMPLES)
samples_lock = threading.Lock()

_MAX_COLUMNS = 300
spec_columns: deque[np.ndarray] = deque(maxlen=_MAX_COLUMNS)
spec_columns_lock = threading.Lock()

transcription: list[str] = [""]
_stop_event = threading.Event()
_paused = threading.Event()

_chunk_samples = int(settings.chunk_duration * settings.sample_rate)
_spec_state: SpectrogramState = build_spec_state(settings.sample_rate, _chunk_samples)


def _on_chunk(chunk: np.ndarray) -> None:
    """Feed the latest mic chunk into the active visualisation buffer."""
    if _paused.is_set():
        return
    if _VIZ_MODE == "waveform":
        with samples_lock:
            samples.extend(chunk.tolist())
    else:
        col = compute_column(chunk, _spec_state)
        with spec_columns_lock:
            spec_columns.append(col)


def _refresh_loop(live: Live) -> None:
    """Push visualisation redraws at a fixed rate, independent of transcription."""
    interval = 1.0 / _REFRESH_HZ
    while not _stop_event.is_set():
        width = max(10, live.console.width)
        live.update(
            _make_renderable(
                width,
                transcription,
                samples,
                samples_lock,
                spec_columns,
                spec_columns_lock,
            )
        )
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
            with samples_lock:
                samples.clear()
            with spec_columns_lock:
                spec_columns.clear()


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
