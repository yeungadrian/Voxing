"""Test the STT pipeline with a live audio visualisation.

Usage: uv run scripts/stt.py

Set _VIZ_MODE to switch visualisations.
"""

import threading
from typing import Literal

import numpy as np
from rich.console import Group
from rich.live import Live
from rich.panel import Panel
from rich.text import Text

from voxing.config import Settings
from voxing.palette import BLUE, TEXT
from voxing.parakeet import load_stt
from voxing.stt import RealtimeTranscriber
from voxing.viz import (
    BRAILLE_BASE,
    OscilloscopeViz,
    Visualizer,
    VizFrame,
    WaveformViz,
)

_VIZ_MODE: Literal["waveform", "oscilloscope"] = "oscilloscope"
_REFRESH_HZ = 30
_VIZ_HEIGHT = 3


def _render_frame(frame: VizFrame) -> str:
    """Convert any VizFrame to a Rich-markup string."""
    lines: list[str] = []
    for row_idx, row in enumerate(frame.grid):
        color_row = (
            frame.colors[row_idx]
            if frame.colors is not None and row_idx < len(frame.colors)
            else None
        )
        chars: list[str] = []
        for col_idx, bits in enumerate(row):
            if bits:
                color = color_row[col_idx] if color_row is not None else BLUE
                chars.append(f"[{color}]{chr(BRAILLE_BASE + bits)}[/]")
            else:
                chars.append(" ")
        lines.append("".join(chars))
    return "\n".join(lines)


settings = Settings()

print("Loading model...")
model = load_stt(settings.model_id)
print("Model loaded.\n")

input("Press Enter to start recording... ")

_chunk_samples = int(settings.chunk_duration * settings.sample_rate)

_viz: Visualizer
match _VIZ_MODE:
    case "waveform":
        _viz = WaveformViz()
    case "oscilloscope":
        _viz = OscilloscopeViz()

transcription: list[str] = [""]
_stop_event = threading.Event()
_paused = threading.Event()


def _on_chunk(chunk: np.ndarray) -> None:
    """Feed the latest mic chunk into the visualiser."""
    if not _paused.is_set():
        _viz.push(chunk)


def _make_renderable(width: int, transcription: list[str]) -> Group:
    """Build the Rich renderable for one Live refresh."""
    frame = _viz.render(width, _VIZ_HEIGHT)
    viz_str = _render_frame(frame)

    if _paused.is_set():
        text = "[dim]Paused \u2014 press Enter to resume[/dim]"
    elif transcription[0]:
        text = transcription[0]
    else:
        text = "[dim]Listening...[/dim]"
    return Group(
        Text.from_markup(viz_str),
        Panel(
            Text.from_markup(text),
            title="transcription",
            border_style=f"dim {TEXT}",
        ),
    )


def _refresh_loop(live: Live) -> None:
    """Push visualisation redraws at a fixed rate, independent of transcription."""
    interval = 1.0 / _REFRESH_HZ
    while not _stop_event.is_set():
        width = max(10, live.console.width)
        live.update(_make_renderable(width, transcription))
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
