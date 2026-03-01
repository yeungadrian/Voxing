"""Test the STT pipeline with a live waveform visualisation.

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
from voxing.tui.theme import FOREGROUND, PRIMARY
from voxing.waveform import BRAILLE_BASE, SUB_ROW_BITS, VIZ_HEIGHT, bar_columns, peaks

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


def _make_renderable(
    samples: deque[float],
    samples_lock: threading.Lock,
    transcription: list[str],
    width: int,
) -> Group:
    """Build the Rich renderable for one Live refresh."""
    with samples_lock:
        samples_snapshot = deque(samples)
    waveform_str = _render_waveform(samples_snapshot, width, VIZ_HEIGHT)
    waveform_text = Text.from_markup(waveform_str)
    if _paused.is_set():
        text = "[dim]Paused — press Enter to resume[/dim]"
    elif transcription[0]:
        text = transcription[0]
    else:
        text = "[dim]Listening...[/dim]"
    return Group(
        waveform_text,
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

# Raw sample ring buffer: 1.5 s of audio at the configured sample rate.
_MAX_SAMPLES = int(3.0 * settings.sample_rate)
samples: deque[float] = deque(maxlen=_MAX_SAMPLES)
samples_lock = threading.Lock()
transcription: list[str] = [""]
_stop_event = threading.Event()
_paused = threading.Event()


def _on_chunk(chunk: np.ndarray) -> None:
    """Extend the raw sample ring buffer with the latest mic chunk."""
    if _paused.is_set():
        return
    with samples_lock:
        samples.extend(chunk.tolist())


def _refresh_loop(live: Live) -> None:
    """Push waveform redraws at a fixed rate, independent of transcription."""
    interval = 1.0 / _REFRESH_HZ
    while not _stop_event.is_set():
        width = max(10, live.console.width)
        live.update(_make_renderable(samples, samples_lock, transcription, width=width))
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
