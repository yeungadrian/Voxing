"""Full-screen radial spectrum visualisation."""

import queue
import threading

import numpy as np
import sounddevice as sd
from rich.segment import Segment
from rich.style import Style
from textual import work
from textual.app import ComposeResult
from textual.events import Key
from textual.screen import Screen
from textual.strip import Strip
from textual.widget import Widget

from voxing.config import Settings
from voxing.tui.widgets import FooterBar
from voxing.viz import BRAILLE_BASE
from voxing.viz._radial import RadialViz

_REFRESH_INTERVAL = 1.0 / 30.0


class RadialWidget(Widget):
    """Full-page radial spectrum rendered with braille characters."""

    DEFAULT_CSS = """
    RadialWidget {
        width: 1fr;
        height: 1fr;
    }
    """

    def __init__(self) -> None:
        super().__init__()
        self._viz = RadialViz()
        self._grid: list[list[int]] = []
        self._colors: list[list[str]] | None = None
        self._color_cache: dict[str, Style] = {}

    def on_mount(self) -> None:
        """Start 30fps refresh timer."""
        self.set_interval(_REFRESH_INTERVAL, self._tick)

    def _tick(self) -> None:
        """Recompute the frame then refresh."""
        w = self.size.width or 1
        h = self.size.height or 1
        frame = self._viz.render(w, h)
        self._grid = frame.grid
        self._colors = frame.colors
        self.refresh()

    def push_chunk(self, chunk: np.ndarray) -> None:
        """Forward an audio chunk to the radial viz."""
        self._viz.push(chunk)

    def _color_style(self, color: str) -> Style:
        """Get or create a cached Style for a hex color."""
        style = self._color_cache.get(color)
        if style is None:
            style = Style.parse(color)
            self._color_cache[color] = style
        return style

    def render_line(self, y: int) -> Strip:
        """Render one terminal row from the current frame."""
        grid = self._grid
        if not grid or y >= len(grid):
            return Strip([Segment(" " * (self.size.width or 1))])

        row = grid[y]
        color_row = self._colors[y] if self._colors is not None and y < len(self._colors) else None
        segments: list[Segment] = []

        for ci, bits in enumerate(row):
            if bits:
                style = (
                    self._color_style(color_row[ci])
                    if color_row is not None
                    else Style()
                )
                segments.append(Segment(chr(BRAILLE_BASE + bits), style))
            else:
                segments.append(Segment(" "))

        return Strip(segments)


class RadialScreen(Screen[None]):
    """Full-screen radial spectrum with live mic input."""

    BINDINGS = [("escape", "dismiss", "Back")]

    def __init__(self) -> None:
        super().__init__()
        self._settings = Settings()
        self._chunk_queue: queue.Queue[np.ndarray | None] = queue.Queue()
        self._stop_event = threading.Event()

    def compose(self) -> ComposeResult:
        yield RadialWidget()
        yield FooterBar()

    def on_mount(self) -> None:
        """Start mic capture worker."""
        self.query_one(FooterBar).set_status("[dim]esc to exit[/]")
        self._capture_mic()

    @property
    def radial_widget(self) -> RadialWidget:
        return self.query_one(RadialWidget)

    @work(thread=True, exclusive=True, group="radial-mic")
    def _capture_mic(self) -> None:
        """Capture mic audio and feed to radial viz."""
        sr = self._settings.sample_rate
        chunk_samples = int(self._settings.chunk_duration * sr)
        try:
            with sd.InputStream(
                samplerate=sr, channels=1, dtype=np.float32
            ) as stream:
                while not self._stop_event.is_set():
                    chunk, _ = stream.read(chunk_samples)
                    mono = chunk[:, 0]
                    self.app.call_from_thread(
                        self.radial_widget.push_chunk, mono
                    )
        except Exception:
            pass

    def on_key(self, event: Key) -> None:
        """Handle ESC to stop and dismiss."""
        if event.key == "escape":
            self._stop_event.set()
            self.dismiss()
