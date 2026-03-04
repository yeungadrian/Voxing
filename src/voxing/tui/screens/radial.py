"""Full-screen radial spectrum visualisation."""

import queue
import threading

import numpy as np
import sounddevice as sd
from textual import work
from textual.app import ComposeResult
from textual.events import Key
from textual.screen import Screen

from voxing.config import Settings
from voxing.tui.widgets import FooterBar, VizWidget
from voxing.viz import RadialViz

_REFRESH_RATE = 30.0


class RadialScreen(Screen[None]):
    """Full-screen radial spectrum with live mic input."""

    BINDINGS = [("escape", "dismiss", "Back")]

    def __init__(self, settings: Settings) -> None:
        super().__init__()
        self._settings = settings
        self._chunk_queue: queue.Queue[np.ndarray | None] = queue.Queue()
        self._stop_event = threading.Event()

    def compose(self) -> ComposeResult:
        yield VizWidget(RadialViz(), refresh_rate=_REFRESH_RATE)
        yield FooterBar()

    def on_mount(self) -> None:
        """Start mic capture worker."""
        self.query_one(FooterBar).set_status("[dim]esc to exit[/]")
        self._capture_mic()

    @property
    def _viz_widget(self) -> VizWidget:
        return self.query_one(VizWidget)

    @work(thread=True, exclusive=True, group="radial-mic")
    def _capture_mic(self) -> None:
        """Capture mic audio and feed to radial viz."""
        sr = self._settings.sample_rate
        chunk_samples = int(self._settings.chunk_duration * sr)
        try:
            with sd.InputStream(samplerate=sr, channels=1, dtype=np.float32) as stream:
                while not self._stop_event.is_set():
                    chunk, _ = stream.read(chunk_samples)
                    mono = chunk[:, 0]
                    self.app.call_from_thread(self._viz_widget.push_chunk, mono)
        except Exception:
            pass

    def on_key(self, event: Key) -> None:
        """Handle ESC to stop and dismiss."""
        if event.key == "escape":
            self._stop_event.set()
            self.dismiss()
