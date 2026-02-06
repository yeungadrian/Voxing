import queue
import threading
import time
from dataclasses import dataclass
from typing import Self

import numpy as np
from reachy_mini import ReachyMini
from rich.console import Console
from scipy.signal import resample

from assistant import (
    COLOR_BLUE,
    COLOR_GREEN,
    ResponseStats,
    VoiceAssistant,
    load_models,
    normalize_audio,
    rms,
)

REACHY_OUTPUT_SAMPLE_RATE = 16000


@dataclass(slots=True)
class ReachyController:
    robot: ReachyMini

    @classmethod
    def connect(cls) -> Self:
        robot = ReachyMini(media_backend="default")
        robot.media.start_recording()
        robot.media.start_playing()
        return cls(robot=robot)

    def record_audio(
        self,
        max_duration: float,
        silence_threshold: float,
        silence_duration: float,
    ) -> np.ndarray | None:
        self.flush_audio_buffer()

        sample_rate = self.robot.media.get_input_audio_samplerate()
        max_samples = int(max_duration * sample_rate)
        silence_samples_needed = int(silence_duration * sample_rate)

        audio_chunks: list[np.ndarray] = []
        silence_count = 0
        recording = False
        total_samples = 0

        while total_samples < max_samples:
            samples = self.robot.media.get_audio_sample()
            if samples is None or len(samples) == 0:
                time.sleep(0.01)
                continue
            if not isinstance(samples, np.ndarray):
                continue

            chunk = self._to_mono(samples)
            total_samples += len(chunk)
            has_voice = rms(chunk) > silence_threshold

            if not recording:
                if has_voice:
                    recording = True
                    audio_chunks.append(chunk)
                    silence_count = 0
                continue

            audio_chunks.append(chunk)

            if has_voice:
                silence_count = 0
            else:
                silence_count += len(chunk)
                if silence_count >= silence_samples_needed:
                    break

        if not audio_chunks:
            return None
        return np.concatenate(audio_chunks, axis=0).flatten()

    def play_beep(self, frequency: float = 880, duration: float = 0.15) -> None:
        """Play a short beep sound."""
        t = np.linspace(
            0, duration, int(REACHY_OUTPUT_SAMPLE_RATE * duration), dtype=np.float32
        )
        beep = (np.sin(2 * np.pi * frequency * t) * 0.5).astype(np.float32)
        self.robot.media.push_audio_sample(beep.reshape(-1, 1))
        time.sleep(duration)

    def flush_audio_buffer(self) -> None:
        while True:
            samples = self.robot.media.get_audio_sample()
            if samples is None or len(samples) == 0:
                break

    def _to_mono(self, samples: np.ndarray) -> np.ndarray:
        if len(samples.shape) > 1 and samples.shape[1] == 2:
            return samples.mean(axis=1).astype(np.float32)
        return np.asarray(samples, dtype=np.float32).flatten()

    def write_audio(self, audio: np.ndarray, sample_rate: int) -> float:
        """Write audio to the robot's media buffer. Returns duration in seconds."""
        audio = normalize_audio(audio).flatten()

        if sample_rate != REACHY_OUTPUT_SAMPLE_RATE:
            num_samples = int(len(audio) * REACHY_OUTPUT_SAMPLE_RATE / sample_rate)
            audio = resample(audio, num_samples).astype(np.float32)

        self.robot.media.push_audio_sample(audio.reshape(-1, 1))
        return len(audio) / REACHY_OUTPUT_SAMPLE_RATE

    def disconnect(self) -> None:
        self.robot.media.stop_recording()
        self.robot.media.stop_playing()
        self.robot.goto_sleep()


@dataclass(slots=True, kw_only=True)
class ReachyVoiceAssistant(VoiceAssistant):
    reachy: ReachyController

    def on_startup(self) -> None:
        self.reachy.robot.goto_sleep()
        self.reachy.play_beep(frequency=880, duration=0.15)
        time.sleep(0.5)

    def on_wake_word(self) -> None:
        self.reachy.robot.wake_up()

    def on_ready_for_command(self) -> None:
        self.reachy.play_beep()
        self.reachy.flush_audio_buffer()

    def on_sleep(self) -> None:
        self.reachy.robot.goto_sleep()

    def on_shutdown(self) -> None:
        self.reachy.disconnect()

    def _record(self) -> np.ndarray | None:
        return self.reachy.record_audio(
            self.max_record_duration,
            self.silence_threshold,
            self.silence_duration,
        )

    def _playback_worker(
        self,
        audio_queue: queue.Queue[np.ndarray | None],
        stats: ResponseStats,
        playback_done: threading.Event | None = None,
    ) -> None:
        """Push audio to robot buffer as fast as possible, wait at end for playback."""
        playback_start: float | None = None
        total_duration = 0.0

        while True:
            audio = audio_queue.get()
            if audio is None:
                break
            if playback_start is None:
                playback_start = time.perf_counter()
            total_duration += self.reachy.write_audio(audio, self.tts_sample_rate)

        # Wait for actual playback to complete
        if playback_start is not None:
            elapsed = time.perf_counter() - playback_start
            remaining = total_duration - elapsed
            if remaining > 0:
                time.sleep(remaining)

        stats.playback_done = time.perf_counter()
        if playback_done is not None:
            playback_done.set()


def main() -> None:
    console = Console()
    stt_model, llm_model, tts_model, tokenizer = load_models(console)

    with console.status(f"[bold {COLOR_BLUE}]Connecting to Reachy Mini..."):
        try:
            reachy = ReachyController.connect()
        except Exception as e:
            console.print(f"[bold red]Failed to connect to Reachy Mini: {e}[/bold red]")
            return

    console.print(f"[bold {COLOR_GREEN}]Connected to Reachy Mini[/bold {COLOR_GREEN}]")

    assistant = ReachyVoiceAssistant(
        stt_model=stt_model,
        llm_model=llm_model,
        tts_model=tts_model,
        tokenizer=tokenizer,
        silence_threshold=0.05,
        console=console,
        reachy=reachy,
    )

    assistant.run()


if __name__ == "__main__":
    main()
