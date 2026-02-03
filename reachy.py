import threading
import time
from dataclasses import dataclass, field
from typing import Self

import numpy as np
from reachy_mini import ReachyMini
from rich.console import Console

from assistant import (
    COLOR_BLUE,
    COLOR_GREEN,
    VoiceAssistant,
    load_models,
)


@dataclass(slots=True)
class AntennaKeyframe:
    left: float
    right: float
    duration: float


ANTENNA_ANIMATIONS: dict[str, list[AntennaKeyframe]] = {
    "startup": [
        AntennaKeyframe(0.0, 0.0, 0.3),
        AntennaKeyframe(0.3, -0.3, 0.2),
        AntennaKeyframe(-0.3, 0.3, 0.2),
        AntennaKeyframe(0.0, 0.0, 0.3),
    ],
    "idle": [
        AntennaKeyframe(0.1, 0.1, 1.0),
        AntennaKeyframe(0.2, 0.2, 1.0),
    ],
    "listening": [
        AntennaKeyframe(0.4, 0.4, 0.2),
        AntennaKeyframe(0.5, 0.5, 0.15),
        AntennaKeyframe(0.45, 0.45, 0.1),
    ],
    "thinking": [
        AntennaKeyframe(0.3, -0.2, 0.4),
        AntennaKeyframe(0.2, -0.3, 0.4),
    ],
    "speaking": [
        AntennaKeyframe(0.4, 0.4, 0.3),
        AntennaKeyframe(0.5, 0.3, 0.2),
        AntennaKeyframe(0.3, 0.5, 0.2),
        AntennaKeyframe(0.4, 0.4, 0.3),
    ],
}

REACHY_OUTPUT_SAMPLE_RATE = 16000


@dataclass(slots=True)
class ReachyController:
    robot: ReachyMini
    _animation_thread: threading.Thread | None = None
    _stop_animation: threading.Event = field(default_factory=threading.Event)

    @classmethod
    def connect(cls) -> Self:
        robot = ReachyMini(media_backend="default")
        robot.media.start_recording()
        robot.media.start_playing()
        return cls(robot=robot)

    def play_animation(self, name: str, loop: bool = False) -> None:
        if name not in ANTENNA_ANIMATIONS:
            return
        self.stop_animation()
        self._stop_animation.clear()
        self._animation_thread = threading.Thread(
            target=self._run_animation,
            args=(ANTENNA_ANIMATIONS[name], loop),
            daemon=True,
        )
        self._animation_thread.start()

    def stop_animation(self) -> None:
        if self._animation_thread is not None and self._animation_thread.is_alive():
            self._stop_animation.set()
            self._animation_thread.join(timeout=1.0)

    def _run_animation(self, keyframes: list[AntennaKeyframe], loop: bool) -> None:
        while not self._stop_animation.is_set():
            for kf in keyframes:
                if self._stop_animation.is_set():
                    return
                self.robot.goto_target(
                    antennas=[kf.left, kf.right], duration=kf.duration
                )
                time.sleep(kf.duration)
            if not loop:
                break

    def record_audio(
        self, max_duration: float, silence_threshold: float, silence_duration: float
    ) -> np.ndarray | None:
        self._flush_audio_buffer()

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
            has_voice = np.sqrt(np.mean(chunk**2)) > silence_threshold

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

    def _flush_audio_buffer(self) -> None:
        while True:
            samples = self.robot.media.get_audio_sample()
            if samples is None or len(samples) == 0:
                break

    def _to_mono(self, samples: np.ndarray) -> np.ndarray:
        if len(samples.shape) > 1 and samples.shape[1] == 2:
            return samples.mean(axis=1).astype(np.float32)
        return np.asarray(samples, dtype=np.float32).flatten()

    def play_audio(self, audio: np.ndarray, sample_rate: int) -> None:
        audio = audio.flatten().astype(np.float32)
        max_val = np.abs(audio).max()
        if max_val > 0:
            audio = audio / max_val * 0.9

        if sample_rate != REACHY_OUTPUT_SAMPLE_RATE:
            from scipy.signal import resample

            num_samples = int(len(audio) * REACHY_OUTPUT_SAMPLE_RATE / sample_rate)
            audio = resample(audio, num_samples).astype(np.float32)

        self.robot.media.push_audio_sample(audio.reshape(-1, 1))
        time.sleep(len(audio) / REACHY_OUTPUT_SAMPLE_RATE)

    def disconnect(self) -> None:
        self.stop_animation()
        self.robot.media.stop_recording()
        self.robot.media.stop_playing()
        self.robot.goto_sleep()


@dataclass(slots=True)
class ReachyVoiceAssistant(VoiceAssistant):
    reachy: ReachyController

    def on_startup(self) -> None:
        self.reachy.play_animation("startup", loop=False)
        time.sleep(1.0)

    def on_idle(self) -> None:
        self.reachy.play_animation("idle", loop=True)

    def on_wake_word(self) -> None:
        self.reachy.play_animation("listening", loop=False)

    def on_thinking(self) -> None:
        self.reachy.play_animation("thinking", loop=True)

    def on_speaking(self) -> None:
        self.reachy.play_animation("speaking", loop=True)

    def on_shutdown(self) -> None:
        self.reachy.disconnect()

    def _record(self) -> np.ndarray | None:
        return self.reachy.record_audio(
            self.max_record_duration, self.silence_threshold, self.silence_duration
        )

    def _play_audio(self, audio: np.ndarray) -> None:
        self.reachy.play_audio(audio, self.tts_sample_rate)


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
        silence_threshold=0.01,
        console=console,
        reachy=reachy,
    )

    assistant.run()


if __name__ == "__main__":
    main()
