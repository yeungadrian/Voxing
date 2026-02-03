import logging
import threading
import time
import warnings
from dataclasses import dataclass, field
from typing import cast

import mlx.nn as nn
import numpy as np
from reachy_mini import ReachyMini
from langdetect import detect
from mlx_audio.stt import load_model as load_stt_model
from mlx_audio.tts.utils import load_model as load_tts_model
from mlx_lm import load, stream_generate
from mlx_lm.tokenizer_utils import TokenizerWrapper
from rich.console import Console
from rich.live import Live
from rich.spinner import Spinner
from rich.text import Text

logging.getLogger("mlx").setLevel(logging.ERROR)
logging.getLogger("mlx_audio").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message=".*deprecated.*")

# Tokyo Night colors
COLOR_BLUE = "#7aa2f7"
COLOR_CYAN = "#7dcfff"
COLOR_GREEN = "#9ece6a"
COLOR_PURPLE = "#bb9af7"
COLOR_YELLOW = "#e0af68"
COLOR_COMMENT = "#565f89"

KOKORO_VOICES = {
    "en": ("af_heart", "a"),        # American English - Grade A
    "es": ("ef_dora", "e"),         # Spanish
    "fr": ("ff_siwis", "f"),        # French - Grade B-
    "hi": ("hf_alpha", "h"),        # Hindi - Grade C
    "it": ("if_sara", "i"),         # Italian - Grade C
    "ja": ("jf_alpha", "j"),        # Japanese - Grade C+
    "pt": ("pf_dora", "p"),         # Brazilian Portuguese
    "zh-cn": ("zf_xiaobei", "z"),   # Mandarin Chinese
    "zh-tw": ("zf_xiaobei", "z"),   # Mandarin Chinese
}
DEFAULT_VOICE = ("af_heart", "a")


WAKE_WORDS = ("reachy", "reach", "richy", "reechy", "hello")


@dataclass
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


@dataclass
class ReachyController:
    robot: ReachyMini
    _animation_thread: threading.Thread | None = None
    _stop_animation: threading.Event = field(default_factory=threading.Event)

    @classmethod
    def connect(cls) -> "ReachyController":
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
                self.robot.goto_target(antennas=[kf.left, kf.right], duration=kf.duration)
                time.sleep(kf.duration)
            if not loop:
                break

    def record_audio(self, max_duration: float, silence_threshold: float, silence_duration: float) -> np.ndarray | None:
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


@dataclass
class VoiceAssistant:
    stt_model: nn.Module
    llm_model: nn.Module
    tts_model: nn.Module
    tokenizer: TokenizerWrapper
    reachy: ReachyController
    console: Console = field(default_factory=Console)
    tts_sample_rate: int = 24000
    silence_threshold: float = 0.0
    silence_duration: float = 0.5
    max_record_duration: float = 10.0
    max_tokens: int = 512
    wake_words: tuple[str, ...] = WAKE_WORDS

    def _has_wake_word(self, text: str) -> bool:
        return any(word in text.lower() for word in self.wake_words)

    def _format_prompt(self, user_input: str) -> str:
        if self.tokenizer.chat_template:
            messages = [
                {
                    "role": "assistant",
                    "content": "You are a voice assistant. All of your responses will be outputed via tts so keep use of punctuation minimal.",
                },
                {"role": "user", "content": user_input},
            ]
            return self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
        return user_input

    def _speak(self, text: str) -> None:
        if not text.strip():
            return
        self.reachy.play_animation("speaking", loop=True)
        detected_lang = detect(text)
        voice, lang_code = KOKORO_VOICES.get(detected_lang, DEFAULT_VOICE)
        for result in self.tts_model.generate(
            text=text, voice=voice, lang_code=lang_code, speed=1.0
        ):
            self.reachy.play_audio(np.array(result.audio, dtype=np.float32), self.tts_sample_rate)

    def _respond(self, user_input: str, status) -> None:
        self.reachy.play_animation("thinking", loop=True)

        phrase_buffer = ""
        full_response = ""
        is_speaking = False

        def make_display(text: str) -> Text:
            style = COLOR_PURPLE if is_speaking else COLOR_CYAN
            spinner = Spinner("dots", style=style)
            rendered = cast(Text, spinner.render(self.console.get_time()))
            return Text.assemble(rendered, " ", ("> Assistant: ", COLOR_COMMENT), text)

        status.stop()

        with Live(
            make_display(""), console=self.console, refresh_per_second=10
        ) as live:
            for chunk in stream_generate(
                model=self.llm_model,
                tokenizer=self.tokenizer,
                prompt=self._format_prompt(user_input),
                max_tokens=self.max_tokens,
            ):
                phrase_buffer += chunk.text
                full_response += chunk.text
                live.update(make_display(full_response))

                if "\n" in phrase_buffer:
                    phrase, phrase_buffer = phrase_buffer.rsplit("\n", 1)
                    is_speaking = True
                    live.update(make_display(full_response))
                    self._speak(phrase)
                    self.reachy.play_animation("thinking", loop=True)
                    is_speaking = False

            if phrase_buffer.strip():
                is_speaking = True
                live.update(make_display(full_response))
                self._speak(phrase_buffer)

        status.start()

    def run(self) -> None:
        self.console.print(
            f"[bold {COLOR_GREEN}]Voice Assistant Ready[/bold {COLOR_GREEN}] "
            f"[{COLOR_COMMENT}](wake words: {', '.join(self.wake_words)})[/{COLOR_COMMENT}]"
        )
        awaiting_command = False
        self.reachy.play_animation("idle", loop=True)

        with self.console.status(f"[bold {COLOR_BLUE}]Listening...") as status:
            while True:
                if awaiting_command:
                    status.update(f"[bold {COLOR_YELLOW}]Awake - listening for command...")
                else:
                    status.update(f"[bold {COLOR_BLUE}]Listening...")

                audio = self.reachy.record_audio(
                    self.max_record_duration, self.silence_threshold, self.silence_duration
                )
                if audio is None:
                    awaiting_command = False
                    continue

                status.update(f"[bold {COLOR_GREEN}]Transcribing...")
                self.reachy.play_animation("thinking", loop=True)
                text = self.stt_model.generate(audio).text.strip()
                if not text:
                    self.reachy.play_animation("idle", loop=True)
                    continue

                if not awaiting_command:
                    if self._has_wake_word(text):
                        self.console.print(f"[bold {COLOR_YELLOW}]Wake word detected![/bold {COLOR_YELLOW}]")
                        awaiting_command = True
                        self.reachy.play_animation("listening", loop=False)
                    else:
                        self.reachy.play_animation("idle", loop=True)
                    continue

                self.console.print(f"[{COLOR_COMMENT}]> You:[/{COLOR_COMMENT}] {text}")
                self._respond(text, status)
                awaiting_command = False
                self.reachy.play_animation("idle", loop=True)


def main() -> None:
    console = Console()

    with console.status(f"[bold {COLOR_BLUE}]Loading STT model...") as status:
        stt_model = load_stt_model("mlx-community/whisper-large-v3-turbo-asr-fp16")

        status.update(f"[bold {COLOR_BLUE}]Loading LLM model...")
        llm_model, tokenizer = load("LiquidAI/LFM2.5-1.2B-Instruct-MLX-8bit")  # ty:ignore[invalid-assignment]

        status.update(f"[bold {COLOR_BLUE}]Loading TTS model...")
        tts_model = load_tts_model("mlx-community/Kokoro-82M-bf16")  # ty:ignore[invalid-argument-type]

        status.update(f"[bold {COLOR_BLUE}]Warming up models...")
        stt_model.generate(np.zeros(16000, dtype=np.float32))  # 1s of silence

        status.update(f"[bold {COLOR_BLUE}]Connecting to Reachy Mini...")
        try:
            reachy = ReachyController.connect()
        except Exception as e:
            console.print(f"[bold red]Failed to connect to Reachy Mini: {e}[/bold red]")
            return

        console.print(f"[bold {COLOR_GREEN}]Connected to Reachy Mini[/bold {COLOR_GREEN}]")
        reachy.play_animation("startup", loop=False)
        time.sleep(1.0)

    assistant = VoiceAssistant(
        stt_model=stt_model,
        llm_model=llm_model,
        tts_model=tts_model,
        tokenizer=tokenizer,
        console=console,
        reachy=reachy,
    )

    try:
        assistant.run()
    finally:
        reachy.disconnect()
        del stt_model, llm_model, tts_model


if __name__ == "__main__":
    main()
