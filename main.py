import logging
import warnings
from dataclasses import dataclass, field
from typing import cast

import mlx.nn as nn
import numpy as np
import sounddevice as sd
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
    "en": ("af_heart", "a"),  # American English - Grade A
    "es": ("ef_dora", "e"),  # Spanish
    "fr": ("ff_siwis", "f"),  # French - Grade B-
    "hi": ("hf_alpha", "h"),  # Hindi - Grade C
    "it": ("if_sara", "i"),  # Italian - Grade C
    "ja": ("jf_alpha", "j"),  # Japanese - Grade C+
    "pt": ("pf_dora", "p"),  # Brazilian Portuguese
    "zh-cn": ("zf_xiaobei", "z"),  # Mandarin Chinese
    "zh-tw": ("zf_xiaobei", "z"),  # Mandarin Chinese
}
DEFAULT_VOICE = ("af_heart", "a")


@dataclass
class VoiceAssistant:
    # Models
    stt_model: nn.Module
    llm_model: nn.Module
    tts_model: nn.Module
    tokenizer: TokenizerWrapper
    # Audio
    input_sample_rate: int = 16000
    tts_sample_rate: int = 24000
    silence_threshold: float = 0.0
    silence_duration: float = 0.5
    max_record_duration: float = 10.0
    # Wake words
    wake_words: tuple[str, ...] = ("reachy", "reach", "richy", "reechy", "hello")
    # LLM settings
    max_tokens: int = 512
    # Console
    console: Console = field(default_factory=Console)

    def _transcribe(self, audio: np.ndarray) -> str:
        result = self.stt_model.generate(audio)
        return result.text.strip()

    def _has_wake_word(self, text: str) -> bool:
        return any(word in text.lower() for word in self.wake_words)

    def _play_audio(self, audio: np.ndarray) -> None:
        audio = audio.flatten().astype(np.float32)
        max_val = np.abs(audio).max()
        if max_val > 0:
            audio = audio / max_val * 0.9
        with sd.OutputStream(
            samplerate=self.tts_sample_rate,
            channels=1,
            dtype=np.float32,
            blocksize=4096,
        ) as stream:
            stream.write(audio.reshape(-1, 1))

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
            return None
        detected_lang = detect(text)
        voice, lang_code = KOKORO_VOICES.get(detected_lang, DEFAULT_VOICE)
        for result in self.tts_model.generate(
            text=text, voice=voice, lang_code=lang_code, speed=1.0
        ):
            self._play_audio(np.array(result.audio, dtype=np.float32))

    def _respond(self, user_input: str, status) -> None:
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
                    is_speaking = False

            if phrase_buffer.strip():
                is_speaking = True
                live.update(make_display(full_response))
                self._speak(phrase_buffer)

        status.start()

    def _record(self) -> np.ndarray | None:
        chunk_duration = 0.1
        chunk_samples = int(self.input_sample_rate * chunk_duration)
        silence_chunks_needed = int(self.silence_duration / chunk_duration)
        max_chunks = int(self.max_record_duration / chunk_duration)

        audio_chunks: list[np.ndarray] = []
        silence_count = 0
        recording = False

        with sd.InputStream(
            samplerate=self.input_sample_rate, channels=1, dtype=np.float32
        ) as stream:
            while len(audio_chunks) < max_chunks:
                chunk, _ = stream.read(chunk_samples)
                rms = np.sqrt(np.mean(chunk**2))
                has_voice = rms > self.silence_threshold

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
                    silence_count += 1
                    if silence_count >= silence_chunks_needed:
                        break

        if not audio_chunks:
            return None
        return np.concatenate(audio_chunks, axis=0).flatten()

    def run(self) -> None:
        self.console.print(
            f"[bold {COLOR_GREEN}]Voice Assistant Ready[/bold {COLOR_GREEN}] "
            f"[{COLOR_COMMENT}](wake words: {', '.join(self.wake_words)})[/{COLOR_COMMENT}]"
        )
        awaiting_command = False

        with self.console.status(f"[bold {COLOR_BLUE}]Listening...") as status:
            while True:
                if awaiting_command:
                    status.update(
                        f"[bold {COLOR_YELLOW}]Awake - listening for command..."
                    )
                else:
                    status.update(f"[bold {COLOR_BLUE}]Listening...")

                audio = self._record()
                if audio is None:
                    awaiting_command = False
                    continue

                status.update(f"[bold {COLOR_GREEN}]Transcribing...")
                text = self._transcribe(audio)
                if not text:
                    continue

                if not awaiting_command:
                    if self._has_wake_word(text):
                        self.console.print(
                            f"[bold {COLOR_YELLOW}]Wake word detected![/bold {COLOR_YELLOW}]"
                        )
                        awaiting_command = True
                    continue

                self.console.print(f"[{COLOR_COMMENT}]> You:[/{COLOR_COMMENT}] {text}")
                self._respond(text, status)
                awaiting_command = False


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

    assistant = VoiceAssistant(
        stt_model=stt_model,
        llm_model=llm_model,
        tts_model=tts_model,
        tokenizer=tokenizer,
        console=console,
    )

    try:
        assistant.run()
    finally:
        del stt_model, llm_model, tts_model


if __name__ == "__main__":
    main()
