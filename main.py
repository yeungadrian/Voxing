import re
from collections.abc import Iterator
from dataclasses import dataclass, field

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import sounddevice as sd
from langdetect import detect
from mlx_audio.stt import load_model as load_stt_model
from mlx_audio.tts.utils import load_model as load_tts_model
from mlx_lm import load, stream_generate
from mlx_lm.tokenizer_utils import TokenizerWrapper
from rich.console import Console

PHRASE_END = re.compile(r"[.!?,;:]+\s*")

# Maps langdetect codes to Kokoro (voice, lang_code)
KOKORO_VOICES = {
    "en": ("af_heart", "a"),      # American English - Grade A
    "es": ("ef_dora", "e"),       # Spanish
    "fr": ("ff_siwis", "f"),      # French - Grade B-
    "hi": ("hf_alpha", "h"),      # Hindi - Grade C
    "it": ("if_sara", "i"),       # Italian - Grade C
    "ja": ("jf_alpha", "j"),      # Japanese - Grade C+
    "pt": ("pf_dora", "p"),       # Brazilian Portuguese
    "zh-cn": ("zf_xiaobei", "z"), # Mandarin Chinese
    "zh-tw": ("zf_xiaobei", "z"), # Mandarin Chinese
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
    silence_duration: float = 1.0
    max_record_duration: float = 10.0
    min_record_duration: float = 0.3
    # Wake words
    wake_words: tuple[str, ...] = ("reachy", "reach", "richy", "reechy", "hello")
    # TTS/LLM settings
    tts_voice: str = "af_heart"
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
        max_val = max(abs(audio.max()), abs(audio.min()))
        if max_val > 1.0:
            audio = audio / max_val
        sd.play(audio, self.tts_sample_rate)
        sd.wait()

    def _speak(self, text: str) -> None:
        if not text.strip():
            return
        detected_lang = detect(text)
        voice, lang_code = KOKORO_VOICES.get(detected_lang, DEFAULT_VOICE)
        for result in self.tts_model.generate(
            text=text,
            voice=voice,
            lang_code=lang_code,
            speed=1.0,
        ):
            audio = np.array(result.audio, dtype=np.float32)
            self._play_audio(audio)

    def _format_prompt(self, user_input: str) -> str:
        if self.tokenizer.chat_template:
            messages = [
                {
                    "role": "assistant",
                    "content": "You are a voice assistant. All of your responses will be outputed via tts.",
                },
                {"role": "user", "content": user_input},
            ]
            return self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
        return user_input

    def _stream_llm(self, user_input: str) -> Iterator[str]:
        for chunk in stream_generate(
            model=self.llm_model,
            tokenizer=self.tokenizer,
            prompt=self._format_prompt(user_input),
            max_tokens=self.max_tokens,
        ):
            yield chunk.text

    def _respond(self, user_input: str, status) -> None:
        phrase_buffer = ""
        full_response = ""

        status.update("[bold cyan]Thinking...")
        for token in self._stream_llm(user_input):
            phrase_buffer += token
            full_response += token

            match = PHRASE_END.search(phrase_buffer)
            if match:
                phrase = phrase_buffer[: match.end()]
                phrase_buffer = phrase_buffer[match.end() :]
                status.update("[bold magenta]Speaking...")
                self._speak(phrase)
                status.update("[bold cyan]Thinking...")

        if phrase_buffer.strip():
            status.update("[bold magenta]Speaking...")
            self._speak(phrase_buffer)

        self.console.print(f"[dim]> Assistant:[/dim] {full_response}")

    def _record(self) -> np.ndarray | None:
        """Record audio until silence. Returns None if too short."""
        chunk_duration = 0.1
        chunk_samples = int(self.input_sample_rate * chunk_duration)
        silence_chunks_needed = int(self.silence_duration / chunk_duration)
        max_chunks = int(self.max_record_duration / chunk_duration)
        min_chunks = int(self.min_record_duration / chunk_duration)

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

        if len(audio_chunks) < min_chunks:
            return None

        return np.concatenate(audio_chunks, axis=0).flatten()

    def run(self) -> None:
        self.console.print(
            f"[bold green]Voice Assistant Ready[/bold green] "
            f"[dim](wake words: {', '.join(self.wake_words)})[/dim]"
        )
        awaiting_command = False

        with self.console.status("[bold blue]Listening...") as status:
            while True:
                if awaiting_command:
                    status.update("[bold yellow]Awake - listening for command...")
                else:
                    status.update("[bold blue]Listening...")

                audio = self._record()
                if audio is None:
                    awaiting_command = False
                    continue

                status.update("[bold green]Transcribing...")
                text = self._transcribe(audio)
                if not text:
                    continue

                if not awaiting_command:
                    if self._has_wake_word(text):
                        self.console.print("[bold yellow]Wake word detected![/bold yellow]")
                        awaiting_command = True
                    continue

                self.console.print(f"[dim]> You:[/dim] {text}")
                self._respond(text, status)
                awaiting_command = False


def main() -> None:
    console = Console()

    with console.status("[bold blue]Loading STT model...") as status:
        stt_model = load_stt_model("mlx-community/whisper-large-v3-turbo-asr-fp16")

        status.update("[bold blue]Loading LLM model...")
        llm_model, tokenizer = load("LiquidAI/LFM2.5-1.2B-Instruct-MLX-8bit")  # ty:ignore[invalid-assignment]

        status.update("[bold blue]Loading TTS model...")
        tts_model = load_tts_model("mlx-community/Kokoro-82M-bf16")  # ty:ignore[invalid-argument-type]

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
        mx.clear_cache()


if __name__ == "__main__":
    main()
