import logging
import re
from collections.abc import Iterator
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import sounddevice as sd
from langdetect import detect
from mlx_audio.stt import load_model as load_stt_model
from mlx_audio.tts.utils import load_model as load_tts_model
from mlx_lm import load, stream_generate
from mlx_lm.tokenizer_utils import TokenizerWrapper


# TODO: Replace logging with rich
# TODO: Silence all other logs?
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

PHRASE_END = re.compile(r"[.!?,;:]+\s*")

KOKORO_MAPPING = {
    "en": "a",
    "es": "e",
    "fr": "f",
    "hi": "h",
    "it": "i",
    "ja": "j",
    "pt": "p",
    "zh-cn": "z",
    "zh-tw": "z",
}


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
        _lang_code = detect(text)
        print(_lang_code)
        for result in self.tts_model.generate(
            text=text,
            voice=self.tts_voice,
            speed=1.0,
            lang_code=KOKORO_MAPPING.get(_lang_code, "a"),
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

    def _respond(self, user_input: str) -> None:
        phrase_buffer = ""
        full_response = ""

        for token in self._stream_llm(user_input):
            phrase_buffer += token
            full_response += token

            match = PHRASE_END.search(phrase_buffer)
            if match:
                phrase = phrase_buffer[: match.end()]
                phrase_buffer = phrase_buffer[match.end() :]
                self._speak(phrase)

        if phrase_buffer.strip():
            self._speak(phrase_buffer)

        logger.info(f"Full response: {full_response}")

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
        logger.info(f"Wake words: {', '.join(self.wake_words)}")
        awaiting_command = False

        while True:
            audio = self._record()
            if audio is None:
                awaiting_command = False
                continue

            text = self._transcribe(audio)
            if not text:
                continue

            logger.info(f"Heard: {text}")

            if not awaiting_command:
                if self._has_wake_word(text):
                    logger.info("Wake word detected!")
                    awaiting_command = True
                continue

            logger.info(f"Command: {text}")
            self._respond(text)
            awaiting_command = False


def main() -> None:
    # Poor mlx typing
    stt_model = load_stt_model("mlx-community/whisper-large-v3-turbo-asr-fp16")
    llm_model, tokenizer = load("LiquidAI/LFM2.5-1.2B-Instruct-MLX-8bit")  # ty:ignore[invalid-assignment]
    tts_model = load_tts_model("mlx-community/Kokoro-82M-bf16")  # ty:ignore[invalid-argument-type]

    assistant = VoiceAssistant(
        stt_model=stt_model,
        llm_model=llm_model,
        tts_model=tts_model,
        tokenizer=tokenizer,
    )

    try:
        assistant.run()
    finally:
        del stt_model, llm_model, tts_model
        mx.clear_cache()


if __name__ == "__main__":
    main()
