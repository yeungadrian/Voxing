"""Simple voice assistant with wake word detection."""

import logging
import re
from collections.abc import Callable
from dataclasses import dataclass, field

import mlx.core as mx
import numpy as np
import sounddevice as sd
from mlx_audio.stt import load_model as load_stt_model
from mlx_audio.tts.utils import load_model as load_tts_model
from mlx_lm import load, stream_generate

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class Config:
    sample_rate: int = 16000
    tts_sample_rate: int = 24000
    silence_threshold: float = 0.02
    silence_duration: float = 0.5
    max_record_duration: float = 10.0
    min_record_duration: float = 0.3
    wake_words: list[str] = field(
        default_factory=lambda: ["reachy", "reach", "richy", "reechy", "hello"]
    )
    stt_model: str = "mlx-community/whisper-large-v3-turbo-asr-fp16"
    llm_model: str = "mlx-community/Jan-v3-4B-base-instruct-8bit"
    tts_model: str = "mlx-community/Kokoro-82M-bf16"
    tts_voice: str = "af_heart"
    max_tokens: int = 512


def record_until_silence(
    sample_rate: int,
    silence_threshold: float,
    silence_duration: float,
    max_duration: float,
    min_duration: float,
) -> np.ndarray | None:
    """Record audio until silence. Returns None if too short."""
    chunk_duration = 0.1
    chunk_samples = int(sample_rate * chunk_duration)
    silence_chunks_needed = int(silence_duration / chunk_duration)
    max_chunks = int(max_duration / chunk_duration)
    min_chunks = int(min_duration / chunk_duration)

    audio_chunks = []
    silence_count = 0
    recording = False

    with sd.InputStream(samplerate=sample_rate, channels=1, dtype=np.float32) as stream:
        while len(audio_chunks) < max_chunks:
            chunk, _ = stream.read(chunk_samples)
            rms = np.sqrt(np.mean(chunk**2))
            has_voice = rms > silence_threshold

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


def play_audio(audio: np.ndarray, sample_rate: int) -> None:
    audio = audio.flatten().astype(np.float32)
    max_val = max(abs(audio.max()), abs(audio.min()), 1.0)
    if max_val > 1.0:
        audio = audio / max_val
    sd.play(audio, sample_rate)
    sd.wait()


PHRASE_END = re.compile(r"[.!?,;:]+\s*")


class VoiceAssistant:
    def __init__(
        self,
        config: Config | None = None,
        command_handler: Callable[[str], str | None] | None = None,
    ):
        self.config = config or Config()
        self.command_handler = command_handler
        self._load_models()

    def _load_models(self) -> None:
        logger.info("Loading STT model...")
        self.stt_model = load_stt_model(self.config.stt_model)

        logger.info("Loading LLM model...")
        self.llm_model, self.tokenizer = load(self.config.llm_model)

        logger.info("Loading TTS model...")
        self.tts_model = load_tts_model(self.config.tts_model)  # ty:ignore[invalid-argument-type]
        logger.info("Models loaded.")

    def _transcribe(self, audio: np.ndarray) -> str:
        result = self.stt_model.generate(audio)
        mx.eval()
        return result.text.strip()

    def _has_wake_word(self, text: str) -> bool:
        text_lower = text.lower()
        return any(word in text_lower for word in self.config.wake_words)

    def _speak(self, text: str) -> None:
        if not text.strip():
            return
        for result in self.tts_model.generate(
            text=text,
            voice=self.config.tts_voice,
            speed=1.0,
            lang_code="a",
        ):
            mx.eval()
            audio = np.array(result.audio, dtype=np.float32)
            play_audio(audio, self.config.tts_sample_rate)

    def _format_prompt(self, command: str) -> str:
        if self.tokenizer.chat_template:
            messages = [{"role": "user", "content": command}]
            return self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
        return command

    def _stream_llm(self, command: str):
        for chunk in stream_generate(
            model=self.llm_model,
            tokenizer=self.tokenizer,
            prompt=self._format_prompt(command),
            max_tokens=self.config.max_tokens,
        ):
            yield chunk.text

    def _respond(self, command: str) -> None:
        if self.command_handler:
            response = self.command_handler(command)
            if response is not None:
                logger.info(f"Response: {response}")
                self._speak(response)
                return

        phrase_buffer = ""
        full_response = ""

        for token in self._stream_llm(command):
            phrase_buffer += token
            full_response += token

            match = PHRASE_END.search(phrase_buffer)
            if match:
                phrase = phrase_buffer[: match.end()]
                phrase_buffer = phrase_buffer[match.end() :]
                mx.eval()
                self._speak(phrase)

        if phrase_buffer.strip():
            self._speak(phrase_buffer)

        logger.info(f"Full response: {full_response}")

    def _record(self) -> np.ndarray | None:
        return record_until_silence(
            sample_rate=self.config.sample_rate,
            silence_threshold=self.config.silence_threshold,
            silence_duration=self.config.silence_duration,
            max_duration=self.config.max_record_duration,
            min_duration=self.config.min_record_duration,
        )

    def run(self) -> None:
        logger.info(f"Wake words: {', '.join(self.config.wake_words)}")
        logger.info("Waiting for wake word...")

        listening = False

        try:
            while True:
                audio = self._record()
                if audio is None:
                    if listening:
                        logger.info("No command heard. Waiting for wake word...")
                        listening = False
                    continue

                text = self._transcribe(audio)
                if not text:
                    if listening:
                        listening = False
                    continue

                logger.info(f"Heard: {text}")

                if not listening:
                    if self._has_wake_word(text):
                        logger.info("Wake word detected! Listening...")
                        listening = True
                    continue

                logger.info(f"Command: {text}")
                self._respond(text)
                logger.info("Waiting for wake word...")
                listening = False

        except KeyboardInterrupt:
            logger.info("Shutting down...")

    def cleanup(self) -> None:
        del self.tts_model
        del self.llm_model
        del self.stt_model
        mx.clear_cache()


# Example: Robot command handler
def robot_handler(command: str) -> str | None:
    """Handle robot commands. Returns None to fall through to LLM."""
    commands = {
        "forward": "Moving forward",
        "backward": "Moving backward",
        "left": "Turning left",
        "right": "Turning right",
        "stop": "Stopping",
    }
    command_lower = command.lower()
    for keyword, response in commands.items():
        if keyword in command_lower:
            # Add GPIO control here
            logger.info(f"Robot command: {keyword}")
            return response
    return None


def main() -> None:
    config = Config()
    # Pass robot_handler to enable robot commands:
    # assistant = VoiceAssistant(config, command_handler=robot_handler)
    assistant = VoiceAssistant(config)

    try:
        assistant.run()
    finally:
        assistant.cleanup()


if __name__ == "__main__":
    main()
