"""Simple voice assistant with wake word detection."""

import logging
import re
import tempfile
import threading
import queue
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto

import mlx.core as mx
import numpy as np
import sounddevice as sd
import soundfile as sf
from mlx_audio.stt.generate import generate_transcription, load_model as load_stt_model
from mlx_audio.tts.utils import load_model as load_tts_model
from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_logits_processors, make_sampler

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
    temperature: float = 0.1
    top_k: int = 50
    top_p: float = 0.1
    repetition_penalty: float = 1.05


class State(Enum):
    WAITING_FOR_WAKE_WORD = auto()
    LISTENING_FOR_COMMAND = auto()
    RESPONDING = auto()


class CommandHandler(ABC):
    """Base class for command handlers. Extend to add custom commands."""

    @abstractmethod
    def can_handle(self, command: str) -> bool:
        pass

    @abstractmethod
    def handle(self, command: str) -> str:
        pass


class LLMCommandHandler(CommandHandler):
    def __init__(self, llm_model, tokenizer, sampler, logits_processors, config: Config):
        self.llm_model = llm_model
        self.tokenizer = tokenizer
        self.sampler = sampler
        self.logits_processors = logits_processors
        self.config = config

    def can_handle(self, command: str) -> bool:
        return True

    def _format_prompt(self, command: str) -> str:
        if self.tokenizer.chat_template:
            messages = [{"role": "user", "content": command}]
            return self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
        return command

    def handle(self, command: str) -> str:
        response = ""
        for chunk in stream_generate(
            model=self.llm_model,
            tokenizer=self.tokenizer,
            prompt=self._format_prompt(command),
            max_tokens=self.config.max_tokens,
            sampler=self.sampler,
            logits_processors=self.logits_processors,
        ):
            response += chunk.text
        return response

    def stream(self, command: str):
        for chunk in stream_generate(
            model=self.llm_model,
            tokenizer=self.tokenizer,
            prompt=self._format_prompt(command),
            max_tokens=self.config.max_tokens,
            sampler=self.sampler,
            logits_processors=self.logits_processors,
        ):
            yield chunk.text


def record_until_silence(
    sample_rate: int,
    silence_threshold: float,
    silence_duration: float,
    max_duration: float,
    min_duration: float,
) -> np.ndarray | None:
    """Record audio until silence is detected. Returns None if too short."""
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


SENTENCE_END = re.compile(r"[.!?]+\s*")


class VoiceAssistant:
    def __init__(self, config: Config | None = None):
        self.config = config or Config()
        self.state = State.WAITING_FOR_WAKE_WORD
        self.command_handlers: list[CommandHandler] = []
        self._tts_queue: queue.Queue[str | None] = queue.Queue()
        self._shutdown = threading.Event()

        self._load_models()
        self._setup_default_handlers()

    def _load_models(self) -> None:
        logger.info("Loading STT model...")
        self.stt_model = load_stt_model(self.config.stt_model)

        logger.info("Loading LLM model...")
        self.llm_model, self.tokenizer = load(self.config.llm_model)
        self.sampler = make_sampler(
            temp=self.config.temperature,
            top_k=self.config.top_k,
            top_p=self.config.top_p,
        )
        self.logits_processors = make_logits_processors(
            repetition_penalty=self.config.repetition_penalty
        )

        logger.info("Loading TTS model...")
        self.tts_model = load_tts_model(self.config.tts_model)  # ty:ignore[invalid-argument-type]
        logger.info("Models loaded.")

    def _setup_default_handlers(self) -> None:
        llm_handler = LLMCommandHandler(
            self.llm_model,
            self.tokenizer,
            self.sampler,
            self.logits_processors,
            self.config,
        )
        self.command_handlers.append(llm_handler)

    def add_command_handler(self, handler: CommandHandler) -> None:
        """Add a command handler. Handlers are checked in order."""
        self.command_handlers.insert(0, handler)

    def _transcribe(self, audio: np.ndarray) -> str:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as f:
            sf.write(f.name, audio, self.config.sample_rate)
            result = generate_transcription(model=self.stt_model, audio=f.name)
            mx.eval()
        return result.text.strip() if hasattr(result, "text") else ""

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

    def _respond_streaming(self, command: str) -> None:
        handler = None
        for h in self.command_handlers:
            if h.can_handle(command):
                handler = h
                break

        if handler is None:
            logger.error("No handler found for command")
            return

        if not isinstance(handler, LLMCommandHandler):
            response = handler.handle(command)
            logger.info(f"Response: {response}")
            self._speak(response)
            return

        sentence_buffer = ""
        full_response = ""

        for token in handler.stream(command):
            sentence_buffer += token
            full_response += token

            match = SENTENCE_END.search(sentence_buffer)
            if match:
                sentence = sentence_buffer[: match.end()]
                sentence_buffer = sentence_buffer[match.end() :]
                mx.eval()
                self._speak(sentence)

        if sentence_buffer.strip():
            self._speak(sentence_buffer)

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

        try:
            while True:
                if self.state == State.WAITING_FOR_WAKE_WORD:
                    audio = self._record()
                    if audio is None:
                        continue

                    text = self._transcribe(audio)
                    if not text:
                        continue

                    logger.info(f"Heard: {text}")

                    if self._has_wake_word(text):
                        logger.info("Wake word detected! Listening...")
                        self.state = State.LISTENING_FOR_COMMAND

                elif self.state == State.LISTENING_FOR_COMMAND:
                    audio = self._record()
                    if audio is None:
                        logger.info("No command heard. Waiting for wake word...")
                        self.state = State.WAITING_FOR_WAKE_WORD
                        continue

                    command = self._transcribe(audio)
                    if not command:
                        self.state = State.WAITING_FOR_WAKE_WORD
                        continue

                    logger.info(f"Command: {command}")
                    self.state = State.RESPONDING

                    self._respond_streaming(command)

                    logger.info("Waiting for wake word...")
                    self.state = State.WAITING_FOR_WAKE_WORD

        except KeyboardInterrupt:
            logger.info("Shutting down...")
            self._shutdown.set()

    def cleanup(self) -> None:
        del self.tts_model
        del self.llm_model
        del self.stt_model
        mx.metal.clear_cache()


class RobotCommandHandler(CommandHandler):
    """Example handler for robot commands. Extend for Raspberry Pi control."""

    COMMANDS = {
        "forward": "Moving forward",
        "backward": "Moving backward",
        "left": "Turning left",
        "right": "Turning right",
        "stop": "Stopping",
    }

    def can_handle(self, command: str) -> bool:
        command_lower = command.lower()
        return any(keyword in command_lower for keyword in self.COMMANDS)

    def handle(self, command: str) -> str:
        command_lower = command.lower()
        for keyword, response in self.COMMANDS.items():
            if keyword in command_lower:
                # Add GPIO control here for actual robot
                logger.info(f"Robot command: {keyword}")
                return response
        return "I don't understand that command."


def main() -> None:
    config = Config()
    assistant = VoiceAssistant(config)
    # assistant.add_command_handler(RobotCommandHandler())

    try:
        assistant.run()
    finally:
        assistant.cleanup()


if __name__ == "__main__":
    main()
