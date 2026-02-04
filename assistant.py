import logging
import queue
import threading
import time
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


@dataclass(slots=True)
class VoiceAssistant:
    # Models
    stt_model: nn.Module
    llm_model: nn.Module
    tts_model: nn.Module
    tokenizer: TokenizerWrapper
    # Audio
    silence_threshold: float
    input_sample_rate: int = 16000
    tts_sample_rate: int = 24000
    silence_duration: float = 0.7
    max_record_duration: float = 10.0
    # Wake/sleep phrases (must match as complete phrase)
    wake_phrases: tuple[str, ...] = (
        "hello reachy",
        "hello reachi",
        "hello richy",
        "hello richi",
        "hello ritchie",
        "hello ritchi",
        "hello ricci",
        "hello reach",
    )
    sleep_phrases: tuple[str, ...] = (
        "shutdown reachy",
        "shutdown reachi",
        "shutdown richy",
        "shutdown richi",
        "shutdown ritchie",
        "shutdown ritchi",
        "shutdown ricci",
        "shutdown reach",
        "shut down reachy",
        "shut down reachi",
        "shut down richy",
        "shut down richi",
        "shut down ritchie",
        "shut down ritchi",
        "shut down ricci",
        "shut down reach",
    )
    # LLM settings
    max_tokens: int = 512
    # Console
    console: Console = field(default_factory=Console)

    def _log(self, message: str) -> None:
        self.console.print(f"[{COLOR_COMMENT}]{message}[/{COLOR_COMMENT}]")

    def _log_bold(self, message: str, color: str = COLOR_GREEN) -> None:
        self.console.print(f"[bold {color}]{message}[/bold {color}]")

    # Lifecycle hooks - override in subclasses
    def on_startup(self) -> None:
        """Called after initialization, before entering main loop."""
        pass

    def on_wake_word(self) -> None:
        """Called when wake word is detected."""
        pass

    def on_ready_for_command(self) -> None:
        """Called after wake word, before recording command. Use for beep/delay."""
        pass

    def on_speaking(self) -> None:
        """Called when starting to speak."""
        pass

    def on_shutdown(self) -> None:
        """Called during cleanup."""
        pass

    def on_sleep(self) -> None:
        """Called when sleep word is detected, returning to standby."""
        pass

    def _transcribe(self, audio: np.ndarray) -> str:
        result = self.stt_model.generate(audio)
        return result.text.strip()

    def _has_wake_phrase(self, text: str) -> bool:
        return any(phrase in text.lower() for phrase in self.wake_phrases)

    def _has_sleep_phrase(self, text: str) -> bool:
        return any(phrase in text.lower() for phrase in self.sleep_phrases)

    def _rms(self, audio: np.ndarray) -> float:
        return float(np.sqrt(np.mean(audio**2)))

    def _normalize_audio(self, audio: np.ndarray) -> np.ndarray:
        """Normalize audio to prevent clipping."""
        audio = audio.flatten().astype(np.float32)
        max_val = np.abs(audio).max()
        if max_val > 0:
            audio = audio / max_val * 0.9
        return audio.reshape(-1, 1)

    def _create_output_stream(self) -> sd.OutputStream:
        """Create an output stream. Override in subclass for different audio output."""
        return sd.OutputStream(
            samplerate=self.tts_sample_rate,
            channels=1,
            dtype=np.float32,
            blocksize=4096,
        )

    def _write_audio(self, stream: sd.OutputStream, audio: np.ndarray) -> None:
        """Write audio to stream. Override in subclass for different audio output."""
        stream.write(self._normalize_audio(audio))

    def _record(self) -> np.ndarray | None:
        """Record audio. Override in subclass for different audio input."""
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
                has_voice = self._rms(chunk) > self.silence_threshold

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

    def _format_prompt(self, user_input: str) -> str:
        if self.tokenizer.chat_template:
            messages = [
                {
                    "role": "system",
                    "content": "You are a voice assistant. All of your responses will be output via tts so keep use of punctuation minimal.",
                },
                {"role": "user", "content": user_input},
            ]
            return self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
        return user_input

    def _tts_worker(
        self,
        phrase_queue: queue.Queue[str | None],
        audio_queue: queue.Queue[np.ndarray | None],
        stats: dict[str, float],
    ) -> None:
        """Worker thread: takes phrases, generates audio, puts in audio queue."""
        first_audio = True
        while True:
            phrase = phrase_queue.get()
            if phrase is None:
                audio_queue.put(None)
                break
            if not phrase.strip():
                continue
            detected_lang = detect(phrase)
            voice, lang_code = KOKORO_VOICES.get(detected_lang, DEFAULT_VOICE)
            for result in self.tts_model.generate(
                text=phrase, voice=voice, lang_code=lang_code, speed=1.0
            ):
                if first_audio:
                    stats["first_audio"] = time.perf_counter()
                    first_audio = False
                audio_queue.put(np.array(result.audio, dtype=np.float32))

    def _playback_worker(
        self,
        audio_queue: queue.Queue[np.ndarray | None],
        stats: dict[str, float],
        playback_done: threading.Event | None = None,
    ) -> None:
        """Worker thread: takes audio chunks and plays them with a single stream."""
        first_chunk = True
        with self._create_output_stream() as stream:
            while True:
                audio = audio_queue.get()
                if audio is None:
                    break
                if first_chunk:
                    self.on_speaking()
                    first_chunk = False
                self._write_audio(stream, audio)
        stats["playback_done"] = time.perf_counter()
        if playback_done is not None:
            playback_done.set()

    def _respond(self, user_input: str, status) -> None:
        phrase_queue: queue.Queue[str | None] = queue.Queue()
        audio_queue: queue.Queue[np.ndarray | None] = queue.Queue()
        stats: dict[str, float] = {"start": time.perf_counter()}
        playback_done = threading.Event()

        tts_thread = threading.Thread(
            target=self._tts_worker,
            args=(phrase_queue, audio_queue, stats),
            daemon=True,
        )
        playback_thread = threading.Thread(
            target=self._playback_worker,
            args=(audio_queue, stats, playback_done),
            daemon=True,
        )
        tts_thread.start()
        playback_thread.start()

        phrase_buffer = ""
        full_response = ""
        first_token = True
        token_count = 0

        def make_display(text: str, is_speaking: bool = False) -> Text:
            if is_speaking:
                spinner = Spinner("dots", style=COLOR_PURPLE)
                label = "> Speaking: "
            else:
                spinner = Spinner(
                    "dots",
                    style=COLOR_PURPLE if not audio_queue.empty() else COLOR_CYAN,
                )
                label = "> Assistant: "
            rendered = cast(Text, spinner.render(self.console.get_time()))
            return Text.assemble(rendered, " ", (label, COLOR_COMMENT), text)

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
                if first_token:
                    stats["first_token"] = time.perf_counter()
                    first_token = False
                token_count += 1
                phrase_buffer += chunk.text
                full_response += chunk.text
                live.update(make_display(full_response))

                if "\n" in phrase_buffer:
                    phrase, phrase_buffer = phrase_buffer.rsplit("\n", 1)
                    phrase_queue.put(phrase)

            if phrase_buffer.strip():
                phrase_queue.put(phrase_buffer)

            stats["llm_done"] = time.perf_counter()
            phrase_queue.put(None)

            while not playback_done.is_set():
                live.update(make_display(full_response, is_speaking=True))
                time.sleep(0.1)

        tts_thread.join()
        playback_thread.join()

        start = stats["start"]
        ttft = stats.get("first_token", start) - start
        llm_time = stats.get("llm_done", start) - start
        first_audio = stats.get("first_audio", start) - start
        total_time = stats.get("playback_done", start) - start
        tokens_per_sec = token_count / llm_time if llm_time > 0 else 0

        self._log(
            f"TTFT: {ttft:.2f}s | "
            f"LLM: {llm_time:.2f}s ({tokens_per_sec:.1f} tok/s) | "
            f"First audio: {first_audio:.2f}s | "
            f"Total: {total_time:.2f}s"
        )

        status.start()

    def run(self) -> None:
        self._log_bold("Voice Assistant Ready", COLOR_GREEN)
        self._log(f'Say "{self.wake_phrases[0]}" to start, Ctrl+C to exit')
        self.on_startup()
        awaiting_command = False

        try:
            with self.console.status(f"[bold {COLOR_BLUE}]Listening...") as status:
                while True:
                    if awaiting_command:
                        status.update(
                            f"[bold {COLOR_YELLOW}]Awake - listening for command..."
                        )
                    else:
                        status.update(f"[bold {COLOR_BLUE}]Listening...")

                    record_start = time.perf_counter()
                    audio = self._record()
                    record_time = time.perf_counter() - record_start
                    if audio is None:
                        if awaiting_command:
                            self._log("No input detected, going to sleep...")
                            self.on_sleep()
                        awaiting_command = False
                        continue

                    status.update(f"[bold {COLOR_GREEN}]Transcribing...")
                    transcribe_start = time.perf_counter()
                    text = self._transcribe(audio)
                    transcribe_time = time.perf_counter() - transcribe_start
                    if not text:
                        continue

                    audio_duration = len(audio) / self.input_sample_rate
                    self._log(
                        f"Audio: {audio_duration:.1f}s | "
                        f"Record wait: {record_time:.1f}s | "
                        f"Transcribe: {transcribe_time:.2f}s"
                    )

                    if not awaiting_command:
                        self._log(f"Heard: {text}")
                        if self._has_wake_phrase(text):
                            self._log_bold("Wake phrase detected!", COLOR_YELLOW)
                            self.on_wake_word()
                            self.on_ready_for_command()
                            awaiting_command = True
                        continue

                    if self._has_sleep_phrase(text):
                        self._log_bold("Going to standby...", COLOR_BLUE)
                        self.on_sleep()
                        awaiting_command = False
                        continue

                    self._log(f"> You: {text}")
                    self._respond(text, status)
                    self.on_ready_for_command()
        except KeyboardInterrupt:
            self._log("\nShutting down...")
        finally:
            self.on_shutdown()


def load_models(
    console: Console,
) -> tuple[nn.Module, nn.Module, nn.Module, TokenizerWrapper]:
    """Load and warm up all models. Returns (stt_model, llm_model, tts_model, tokenizer)."""
    with console.status(f"[bold {COLOR_BLUE}]Loading STT model...") as status:
        stt_model = load_stt_model("mlx-community/whisper-large-v3-turbo-asr-fp16")

        status.update(f"[bold {COLOR_BLUE}]Loading LLM model...")
        llm_model, tokenizer = load("LiquidAI/LFM2.5-1.2B-Instruct-MLX-8bit")  # ty:ignore[invalid-assignment]

        status.update(f"[bold {COLOR_BLUE}]Loading TTS model...")
        tts_model = load_tts_model("mlx-community/Kokoro-82M-bf16")  # ty:ignore[invalid-argument-type]

        status.update(f"[bold {COLOR_BLUE}]Warming up models...")
        stt_model.generate(np.zeros(16000, dtype=np.float32))

    return stt_model, llm_model, tts_model, tokenizer
