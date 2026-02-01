"""Production-quality voice assistant with wake word detection using local models."""

import logging
import queue
import re
import tempfile
import threading
import time
from dataclasses import dataclass, field
from enum import Enum, auto

import numpy as np
import sounddevice as sd
import soundfile as sf
from mlx_audio.stt.generate import generate_transcription, load_model as load_stt_model
from mlx_audio.tts.utils import load_model as load_tts_model
from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_logits_processors, make_sampler
from rich.console import Console
from rich.live import Live
from rich.panel import Panel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

console = Console()


class State(Enum):
    """Voice assistant states."""

    IDLE = auto()  # Waiting for wake word
    LISTENING = auto()  # Activated, listening for command
    PROCESSING = auto()  # Running STT on captured audio
    RESPONDING = auto()  # Generating and speaking response


@dataclass
class Config:
    """Voice assistant configuration."""

    # Audio settings
    sample_rate: int = 16000
    tts_sample_rate: int = 24000
    chunk_size: int = 1600  # 100ms chunks
    audio_queue_size: int = 100  # Max queued audio chunks
    tts_queue_size: int = 50  # Max queued TTS chunks
    playback_blocksize: int = 2048  # Larger blocks for smoother playback

    # Voice activity detection
    silence_threshold: float = 0.02  # RMS energy threshold
    silence_duration: float = 0.5  # Seconds of silence before processing
    max_duration: float = 10.0  # Max recording duration
    min_duration: float = 0.3  # Min duration to process

    # Post-playback cooldown (prevents hearing own TTS output)
    cooldown_duration: float = 0.8  # Seconds to wait after playback before listening

    # Interrupt detection (during playback)
    interrupt_threshold: float = 0.15  # Higher threshold to detect speech over playback
    interrupt_duration: float = 0.3  # Seconds of loud audio to trigger interrupt

    # Wake words
    wake_words: list[str] = field(
        default_factory=lambda: ["reachy", "reach", "richy", "reechy", "hello"]
    )

    # Models
    stt_model: str = "mlx-community/whisper-large-v3-turbo-asr-fp16"
    llm_model: str = "mlx-community/Jan-v3-4B-base-instruct-8bit"
    tts_model: str = "mlx-community/Kokoro-82M-bf16"
    tts_voice: str = "af_heart"

    # LLM settings
    max_tokens: int = 512
    temperature: float = 0.1
    top_k: int = 50
    top_p: float = 0.1
    repetition_penalty: float = 1.05


SENTENCE_END_PATTERN = re.compile(r"[.!?]+\s*")


class VoiceAssistant:
    """Production-quality voice assistant with state machine architecture."""

    def __init__(self, config: Config | None = None):
        self.config = config or Config()

        # State management (thread-safe)
        self._state = State.IDLE
        self._state_lock = threading.Lock()

        # Thread control events
        self._shutdown = threading.Event()
        self._interrupt = threading.Event()
        self._playback_complete = threading.Event()
        self._playback_complete.set()  # Initially complete (nothing playing)

        # Cooldown tracking (prevents hearing own TTS output)
        self._cooldown_until: float = 0.0
        self._cooldown_lock = threading.Lock()

        # Audio queues (bounded)
        self._audio_queue: queue.Queue[np.ndarray] = queue.Queue(
            maxsize=self.config.audio_queue_size
        )
        self._tts_text_queue: queue.Queue[str | None] = queue.Queue(
            maxsize=self.config.tts_queue_size
        )
        self._playback_queue: queue.Queue[np.ndarray | None] = queue.Queue(
            maxsize=self.config.tts_queue_size
        )

        # TTS completion tracking
        self._tts_complete = threading.Event()
        self._tts_complete.set()  # Initially complete

        # Workers
        self._playback_thread: threading.Thread | None = None
        self._tts_thread: threading.Thread | None = None
        self._processing_thread: threading.Thread | None = None

        # Load models
        self._load_models()

    @property
    def state(self) -> State:
        """Get current state (thread-safe)."""
        with self._state_lock:
            return self._state

    @state.setter
    def state(self, new_state: State) -> None:
        """Set state (thread-safe) with logging."""
        with self._state_lock:
            old_state = self._state
            self._state = new_state
            if old_state != new_state:
                logger.debug(f"State transition: {old_state.name} -> {new_state.name}")

    def _load_models(self) -> None:
        """Load all ML models with progress indication."""
        console.print(
            Panel.fit("[bold blue]Loading Models[/]", border_style="blue")
        )

        console.print(f"  [dim]STT:[/] {self.config.stt_model}")
        self._stt_model = load_stt_model(self.config.stt_model)

        console.print(f"  [dim]LLM:[/] {self.config.llm_model}")
        self._llm_model, self._tokenizer = load(self.config.llm_model)
        self._sampler = make_sampler(
            temp=self.config.temperature,
            top_k=self.config.top_k,
            top_p=self.config.top_p,
        )
        self._logits_processors = make_logits_processors(
            repetition_penalty=self.config.repetition_penalty
        )

        console.print(f"  [dim]TTS:[/] {self.config.tts_model}")
        self._tts_model = load_tts_model(self.config.tts_model)  # ty:ignore[invalid-argument-type]

    def _clear_queue(self, q: queue.Queue) -> None:
        """Clear all items from a queue."""
        try:
            while True:
                q.get_nowait()
        except queue.Empty:
            pass

    def _is_in_cooldown(self) -> bool:
        """Check if we're still in post-playback cooldown."""
        with self._cooldown_lock:
            return time.monotonic() < self._cooldown_until

    def _start_cooldown(self) -> None:
        """Start the post-playback cooldown period."""
        with self._cooldown_lock:
            self._cooldown_until = time.monotonic() + self.config.cooldown_duration

    def _audio_callback(
        self, indata: np.ndarray, frames: int, time_info: object, status: object
    ) -> None:
        """Audio input callback - feeds audio to processing queue."""
        if status:
            logger.warning(f"Audio input status: {status}")

        current_state = self.state

        # Check for interrupt during response (high threshold to detect speech over playback)
        if current_state == State.RESPONDING:
            rms = np.sqrt(np.mean(indata**2))
            if rms > self.config.interrupt_threshold:
                self._interrupt.set()
            return

        # Don't capture audio during cooldown (prevents hearing TTS echo)
        if self._is_in_cooldown():
            return

        # Only queue audio when in appropriate states
        if current_state in (State.IDLE, State.LISTENING):
            try:
                self._audio_queue.put_nowait(indata.copy())
            except queue.Full:
                logger.warning("Audio queue full, dropping chunk")

    def _has_voice_activity(self, audio_data: np.ndarray) -> bool:
        """Check if audio chunk contains voice activity."""
        rms = np.sqrt(np.mean(audio_data**2))
        return rms > self.config.silence_threshold

    def _check_wake_word(self, text: str) -> bool:
        """Check if transcription contains a wake word."""
        text_lower = text.lower()
        return any(word in text_lower for word in self.config.wake_words)

    def _transcribe(self, audio_data: np.ndarray) -> str:
        """Transcribe audio data to text."""
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as f:
                sf.write(f.name, audio_data, self.config.sample_rate)
                result = generate_transcription(model=self._stt_model, audio=f.name)
            return result.text.strip() if hasattr(result, "text") else ""
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return ""

    def _queue_text_for_speech(self, text: str) -> None:
        """Queue text for TTS generation (non-blocking)."""
        if not text.strip():
            return
        try:
            self._tts_text_queue.put_nowait(text)
        except queue.Full:
            logger.warning("TTS text queue full, dropping sentence")

    def _tts_worker(self) -> None:
        """Worker thread that generates TTS audio from queued text."""
        logger.debug("TTS worker started")

        while not self._shutdown.is_set():
            try:
                text = self._tts_text_queue.get(timeout=0.5)
                if text is None:
                    # Signal end of current response
                    self._tts_complete.set()
                    continue

                # Mark TTS as active
                self._tts_complete.clear()

                # Check for interrupt before generating
                if self._interrupt.is_set():
                    self._clear_queue(self._tts_text_queue)
                    self._tts_complete.set()
                    continue

                # Generate TTS audio
                try:
                    for result in self._tts_model.generate(
                        text=text,
                        voice=self.config.tts_voice,
                        speed=1.0,
                        lang_code="a",
                    ):
                        if self._interrupt.is_set():
                            logger.info("TTS generation interrupted")
                            self._clear_queue(self._tts_text_queue)
                            break
                        audio = np.array(result.audio, dtype=np.float32)
                        try:
                            self._playback_queue.put(audio, timeout=2.0)
                        except queue.Full:
                            logger.warning("Playback queue full, dropping TTS chunk")
                except Exception as e:
                    logger.error(f"TTS generation error: {e}")

            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"TTS worker error: {e}")

        logger.debug("TTS worker stopped")

    def _normalize_audio(self, audio: np.ndarray) -> np.ndarray:
        """Normalize audio to [-1, 1] range."""
        audio = audio.flatten().astype(np.float32)
        max_val = max(abs(audio.max()), abs(audio.min()))
        if max_val > 1.0:
            audio = audio / max_val
        return audio

    def _playback_worker(self) -> None:
        """Persistent worker thread for audio playback using OutputStream for smooth audio."""
        logger.debug("Playback worker started")

        while not self._shutdown.is_set():
            try:
                # Wait for audio to play
                audio = self._playback_queue.get(timeout=0.5)
                if audio is None:
                    # Signal end of current response
                    self._start_cooldown()
                    self._playback_complete.set()
                    continue

                # Mark playback as active
                self._playback_complete.clear()

                # Check for interrupt before playing
                if self._interrupt.is_set():
                    self._clear_queue(self._playback_queue)
                    self._start_cooldown()
                    self._playback_complete.set()
                    continue

                # Normalize audio
                audio = self._normalize_audio(audio)

                # Use OutputStream for smoother playback with larger buffer
                try:
                    with sd.OutputStream(
                        samplerate=self.config.tts_sample_rate,
                        channels=1,
                        dtype=np.float32,
                        blocksize=self.config.playback_blocksize,
                    ) as stream:
                        # Write audio in chunks for smoother playback
                        chunk_size = self.config.playback_blocksize
                        for i in range(0, len(audio), chunk_size):
                            if self._interrupt.is_set():
                                break
                            chunk = audio[i : i + chunk_size]
                            stream.write(chunk)
                except Exception as e:
                    logger.error(f"Audio playback error: {e}")

            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Playback worker error: {e}")

        logger.debug("Playback worker stopped")

    def _get_llm_response_streaming(self, prompt: str) -> str:
        """Stream LLM response with sentence-level TTS."""
        if self._tokenizer.chat_template is not None:
            messages = [{"role": "user", "content": prompt}]
            formatted_prompt = self._tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
            )
        else:
            formatted_prompt = prompt

        full_response = ""
        sentence_buffer = ""
        interrupted = False

        try:
            # Use transient=True so Live display disappears when done
            with Live(
                Panel("", title="[bold magenta]Assistant[/]", border_style="magenta"),
                console=console,
                refresh_per_second=10,
                transient=True,
            ) as live:
                for response in stream_generate(
                    model=self._llm_model,
                    tokenizer=self._tokenizer,
                    prompt=formatted_prompt,
                    max_tokens=self.config.max_tokens,
                    sampler=self._sampler,
                    logits_processors=self._logits_processors,
                ):
                    # Check for interrupt
                    if self._interrupt.is_set():
                        logger.info("LLM generation interrupted by user")
                        interrupted = True
                        break

                    token_text = response.text
                    full_response += token_text
                    sentence_buffer += token_text

                    live.update(
                        Panel(full_response, title="[bold magenta]Assistant[/]", border_style="magenta")
                    )

                    # Send complete sentences to TTS (non-blocking)
                    match = SENTENCE_END_PATTERN.search(sentence_buffer)
                    if match:
                        sentence = sentence_buffer[: match.end()]
                        sentence_buffer = sentence_buffer[match.end():]
                        self._queue_text_for_speech(sentence)

            # Print final response panel
            if interrupted:
                console.print(
                    Panel(f"{full_response}\n[dim red][interrupted][/]", title="[bold magenta]Assistant[/]", border_style="magenta")
                )
            else:
                console.print(
                    Panel(full_response, title="[bold magenta]Assistant[/]", border_style="magenta")
                )

            # Queue remaining text for speech (if not interrupted)
            if not interrupted and sentence_buffer.strip():
                self._queue_text_for_speech(sentence_buffer)

        except Exception as e:
            logger.error(f"LLM generation error: {e}")

        return full_response

    def _wait_for_playback_complete(self) -> None:
        """Wait for TTS generation and audio playback to finish, then start cooldown."""
        # Signal end of TTS text stream
        try:
            self._tts_text_queue.put(None, timeout=1.0)
        except queue.Full:
            logger.warning("Could not signal TTS end - queue full")

        # Wait for TTS generation to complete
        if not self._interrupt.is_set():
            self._tts_complete.wait(timeout=60.0)

        # Signal end of audio stream
        try:
            self._playback_queue.put(None, timeout=1.0)
        except queue.Full:
            logger.warning("Could not signal playback end - queue full")

        # Wait for playback to complete (with timeout)
        if not self._interrupt.is_set():
            self._playback_complete.wait(timeout=60.0)

        # Cooldown is started by the playback worker when it processes None

    def _processing_worker(self) -> None:
        """Main audio processing loop."""
        logger.debug("Processing worker started")

        audio_buffer: list[np.ndarray] = []
        is_recording = False
        silence_chunks = 0
        recording_started_in_listening = False  # Track state when recording began

        silence_chunks_needed = int(
            self.config.silence_duration * self.config.sample_rate / self.config.chunk_size
        )
        max_chunks = int(
            self.config.max_duration * self.config.sample_rate / self.config.chunk_size
        )
        min_chunks = int(
            self.config.min_duration * self.config.sample_rate / self.config.chunk_size
        )

        while not self._shutdown.is_set():
            try:
                chunk = self._audio_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            current_state = self.state

            # Skip processing if we're responding or processing
            if current_state in (State.RESPONDING, State.PROCESSING):
                continue

            has_voice = self._has_voice_activity(chunk)

            if not is_recording:
                if has_voice:
                    is_recording = True
                    audio_buffer = [chunk]
                    silence_chunks = 0
                    # Remember what state we were in when recording started
                    recording_started_in_listening = current_state == State.LISTENING
                continue

            audio_buffer.append(chunk)

            if has_voice:
                silence_chunks = 0
            else:
                silence_chunks += 1

            # Process when silence detected or max duration reached
            should_process = (
                silence_chunks >= silence_chunks_needed or len(audio_buffer) >= max_chunks
            )

            if should_process:
                if len(audio_buffer) >= min_chunks:
                    audio_data = np.concatenate(audio_buffer, axis=0).flatten()
                    self._handle_audio(audio_data, was_listening=recording_started_in_listening)

                audio_buffer = []
                is_recording = False
                silence_chunks = 0
                recording_started_in_listening = False

        logger.debug("Processing worker stopped")

    def _handle_audio(self, audio_data: np.ndarray, was_listening: bool) -> None:
        """Handle captured audio - transcribe and respond.

        Args:
            audio_data: The captured audio samples.
            was_listening: True if we were in LISTENING state (expecting command),
                          False if we were in IDLE state (checking for wake word).
        """
        self.state = State.PROCESSING
        text = self._transcribe(audio_data)

        if not text:
            # No transcription - return to previous state
            self.state = State.LISTENING if was_listening else State.IDLE
            return

        console.print(f"[dim]Heard:[/] {text}")

        if not was_listening:
            # We were IDLE - check for wake word
            if self._check_wake_word(text):
                console.print(
                    Panel("[bold green]Wake word detected![/] Listening...", border_style="green")
                )
                self.state = State.LISTENING
                # Clear any buffered audio
                self._clear_queue(self._audio_queue)
            else:
                # No wake word - back to idle
                self.state = State.IDLE
            return

        # We were LISTENING - process as command
        console.print(Panel(f"[bold cyan]You:[/] {text}", border_style="cyan"))

        # Clear interrupt flag and switch to responding
        self._interrupt.clear()
        self.state = State.RESPONDING

        # Generate and speak response
        self._get_llm_response_streaming(text)

        # Wait for all audio to play
        self._wait_for_playback_complete()

        # Clear any audio captured during response
        self._clear_queue(self._audio_queue)

        # Reset state
        if self._interrupt.is_set():
            console.print("[dim yellow]Interrupted - listening...[/]")
            self._interrupt.clear()
            self.state = State.LISTENING
        else:
            self.state = State.IDLE
            console.print("[dim]Waiting for wake word...[/]")

    def start(self) -> None:
        """Start the voice assistant."""
        console.print(
            Panel.fit(
                "[bold]Reachy Voice Assistant[/]\n"
                f"[dim]Wake words:[/] {', '.join(self.config.wake_words)}\n"
                f"[dim]Say a wake word, then speak your request.[/]",
                border_style="blue",
            )
        )
        console.print("[dim]Waiting for wake word...[/]\n")

        # Start worker threads
        self._playback_thread = threading.Thread(
            target=self._playback_worker, daemon=True, name="PlaybackWorker"
        )
        self._playback_thread.start()

        self._tts_thread = threading.Thread(
            target=self._tts_worker, daemon=True, name="TTSWorker"
        )
        self._tts_thread.start()

        self._processing_thread = threading.Thread(
            target=self._processing_worker, daemon=True, name="ProcessingWorker"
        )
        self._processing_thread.start()

        # Start audio input stream
        try:
            with sd.InputStream(
                samplerate=self.config.sample_rate,
                channels=1,
                dtype=np.float32,
                blocksize=self.config.chunk_size,
                callback=self._audio_callback,
            ):
                while not self._shutdown.is_set():
                    sd.sleep(100)
        except KeyboardInterrupt:
            console.print("\n[bold red]Shutting down...[/]")
        finally:
            self.stop()

    def stop(self) -> None:
        """Stop the voice assistant gracefully."""
        logger.info("Stopping voice assistant")
        self._shutdown.set()

        # Wait for threads to finish
        if self._processing_thread and self._processing_thread.is_alive():
            self._processing_thread.join(timeout=2.0)

        if self._tts_thread and self._tts_thread.is_alive():
            self._tts_thread.join(timeout=2.0)

        if self._playback_thread and self._playback_thread.is_alive():
            self._playback_thread.join(timeout=2.0)

        console.print("[dim]Goodbye![/]")


def main() -> None:
    """Entry point."""
    config = Config()
    assistant = VoiceAssistant(config)
    assistant.start()


if __name__ == "__main__":
    main()
