#!/usr/bin/env python3
"""Real-time speech-to-text transcription using mlx-audio streaming."""

import queue
import threading
import time
import pyaudio
import numpy as np

from mlx_audio.stt.utils import load_model
from mlx_audio.stt.models.whisper.streaming import StreamingDecoder, StreamingConfig
from mlx_audio.stt.models.whisper.audio import log_mel_spectrogram

# Configuration
MODEL_NAME = "mlx-community/whisper-large-v3-turbo-asr-fp16"
SAMPLE_RATE = 16000
CHUNK_DURATION = 0.5  # Process audio every 500ms
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION)
CHANNELS = 1
FORMAT = pyaudio.paFloat32
FRAME_THRESHOLD = 15  # Lower = faster output (~0.3s lookahead)
SILENCE_THRESHOLD = 0.01  # For paFloat32 audio
SILENCE_DURATION = 2.0  # Seconds of silence before reset


class AudioCapture:
    """Non-blocking audio capture using PyAudio callback mode."""

    def __init__(self, audio_queue: queue.Queue):
        self.audio_queue = audio_queue
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.running = False

    def _callback(self, in_data, frame_count, time_info, status):
        """PyAudio callback - runs in separate thread."""
        audio_data = np.frombuffer(in_data, dtype=np.float32)
        self.audio_queue.put(audio_data)
        return (None, pyaudio.paContinue if self.running else pyaudio.paComplete)

    def start(self):
        """Start audio capture."""
        self.running = True
        self.stream = self.audio.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=SAMPLE_RATE,
            input=True,
            frames_per_buffer=CHUNK_SIZE,
            stream_callback=self._callback,
        )
        self.stream.start_stream()

    def stop(self):
        """Stop audio capture."""
        self.running = False
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.audio.terminate()


class TranscriptionWorker:
    """Worker that processes audio chunks and produces transcriptions."""

    def __init__(self, audio_queue: queue.Queue, output_callback):
        self.audio_queue = audio_queue
        self.output_callback = output_callback
        self.running = False
        self.thread = None
        self.model = None
        self.decoder = None
        self.ready_event = threading.Event()

    def _load_model(self):
        """Load the Whisper model and initialize decoder."""
        print("Loading model...")
        self.model = load_model(MODEL_NAME)
        print(f"Model dims: n_mels={self.model.dims.n_mels}")
        config = StreamingConfig(frame_threshold=FRAME_THRESHOLD)
        self.decoder = StreamingDecoder(self.model, config, language="en")
        print("Model loaded.")
        self.ready_event.set()

    def _run(self):
        """Main transcription loop."""
        try:
            self._load_model()

            audio_buffer = []
            silence_chunks = 0
            chunks_for_silence = int(SILENCE_DURATION / CHUNK_DURATION)

            while self.running:
                try:
                    # Get audio chunk with timeout
                    audio_chunk = self.audio_queue.get(timeout=0.1)
                except queue.Empty:
                    continue

                # Detect silence
                volume = np.abs(audio_chunk).mean()
                is_silence = volume < SILENCE_THRESHOLD

                if is_silence:
                    silence_chunks += 1
                    if silence_chunks >= chunks_for_silence and audio_buffer:
                        # Process any remaining audio with is_last=True
                        audio_data = np.concatenate(audio_buffer)
                        mel = log_mel_spectrogram(audio_data, n_mels=self.model.dims.n_mels)
                        result = self.decoder.decode_chunk(mel, is_last=True)
                        if result.text.strip():
                            self.output_callback(result.text, is_final=True)

                        # Reset for next utterance
                        self.decoder.reset()
                        audio_buffer = []
                        silence_chunks = 0
                    continue

                # Reset silence counter on speech
                silence_chunks = 0
                audio_buffer.append(audio_chunk)

                # Process accumulated audio
                audio_data = np.concatenate(audio_buffer)
                mel = log_mel_spectrogram(audio_data, n_mels=self.model.dims.n_mels)
                result = self.decoder.decode_chunk(mel, is_last=False)

                if result.text.strip():
                    self.output_callback(result.text, is_final=False)

        except Exception as e:
            print(f"\nError in transcription worker: {e}")
            import traceback
            traceback.print_exc()

    def start(self):
        """Start transcription worker in background thread."""
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def stop(self):
        """Stop transcription worker."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)


def main():
    """Run real-time transcription."""
    print("MLX-Audio Real-Time Transcription")
    print("=" * 50)
    print("Speak and see words appear in real-time.")
    print("Press Ctrl+C to stop.\n")

    # Thread-safe queue for audio data
    audio_queue = queue.Queue()

    # Output callback - prints transcription results
    def on_transcription(text, is_final):
        print(text, end="" if not is_final else "\n", flush=True)

    # Initialize components
    capture = AudioCapture(audio_queue)
    worker = TranscriptionWorker(audio_queue, on_transcription)

    try:
        # Start transcription worker first (loads model)
        worker.start()

        # Wait for model to load before starting capture
        print("Waiting for model to load...")
        worker.ready_event.wait()

        print("\nListening...\n")
        capture.start()

        # Keep main thread alive
        while True:
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\n\nStopping...")

    finally:
        capture.stop()
        worker.stop()
        print("Goodbye!")


if __name__ == "__main__":
    main()
