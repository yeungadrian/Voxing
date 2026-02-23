# =============================================================================
# TEXT-TO-SPEECH (TTS)
# =============================================================================
#
# TTS converts text into synthesised speech audio.
#
# APPROACHES
# ----------
# Concatenative
#   Splices pre-recorded phoneme/word fragments.  Fast, but robotic and
#   limited to the recorded vocabulary.  Used in early GPS systems, old IVR.
#
# Parametric (HMM / DNN)
#   Models predict acoustic features (mel, F0, duration) from phonemes.
#   Smoother than concatenative but still noticeably synthetic.
#
# Neural autoregressive
#   Generates mel-spectrograms or waveform samples step-by-step (Tacotron 2,
#   VALL-E).  Very natural sounding; sequential decoding means latency scales
#   linearly with output length and cannot be parallelised.
#
# Neural non-autoregressive (flow / diffusion)
#   VITS, Matcha-TTS, StyleTTS2.  Parallel decoding; near-autoregressive
#   quality at a fraction of the latency.
#
# Streaming / chunk-based (used here)
#   Emit audio incrementally as tokens are consumed.  Playback can start on
#   the first chunk, hiding most of the synthesis latency.
#
# THIS EXPERIMENT
# ---------------
# Model   : mlx-community/pocket-tts-8bit
#           Non-autoregressive, streaming, 8-bit quantised for Apple Silicon.
# Backend : MLX — uses the Neural Engine + GPU via unified memory on M-series.
# Playback: callback-driven sd.OutputStream (see audio_player.py).
#           Audio chunks are enqueued as they arrive; playback overlaps synthesis.
#
# Run with: uv run experiments/tts.py
#
# BENEFITS
# --------
# + Fully local — no API key, no usage cost, audio data stays on device.
# + Streaming: first audio plays before synthesis is complete.
# + MLX leverages the Neural Engine + GPU; fast on M-series Macs.
#
# LIMITATIONS
# -----------
# - Apple Silicon only (MLX does not run on x86 / CUDA).
# - Single built-in voice; no voice cloning or style control.
#
# ALTERNATIVES FOR AUDIO HANDLING
# --------------------------------
# Collect all audio first, then sd.play(all_audio)
#   Simplest possible approach.  First audio is delayed by the full synthesis
#   time; acceptable for short sentences, poor for long responses.
#
# sd.OutputStream.write(block) in a loop
#   Blocking write on the synthesis thread; no queue, no callback.  Simpler
#   than the callback pattern but ties up the synthesis thread during playback
#   and gives no natural place to cancel mid-sentence.
#
# Callback-driven sd.OutputStream (used here via audio_player.py)
#   Synthesis and playback run concurrently; the queue decouples their rates.
#   The callback thread must never allocate or block.
#
# soundfile.write() then subprocess play
#   Write the completed audio to a temp WAV file and open it with the system
#   player (afplay on macOS).  Works offline with no sounddevice dependency,
#   but adds file I/O and cannot stream.
# =============================================================================

from pathlib import Path

import numpy as np
from audio_player import AudioPlayer

MODEL_NAME = "mlx-community/pocket-tts-8bit"
DEMO_TEXT = (
    "Hello, this is a demonstration of local text-to-speech synthesis"
    " running entirely on device."
)


def main() -> None:
    # Defer the mlx_audio import to here — importing it at module level triggers
    # Metal initialisation which slows down any script that merely imports this file.
    from mlx_audio.tts.utils import load_model

    print(f"Loading {MODEL_NAME}...", end="", flush=True)
    model = load_model(Path(MODEL_NAME))
    print(" done.")

    # AudioPlayer lazily opens the OutputStream on the first queued chunk.
    player = AudioPlayer()
    print(f"Synthesising: {DEMO_TEXT!r}")

    # model.generate() is a generator: it yields result objects one chunk at a
    # time as synthesis progresses, rather than blocking until fully complete.
    for result in model.generate(text=DEMO_TEXT):
        # Each chunk is a variable-length float32 array.  AudioPlayer slices it
        # into fixed-size blocks and feeds them to the PortAudio callback queue.
        player.queue_audio(np.array(result.audio, dtype=np.float32))

    # stop() flushes the final partial block, enqueues the sentinel, and blocks
    # until PortAudio confirms all queued audio has been played back.
    player.stop()
    print("Playback complete.")


if __name__ == "__main__":
    main()
