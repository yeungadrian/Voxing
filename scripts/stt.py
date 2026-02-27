"""Test the STT pipeline in isolation.

Usage: uv run scripts/stt.py
"""

import threading

from voxing.config import Settings
from voxing.parakeet import load_model
from voxing.stt import RealtimeTranscriber

settings = Settings()

print("Loading model...")
model = load_model(settings.model_id)
print("Model loaded.\n")

input("Press Enter to start recording... ")

with RealtimeTranscriber(model, settings) as transcriber:

    def _wait_for_enter(_stop: RealtimeTranscriber = transcriber) -> None:
        """Stop transcriber when Enter is pressed."""
        input()
        _stop.stop()

    threading.Thread(target=_wait_for_enter, daemon=True).start()
    print("Recording... (press Enter to stop)\n")

    last_text = ""
    for last_text in transcriber:
        print(f"\r{last_text}", end="", flush=True)

print(f"\n\nFinal: {last_text}")
