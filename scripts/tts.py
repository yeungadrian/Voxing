"""Test the TTS pipeline in isolation, mirroring the app's synthesize_and_play path.

Usage: uv run scripts/tts.py
"""

import mlx.core as mx

from voxing.config import Settings
from voxing.tts import load_tts, synthesize_and_play

settings = Settings()
print("Loading model...")
model = load_tts(settings.tts_model_id)
print(f"Model loaded (sample rate: {model.sample_rate} Hz).\n")

_EXAMPLES = [
    "Hello! This is a test of the text to speech system.",
    "The quick brown fox jumps over the lazy dog.",
    "How are you doing today? I hope you are having a wonderful day.",
    """Of course! Here's a short story for you:
Once upon a time, in a quiet little village nestled between rolling hills and a sparkling river, there lived a curious young girl named Lila. She had a wild imagination and loved exploring the woods behind her home. One sunny afternoon, while wandering deeper into the forest than she had ever gone before, Lila stumbled upon a hidden clearing.
In the center of the clearing stood an ancient tree, its branches reaching high into the sky like arms embracing the sun. Lila approached it cautiously, feeling a strange energy in the air. As she reached out to touch the trunk, a soft voice whispered in her ear:
"Welcome, child. You have found the Heart of the Forest."
Lila gasped in wonder. The voice belonged to an old owl perched on a branch above her. The owl explained that the Heart of the Forest was a magical place where dreams and stories came alive. It could grant wishes, but only if the heart behind the wish was pure and true.
Lila thought for a moment and realized she wanted to help her village, which was struggling with a drought. She wished for rain to return to the land. The owl nodded, and with a flick of its wings, the sky opened up, and a gentle rain fell over the village.
From that day on, Lila became known as the Guardian of the Heart, using her gift to bring hope and joy to those around her. And every time she looked up at the stars, she remembered the wise owl and the magic of the forest.
And so, the story of Lila and the Heart of the Forest became a legend, passed down through generations.
Would you like another story or something else?""",
]

for text in _EXAMPLES:
    print(f"Synthesizing: {text}")
    synthesize_and_play(model, text)
    print()

del model
mx.clear_cache()
print("Done.")
