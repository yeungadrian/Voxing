"""Mock speech-to-text for simulating transcription."""

import asyncio
import random

from reachy_tui.mocks import DEFAULT_CONFIG, MockConfig


class MockSTT:
    """Simulates speech-to-text transcription with realistic delays."""

    # Common transcription errors for realism
    MISHEAR_PATTERNS = {
        "weather": ["whether", "wetter"],
        "time": ["thyme", "Tim"],
        "hello": ["halo", "hullo"],
        "ricci": ["Ricky", "richie", "reach"],
    }

    def __init__(self, config: MockConfig = DEFAULT_CONFIG):
        """Initialize the mock STT.

        Args:
            config: Configuration for timing simulation.
        """
        self.config = config

    async def transcribe(self, audio_text: str) -> str:
        """Simulate transcribing audio to text.

        Args:
            audio_text: The simulated audio input (already text).

        Returns:
            Transcribed text (possibly with errors).
        """
        # Simulate processing delay
        delay = random.uniform(self.config.stt_delay_min, self.config.stt_delay_max)
        await asyncio.sleep(delay)

        # Occasionally introduce transcription errors for realism
        if random.random() < self.config.stt_error_rate:
            return self._introduce_error(audio_text)

        return audio_text

    def _introduce_error(self, text: str) -> str:
        """Introduce a realistic transcription error."""
        words = text.lower().split()
        for i, word in enumerate(words):
            if word in self.MISHEAR_PATTERNS and random.random() < 0.7:
                words[i] = random.choice(self.MISHEAR_PATTERNS[word])
                break

        return " ".join(words)
