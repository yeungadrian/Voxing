"""Mock audio recorder for simulating voice input."""

import asyncio
import random

from reachy_tui.mocks import DEFAULT_CONFIG, MockConfig


class MockAudioRecorder:
    """Simulates audio recording with realistic delays."""

    def __init__(self, config: MockConfig = DEFAULT_CONFIG):
        """Initialize the mock audio recorder.

        Args:
            config: Configuration for timing simulation.
        """
        self.config = config

    async def record(self, text_input: str) -> tuple[str, float]:
        """Simulate recording audio input.

        Args:
            text_input: The user's text input (simulating spoken words).

        Returns:
            Tuple of (text_input, recording_duration).
        """
        # Simulate variable recording time based on input length
        base_duration = len(text_input) * 0.05  # ~0.05s per character
        variation = random.uniform(
            self.config.audio_record_min, self.config.audio_record_max
        )
        duration = min(base_duration + variation, self.config.audio_record_max)

        await asyncio.sleep(duration)

        return text_input, duration
