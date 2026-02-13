"""Mock text-to-speech for simulating audio playback."""

import asyncio

from reachy_tui.mocks import DEFAULT_CONFIG, MockConfig


class MockTTS:
    """Simulates text-to-speech generation and playback."""

    def __init__(self, config: MockConfig = DEFAULT_CONFIG):
        """Initialize the mock TTS.

        Args:
            config: Configuration for timing simulation.
        """
        self.config = config

    async def generate_audio(self, text: str) -> float:
        """Simulate generating audio from text.

        Args:
            text: Text to convert to speech.

        Returns:
            Simulated audio duration in seconds.
        """
        # Calculate duration based on character count and speaking rate
        duration = len(text) / self.config.tts_chars_per_second
        # Add small processing delay
        await asyncio.sleep(0.1)
        return duration

    async def playback_audio(self, duration: float) -> None:
        """Simulate playing back generated audio.

        Args:
            duration: Audio duration in seconds.
        """
        await asyncio.sleep(duration)
