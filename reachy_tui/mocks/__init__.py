"""Mock components for simulating voice assistant behavior."""

from dataclasses import dataclass


@dataclass
class MockConfig:
    """Configuration for mock component timing."""

    # Audio recording simulation
    audio_record_min: float = 0.5
    audio_record_max: float = 2.5

    # STT simulation
    stt_delay_min: float = 0.1
    stt_delay_max: float = 0.3
    stt_error_rate: float = 0.05  # 5% chance of mishearing

    # LLM simulation
    llm_ttft_min: float = 0.15  # Time to first token
    llm_ttft_max: float = 0.35
    llm_token_delay_min: float = 0.03
    llm_token_delay_max: float = 0.08

    # TTS simulation
    tts_chars_per_second: float = 15.0  # Average speaking rate

    def __post_init__(self):
        """Validate configuration values."""
        assert self.audio_record_min > 0
        assert self.audio_record_max >= self.audio_record_min
        assert 0 <= self.stt_error_rate <= 1.0
        assert self.llm_ttft_min > 0
        assert self.tts_chars_per_second > 0


# Global default configuration
DEFAULT_CONFIG = MockConfig()
