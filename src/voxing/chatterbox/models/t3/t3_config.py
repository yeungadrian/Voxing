from __future__ import annotations

from dataclasses import dataclass


@dataclass
class T3Config:
    """Configuration for T3 TTS model."""

    # Text tokens
    start_text_token: int = 255
    stop_text_token: int = 0
    text_tokens_dict_size: int = 50276
    max_text_tokens: int = 2048

    # Speech tokens
    start_speech_token: int = 6561
    stop_speech_token: int = 6562
    speech_tokens_dict_size: int = 6563
    max_speech_tokens: int = 4096

    # Model architecture
    speech_cond_prompt_len: int = 375

    # Conditioning
    speaker_embed_size: int = 256

    @property
    def n_channels(self) -> int:
        """Get hidden size from config."""
        return 1024

    @classmethod
    def turbo(cls) -> T3Config:
        """Create configuration for Turbo TTS model."""
        return cls(
            text_tokens_dict_size=50276,
            speech_tokens_dict_size=6563,
            speech_cond_prompt_len=375,
        )
