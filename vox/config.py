"""Application settings via pydantic-settings."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Vox TUI configuration, overridable via VOX_ env vars or .env file."""

    model_config = SettingsConfigDict(env_prefix="VOX_")

    stt_model: str = "mlx-community/parakeet-tdt-0.6b-v3"
    llm_model: str = "LiquidAI/LFM2.5-1.2B-Instruct-MLX-8bit"
    tts_model: str = "mlx-community/chatterbox-fp16"
    system_prompt: str = "You are a helpful assistant"
    llm_max_tokens: int = 4096
    audio_sample_rate: int = 16000
    silence_threshold: float = 0.01
    tts_sample_rate: int = 24000


settings = Settings()
