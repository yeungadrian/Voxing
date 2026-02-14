"""Application settings via pydantic-settings."""

from pydantic_settings import BaseSettings, SettingsConfigDict

STT_MODELS: list[str] = [
    "mlx-community/parakeet-tdt-0.6b-v3",
    "mlx-community/whisper-large-v3-turbo-asr-fp16",
]

LLM_MODELS: list[str] = [
    "LiquidAI/LFM2.5-1.2B-Instruct-MLX-8bit",
    "mlx-community/Nanbeige4.1-3B-8bit",
]

TTS_MODELS: list[str] = [
    "mlx-community/chatterbox-fp16",
]


class Settings(BaseSettings):
    """Vox TUI configuration, overridable via VOX_ env vars or .env file."""

    model_config = SettingsConfigDict(env_prefix="VOX_")

    stt_model: str = STT_MODELS[0]
    llm_model: str = LLM_MODELS[0]
    tts_model: str = TTS_MODELS[0]
    system_prompt: str = "You are a helpful assistant"
    llm_max_tokens: int = 4096
    audio_sample_rate: int = 16000
    silence_threshold: float = 0.01
    tts_sample_rate: int = 24000


settings = Settings()
