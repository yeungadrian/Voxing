"""Application settings via pydantic-settings."""

from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_STT_MODEL = "mlx-community/parakeet-tdt-0.6b-v3"
DEFAULT_LLM_MODEL = "LiquidAI/LFM2.5-1.2B-Instruct-MLX-8bit"
DEFAULT_TTS_MODEL = "mlx-community/Kokoro-82M-bf16"

STT_MODELS: tuple[str, ...] = (
    DEFAULT_STT_MODEL,
    "mlx-community/whisper-large-v3-turbo-asr-fp16",
)

LLM_MODELS: tuple[str, ...] = (
    DEFAULT_LLM_MODEL,
    "mlx-community/Nanbeige4.1-3B-8bit",
    "Qwen/Qwen3-0.6B-MLX-4bit",
)

TTS_MODELS: tuple[str, ...] = (DEFAULT_TTS_MODEL,)


class Settings(BaseSettings):
    """Vox TUI configuration, overridable via VOX_ env vars or .env file."""

    model_config = SettingsConfigDict(env_prefix="VOX_")

    stt_model: str = DEFAULT_STT_MODEL
    llm_model: str = DEFAULT_LLM_MODEL
    tts_model: str = DEFAULT_TTS_MODEL
    system_prompt: str = "You are a helpful assistant"
    llm_max_tokens: int = 4096
    audio_sample_rate: int = 16000
    silence_threshold: float = 0.01
    tts_sample_rate: int = 24000


settings = Settings()
