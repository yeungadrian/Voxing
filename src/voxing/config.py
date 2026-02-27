from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="VOXING_")

    model_id: str = "mlx-community/parakeet-tdt-0.6b-v3"
    sample_rate: int = 16_000
    chunk_duration: float = 0.1
    min_audio_secs: float = 1.5
    silence_duration: float = 3.0
    silence_threshold: float = 0.01
    max_buffer_secs: float = 45.0
    draft_interval_secs: float = 0.5

    llm_model_id: str = "LiquidAI/LFM2.5-1.2B-Instruct-MLX-8bit"
    llm_max_tokens: int = 4096
    llm_temperature: float = 0.1
    llm_top_k: int = 50
    llm_top_p: float = 0.1
    llm_repetition_penalty: float = 1.05
