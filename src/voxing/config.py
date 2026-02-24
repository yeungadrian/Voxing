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
