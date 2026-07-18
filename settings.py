"""Environment-based application configuration."""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

_DEFAULT_BASE_URL = "http://host.docker.internal:11434/v1"
_DEFAULT_API_KEY = "ollama"
_DEFAULT_MODEL = "qwen2.5:7b"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    diarizer_llm_base_url: str = _DEFAULT_BASE_URL
    diarizer_llm_api_key: str = _DEFAULT_API_KEY
    diarizer_llm_model: str = _DEFAULT_MODEL
    classifier_llm_base_url: str = _DEFAULT_BASE_URL
    classifier_llm_api_key: str = _DEFAULT_API_KEY
    classifier_llm_model: str = _DEFAULT_MODEL
    quality_llm_base_url: str = _DEFAULT_BASE_URL
    quality_llm_api_key: str = _DEFAULT_API_KEY
    quality_llm_model: str = _DEFAULT_MODEL
    compliance_llm_base_url: str = _DEFAULT_BASE_URL
    compliance_llm_api_key: str = _DEFAULT_API_KEY
    compliance_llm_model: str = _DEFAULT_MODEL
    summarizer_llm_base_url: str = _DEFAULT_BASE_URL
    summarizer_llm_api_key: str = _DEFAULT_API_KEY
    summarizer_llm_model: str = _DEFAULT_MODEL
    whisper_model: str = "medium"
    whisper_device: str = "cpu"
    whisper_compute_type: str = "int8"
    max_audio_bytes: int = Field(default=50 * 1024 * 1024, gt=0)
    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    return Settings()
