"""Application configuration."""

from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # App
    app_name: str = "ATS Score Calculator API"
    app_version: str = "1.0.0"
    debug: bool = False

    # Server
    host: str = "0.0.0.0"
    port: int = 8000

    # CORS
    allowed_origins: List[str] = ["*"]

    # Upload limits
    max_file_size_mb: int = 10

    # Embeddings / similarity controls
    use_embeddings: bool = True
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_timeout_seconds: float = 6.0
    embedding_batch_size: int = 8

    # Truncation / chunking controls
    max_resume_chars: int = 12000
    max_jd_chars: int = 8000
    resume_max_chunks: int = 10
    jd_max_chunks: int = 6
    target_words_per_chunk: int = 250

    # ✅ Extraction quality thresholds
    min_extracted_chars: int = 50            # HARD reject below this (422)
    low_text_chars_threshold: int = 200      # WARN below this (not reject)

    model_config = SettingsConfigDict(
        env_prefix="ATS_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    @staticmethod
    def _split_csv(value: str) -> List[str]:
        return [v.strip() for v in value.split(",") if v.strip()]

    def model_post_init(self, __context) -> None:
        if isinstance(self.allowed_origins, str):
            self.allowed_origins = self._split_csv(self.allowed_origins)


settings = Settings()