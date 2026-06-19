from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    api_title: str = "Tech Challenge Animal Condition Inference API"
    api_description: str = "API de inferencia que integra modelos Hugging Face para sintomas, imagem e modelo tabular de condicao animal."
    api_version: str = "1.0.0"
    api_env: str = "dev"

    hf_token: str = ""
    hf_symptoms_model_id: str = ""
    hf_image_model_id: str = ""
    hf_model_repo: str = ""
    hf_model_filename: str = "best_model.pkl"

    # Clinical decision defaults can be tuned per model in environment variables.
    triage_positive_labels: str = "emergencia,urgente,grave"
    triage_high_threshold: float = 0.85
    triage_medium_threshold: float = 0.60

    allowed_origins: str = "http://localhost:8501"

    @property
    def cors_origins(self) -> list[str]:
        return [origin.strip() for origin in self.allowed_origins.split(",") if origin.strip()]

    @property
    def positive_labels(self) -> list[str]:
        return [label.strip().lower() for label in self.triage_positive_labels.split(",") if label.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
