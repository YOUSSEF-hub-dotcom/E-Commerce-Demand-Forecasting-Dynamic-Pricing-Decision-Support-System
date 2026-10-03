from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_REFERENCE_WEEK = "2018-05-14"
DEFAULT_MODEL_NAME = "demand_forecasting_final"
DEFAULT_TRACKING_URI = "sqlite:///artifacts/mlflow_tracking.db"


class Settings(BaseSettings):
    """Runtime configuration for the deployment API.

    Values are resolved from environment variables first (see ``.env.example``),
    then from a local ``.env`` file at the repository root, then from the
    defaults below. The Production model is consumed read-only:
    ``mlflow_model_version``/``mlflow_model_alias`` only select an existing
    registered version, they never modify the registry.
    """

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- application -------------------------------------------------------
    app_env: str = "production"
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    log_level: str = "INFO"

    # --- model registry (read-only) ----------------------------------------
    mlflow_tracking_uri: str = DEFAULT_TRACKING_URI
    mlflow_model_name: str = DEFAULT_MODEL_NAME
    mlflow_model_version: str | None = "2"
    mlflow_model_alias: str | None = "Production"
    mlflow_mlruns_dir: str = "mlruns"

    # --- forecasting / explainability --------------------------------------
    reference_week: str = DEFAULT_REFERENCE_WEEK
    shap_background_size: int = 400
    shap_explain_size: int = 1500
    shap_seed: int = 20260924
    shap_top_n: int = 20

    # --- persistence --------------------------------------------------------
    database_url: str = "sqlite:///./artifacts/inference_log.db"
    persistence_timeout_s: int = 5

    # --- network -------------------------------------------------------------
    cors_allowed_origins: str = "http://localhost:8501,http://localhost:8502"
    streamlit_api_base_url: str = "http://localhost:8000"
    max_payload_bytes: int = 65536
    request_id_header: str = "X-Request-ID"

    # --- optional rate limiting (default off) ------------------------------
    rate_limit_enabled: bool = False
    rate_limit_max_requests: int = 120
    rate_limit_window_seconds: int = 60

    # ------------------------------------------------------------------------

    @field_validator("shap_background_size", "shap_explain_size", "shap_top_n")
    @classmethod
    def _positive_ints(cls, value: int) -> int:
        if value < 1:
            raise ValueError("must be a positive integer")
        return value

    @field_validator("reference_week")
    @classmethod
    def _reference_week_format(cls, value: str) -> str:
        from datetime import date

        try:
            date.fromisoformat(value)
        except ValueError as exc:
            raise ValueError("reference_week must be an ISO date YYYY-MM-DD") from exc
        return value

    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.cors_allowed_origins.split(",") if o.strip()]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()