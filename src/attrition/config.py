"""Application configuration via Pydantic Settings.

All tunables are loaded from environment variables (or a ``.env`` file)
with sensible defaults.  Use :func:`get_config` to obtain a cached
singleton.

Constants that define the ML schema (dropped columns, target column,
feature lists) are also defined here so that every module references a
single source of truth.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


# ── Domain constants ────────────────────────────────────────────────────────

DROPPED_COLS: list[str] = [
    "EmployeeCount",
    "EmployeeNumber",
    "Over18",
    "StandardHours",
]

TARGET_COL: str = "Attrition"

NOMINAL_CATS: list[str] = [
    "BusinessTravel",
    "Department",
    "EducationField",
    "Gender",
    "JobRole",
    "MaritalStatus",
    "OverTime",
]

# All remaining features (including ordinal-like integer scales such as
# Education, EnvironmentSatisfaction, JobInvolvement, …) are treated as
# numeric and will be scaled with StandardScaler.


# ── Pydantic Settings ──────────────────────────────────────────────────────

class AppConfig(BaseSettings):
    """Central application configuration.

    Values are resolved in order:  environment variables → ``.env`` file →
    defaults defined below.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # General ----------------------------------------------------------------
    app_env: str = "development"
    app_port: int = 7860
    app_debug: bool = False
    app_secret_key: str = "change-me"

    # Database ---------------------------------------------------------------
    database_url: str = "sqlite:///data/app.db"

    # Model ------------------------------------------------------------------
    model_version: str = "v1.0.0"
    model_dir: str = "models"
    data_path: str = "data/WA_Fn-UseC_-HR-Employee-Attrition.csv"

    # HuggingFace / Retention AI ---------------------------------------------
    hf_api_token: str | None = None
    retention_ai_enabled: bool = False
    retention_ai_timeout: int = 30

    # Monitoring & operations ------------------------------------------------
    drift_alert_threshold: float = 0.2
    prediction_log_retention_days: int = 90

    # Feature flags ----------------------------------------------------------
    fastapi_enabled: bool = False
    batch_scoring_enabled: bool = True
    export_enabled: bool = True

    # Privacy & security -----------------------------------------------------
    min_group_size: int = 5
    hash_salt: str = "change-me"
    session_timeout_minutes: int = 60
    login_max_attempts: int = 5
    login_lockout_minutes: int = 15

    # Reproducibility --------------------------------------------------------
    random_seed: int = 42


# ── Cached accessor ────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def get_config() -> AppConfig:
    """Return a cached :class:`AppConfig` singleton.

    The first call reads environment variables / ``.env``; subsequent
    calls return the same instance.
    """
    return AppConfig()
