# Application settings and configuration.
    
from functools import lru_cache
from typing import List, Optional
from pydantic import field_validator, PostgresDsn, RedisDsn
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # app.
    APP_NAME: str = "MANGOS LLM Reliability Platform"
    APP_VERSION: str = "0.1.0"
    ENVIRONMENT: str = "development"  # development, staging, production
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"

    # API.
    API_PREFIX: str = "/api/v1"
    ALLOWED_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:5173"]

    # security.
    SECRET_KEY: str  # Required, no default. Will fail startup if missing.
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 15
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7
    ALGORITHM: str = "HS256"
    API_KEY_PREFIX: str = "mg_"

    # database.
    DATABASE_URL: PostgresDsn
    DATABASE_POOL_SIZE: int = 10
    DATABASE_MAX_OVERFLOW: int = 20
    DATABASE_POOL_TIMEOUT: int = 30
    DATABASE_ECHO: bool = False  # Set True in dev to log all SQL

    # redis.
    REDIS_URL: RedisDsn = "redis://localhost:6379/0"  # type: ignore
    REDIS_CACHE_TTL_SECONDS: int = 300  # 5 min default cache

    # celery.
    CELERY_BROKER_URL: str = "redis://localhost:6379/1"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/2"
    CELERY_TASK_TIMEOUT_SECONDS: int = 1800  # 30 min max per eval task
    CELERY_MAX_RETRIES: int = 3

    # LLMs.
    OPENAI_API_KEY: Optional[str] = None
    ANTHROPIC_API_KEY: Optional[str] = None
    GOOGLE_API_KEY: Optional[str] = None
    COHERE_API_KEY: Optional[str] = None
    HUGGINGFACE_API_TOKEN: Optional[str] = None

    # evaluator model.
    EVALUATOR_PROVIDER: str = "openai"
    EVALUATOR_MODEL: str = "gpt-4o-mini"

    # embeddings.
    DEFAULT_EMBEDDING_MODEL: str = "text-embedding-3-small"
    DEFAULT_EMBEDDING_PROVIDER: str = "openai"

    # ── Monitoring ────────────────────────────────────────────────────────────
    DRIFT_DETECTION_INTERVAL_MINUTES: int = 5
    DRIFT_BASELINE_WINDOW_HOURS: int = 24
    DRIFT_CURRENT_WINDOW_HOURS: int = 1
    DRIFT_KS_PVALUE_THRESHOLD: float = 0.05
    DRIFT_ZSCORE_THRESHOLD: float = 2.5

    # ── Rate Limiting ─────────────────────────────────────────────────────────
    RATE_LIMIT_DEFAULT_RPM: int = 100       # requests per minute per API key
    RATE_LIMIT_EVAL_RPM: int = 10          # eval jobs are expensive

    # storage.
    # Local disk in dev; swap for S3_BUCKET in prod via same interface
    STORAGE_BACKEND: str = "local"         # local, s3
    LOCAL_STORAGE_PATH: str = "/tmp/mangos_artifacts"
    S3_BUCKET: Optional[str] = None
    S3_REGION: Optional[str] = None

    # notifications.
    SMTP_HOST: Optional[str] = None
    SMTP_PORT: int = 587
    SMTP_USER: Optional[str] = None
    SMTP_PASSWORD: Optional[str] = None
    FROM_EMAIL: str = "alerts@mangos-platform.io"

    @field_validator("ENVIRONMENT")
    @classmethod
    def validate_environment(cls, v: str) -> str:
        allowed = {"development", "staging", "production"}
        if v not in allowed:
            raise ValueError(f"ENVIRONMENT must be one of {allowed}")
        return v

    @field_validator("LOG_LEVEL")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if v.upper() not in allowed:
            raise ValueError(f"LOG_LEVEL must be one of {allowed}")
        return v.upper()

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


@lru_cache()
def get_settings() -> Settings:
    """
    Cached settings singleton.
    lru_cache ensures we parse .env exactly once — not on every request.
    FastAPI Depends(get_settings) injects this safely throughout.
    """
    return Settings()


settings = get_settings()
