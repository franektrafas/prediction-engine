"""Application settings."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os


@dataclass(slots=True)
class DatabaseSettings:
    """Database connection settings."""

    host: str
    port: int
    database: str
    user: str
    password: str

    @property
    def sqlalchemy_url(self) -> str:
        """Build a SQLAlchemy PostgreSQL connection string."""
        return (
            f"postgresql+psycopg://{self.user}:{self.password}"
            f"@{self.host}:{self.port}/{self.database}"
        )


@dataclass(slots=True)
class AppSettings:
    """Global application settings."""

    database: DatabaseSettings
    project_root: Path
    model_artifact_dir: Path
    log_level: str
    default_timezone: str


def _read_env(name: str, default: str) -> str:
    """Read an environment variable with a default fallback."""
    return os.getenv(name, default).strip()


def get_settings() -> AppSettings:
    """Load application settings from environment variables."""
    project_root = Path(__file__).resolve().parents[1]
    database = DatabaseSettings(
        host=_read_env("POSTGRES_HOST", "localhost"),
        port=int(_read_env("POSTGRES_PORT", "5432")),
        database=_read_env("POSTGRES_DATABASE", "football_ai"),
        user=_read_env("POSTGRES_USER", "postgres"),
        password=_read_env("POSTGRES_PASSWORD", "postgres"),
    )
    model_artifact_dir = Path(
        _read_env("MODEL_ARTIFACT_DIR", str(project_root / "models" / "artifacts"))
    )
    return AppSettings(
        database=database,
        project_root=project_root,
        model_artifact_dir=model_artifact_dir,
        log_level=_read_env("LOG_LEVEL", "INFO"),
        default_timezone=_read_env("DEFAULT_TIMEZONE", "UTC"),
    )
