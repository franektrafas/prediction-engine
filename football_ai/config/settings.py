"""Application settings."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os

from dotenv import load_dotenv


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
    inference_data_path: Path
    records_frame_path: Path
    player_reference_path: Path
    injuries_path: Path
    football_api_base_url: str
    football_api_key: str
    football_api_host: str
    football_api_timezone: str
    log_level: str
    default_timezone: str
    api_key: str


def _read_env(name: str, default: str) -> str:
    """Read an environment variable with a default fallback."""
    return os.getenv(name, default).strip()


def get_settings() -> AppSettings:
    """Load application settings from environment variables."""
    project_root = Path(__file__).resolve().parents[1]
    load_dotenv(project_root.parent / ".env")
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
    inference_data_path = Path(
        _read_env(
            "INFERENCE_DATA_PATH",
            str(project_root / "data" / "processed" / "international_results_modern_with_elo.parquet"),
        )
    )
    player_reference_path = Path(
        _read_env(
            "PLAYER_REFERENCE_PATH",
            str(project_root / "data" / "processed" / "player_reference.parquet"),
        )
    )
    injuries_path = Path(
        _read_env(
            "INJURIES_PATH",
            str(project_root / "data" / "processed" / "injuries.parquet"),
        )
    )
    records_frame_path = Path(
        _read_env(
            "RECORDS_FRAME_PATH",
            str(project_root / "data" / "processed" / "international_results_full_history.parquet"),
        )
    )
    return AppSettings(
        database=database,
        project_root=project_root,
        model_artifact_dir=model_artifact_dir,
        inference_data_path=inference_data_path,
        records_frame_path=records_frame_path,
        player_reference_path=player_reference_path,
        injuries_path=injuries_path,
        football_api_base_url=_read_env(
            "FOOTBALL_API_BASE_URL",
            "https://v3.football.api-sports.io",
        ),
        football_api_key=_read_env("FOOTBALL_API_KEY", ""),
        football_api_host=_read_env("FOOTBALL_API_HOST", "v3.football.api-sports.io"),
        football_api_timezone=_read_env("FOOTBALL_API_TIMEZONE", "Europe/Warsaw"),
        log_level=_read_env("LOG_LEVEL", "INFO"),
        default_timezone=_read_env("DEFAULT_TIMEZONE", "UTC"),
        api_key=_read_env("FOOTBALL_AI_API_KEY", ""),
    )
