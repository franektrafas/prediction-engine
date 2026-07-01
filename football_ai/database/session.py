"""Database engine and session helpers."""

from __future__ import annotations

from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from football_ai.config.settings import get_settings
from football_ai.database.base import Base
from football_ai.database import models  # noqa: F401

SETTINGS = get_settings()
ENGINE = create_engine(SETTINGS.database.sqlalchemy_url, future=True)
SessionLocal = sessionmaker(bind=ENGINE, autoflush=False, autocommit=False, future=True)


def get_db_session() -> Generator[Session, None, None]:
    """Yield a SQLAlchemy session and guarantee cleanup."""
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


def create_database() -> None:
    """Create all configured tables."""
    Base.metadata.create_all(bind=ENGINE)


if __name__ == "__main__":
    create_database()
