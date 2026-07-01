"""Database repositories."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from sqlalchemy import select
from sqlalchemy.orm import Session

from football_ai.database.models import EloRating, Match, ModelVersion, Team


@dataclass(slots=True)
class TeamRepository:
    """Repository for team data."""

    session: Session

    def get_by_name(self, name: str) -> Team | None:
        """Fetch a team by canonical name."""
        query = select(Team).where(Team.name == name)
        return self.session.execute(query).scalar_one_or_none()

    def get_or_create(self, name: str, fifa_code: str | None = None) -> Team:
        """Create a team if it does not already exist."""
        team = self.get_by_name(name)
        if team is not None:
            return team
        team = Team(name=name, fifa_code=fifa_code)
        self.session.add(team)
        self.session.flush()
        return team


@dataclass(slots=True)
class MatchRepository:
    """Repository for match data."""

    session: Session

    def get_by_source_id(self, source: str, external_id: str) -> Match | None:
        """Fetch a match using source-specific identifiers."""
        query = select(Match).where(
            Match.source == source,
            Match.external_id == external_id,
        )
        return self.session.execute(query).scalar_one_or_none()

    def save(self, match: Match) -> Match:
        """Persist a match entity."""
        self.session.add(match)
        self.session.flush()
        return match


@dataclass(slots=True)
class EloRepository:
    """Repository for Elo rating snapshots."""

    session: Session

    def get_latest_before(self, team_id: int, rating_date: date) -> EloRating | None:
        """Fetch the latest rating for a team before a given date."""
        query = (
            select(EloRating)
            .where(EloRating.team_id == team_id, EloRating.rating_date <= rating_date)
            .order_by(EloRating.rating_date.desc())
            .limit(1)
        )
        return self.session.execute(query).scalar_one_or_none()


@dataclass(slots=True)
class ModelVersionRepository:
    """Repository for saved model metadata."""

    session: Session

    def get_active(self, target_name: str) -> ModelVersion | None:
        """Fetch the active model version for a target."""
        query = (
            select(ModelVersion)
            .where(ModelVersion.target_name == target_name, ModelVersion.is_active.is_(True))
            .order_by(ModelVersion.trained_at.desc())
        )
        return self.session.execute(query).scalars().first()

    def save(self, model_version: ModelVersion) -> ModelVersion:
        """Persist model version metadata."""
        self.session.add(model_version)
        self.session.flush()
        return model_version
