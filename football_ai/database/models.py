"""SQLAlchemy ORM models for the project."""

from __future__ import annotations

from datetime import date, datetime

from sqlalchemy import JSON, Boolean, CheckConstraint, Date, DateTime, Float, ForeignKey, Integer, String
from sqlalchemy import Index, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from football_ai.database.base import Base


class Team(Base):
    """National team metadata."""

    __tablename__ = "teams"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(120), unique=True, index=True)
    fifa_code: Mapped[str | None] = mapped_column(String(8), index=True)
    country_code: Mapped[str | None] = mapped_column(String(8), index=True)
    confederation: Mapped[str | None] = mapped_column(String(32))
    current_elo: Mapped[float | None] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    players: Mapped[list["Player"]] = relationship(back_populates="team")


class Player(Base):
    """Player metadata."""

    __tablename__ = "players"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    team_id: Mapped[int | None] = mapped_column(ForeignKey("teams.id"), index=True)
    external_id: Mapped[str | None] = mapped_column(String(64), index=True)
    full_name: Mapped[str] = mapped_column(String(120), index=True)
    position: Mapped[str | None] = mapped_column(String(32))
    preferred_foot: Mapped[str | None] = mapped_column(String(16))
    market_value_eur: Mapped[float | None] = mapped_column(Float)
    date_of_birth: Mapped[date | None] = mapped_column(Date)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    team: Mapped[Team | None] = relationship(back_populates="players")


class Match(Base):
    """Match metadata and final result."""

    __tablename__ = "matches"

    __table_args__ = (
        UniqueConstraint("source", "external_id", name="uq_matches_source_external_id"),
        Index("ix_matches_date_teams", "match_date", "home_team_id", "away_team_id"),
        CheckConstraint(
            "competition_type IN ('international', 'club')",
            name="ck_matches_competition_type",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    external_id: Mapped[str] = mapped_column(String(64))
    source: Mapped[str] = mapped_column(String(32), index=True)
    # Generic on purpose: keeps the door open for club football later without
    # renaming tables/classes or backfilling a migration. V1 only populates
    # "international", but every match is tagged from the start.
    competition_type: Mapped[str] = mapped_column(
        String(16), default="international", server_default="international", index=True
    )
    competition_id: Mapped[str | None] = mapped_column(String(64), index=True)
    competition_name: Mapped[str | None] = mapped_column(String(120))
    season_name: Mapped[str | None] = mapped_column(String(64))
    tournament: Mapped[str | None] = mapped_column(String(120), index=True)
    # Round/phase within the tournament ("Group Stage", "Quarter-final", ...).
    # Nullable and unpopulated by martj42 (our current results source doesn't
    # carry this) -- a landing column for sources that do, per the project's
    # "warstwowa jakość danych" rule: not every field needs every source.
    stage: Mapped[str | None] = mapped_column(String(64))
    status: Mapped[str | None] = mapped_column(String(32), default="finished")
    match_date: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    stadium_name: Mapped[str | None] = mapped_column(String(120))
    city: Mapped[str | None] = mapped_column(String(120))
    country: Mapped[str | None] = mapped_column(String(120))
    venue_latitude: Mapped[float | None] = mapped_column(Float)
    venue_longitude: Mapped[float | None] = mapped_column(Float)
    neutral_venue: Mapped[bool] = mapped_column(Boolean, default=False)
    referee: Mapped[str | None] = mapped_column(String(120))
    attendance: Mapped[int | None] = mapped_column(Integer)
    home_team_id: Mapped[int] = mapped_column(ForeignKey("teams.id"), index=True)
    away_team_id: Mapped[int] = mapped_column(ForeignKey("teams.id"), index=True)
    home_score: Mapped[int | None] = mapped_column(Integer)
    away_score: Mapped[int | None] = mapped_column(Integer)
    # Nullable, not a default-False flag: for most historical rows we simply
    # don't know whether a match went to extra time, and asserting False
    # would be a fabricated fact, not an absence of one.
    extra_time: Mapped[bool | None] = mapped_column(Boolean)
    # Unlike extra_time, this one has a real source: martj42's shootouts.csv
    # is matched by (date, home_team, away_team) in `international_results.py`
    # -- a match absent from it structurally almost always means no shootout
    # happened (group-stage/friendly matches can't go to penalties at all),
    # so False is a safe default here, not a guess.
    penalty_shootout: Mapped[bool] = mapped_column(Boolean, default=False)
    penalty_shootout_winner_team_id: Mapped[int | None] = mapped_column(
        ForeignKey("teams.id")
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class MatchEvent(Base):
    """A single in-match event (goal, card, substitution, ...).

    Currently only populated for goal-type events (`goal`/`own_goal`/
    `penalty`), sourced from martj42's goalscorers.csv via
    `international_results.py`'s `normalize_goalscorers` -- the other event
    types in the check constraint below are real canonical-model fields with
    no populating source yet, not aspirational bloat: they're named
    explicitly in the project's Data Engine plan so future sources (event
    feeds, live lineups) land in the same table without a schema change.
    """

    __tablename__ = "match_events"

    __table_args__ = (
        CheckConstraint(
            "event_type IN ('goal', 'own_goal', 'penalty', 'missed_penalty', "
            "'yellow_card', 'second_yellow', 'red_card', 'substitution', "
            "'var_event')",
            name="ck_match_events_event_type",
        ),
        Index("ix_match_events_match_type", "match_id", "event_type"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    match_id: Mapped[int] = mapped_column(ForeignKey("matches.id"), index=True)
    team_id: Mapped[int | None] = mapped_column(ForeignKey("teams.id"), index=True)
    event_type: Mapped[str] = mapped_column(String(32), index=True)
    minute: Mapped[int | None] = mapped_column(Integer)
    added_time: Mapped[int | None] = mapped_column(Integer)
    # Raw source string, not a Player FK -- same reasoning as
    # `international_results.py`'s `normalize_goalscorers`: this project has
    # no player-name-alias infrastructure (unlike `config.team_aliases` for
    # teams), and scorer spelling has been observed consistent within a
    # single source.
    player_name: Mapped[str | None] = mapped_column(String(120))
    assist_player_name: Mapped[str | None] = mapped_column(String(120))
    related_player_name: Mapped[str | None] = mapped_column(String(120))
    event_source: Mapped[str] = mapped_column(String(32))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class Lineup(Base):
    """Match lineups and player availability."""

    __tablename__ = "lineups"

    __table_args__ = (
        UniqueConstraint("match_id", "team_id", "player_id", name="uq_lineups_match_player"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    match_id: Mapped[int] = mapped_column(ForeignKey("matches.id"), index=True)
    team_id: Mapped[int] = mapped_column(ForeignKey("teams.id"), index=True)
    player_id: Mapped[int] = mapped_column(ForeignKey("players.id"), index=True)
    started: Mapped[bool] = mapped_column(Boolean, default=True)
    minutes_played: Mapped[int | None] = mapped_column(Integer)
    shirt_number: Mapped[int | None] = mapped_column(Integer)
    position: Mapped[str | None] = mapped_column(String(32))
    is_captain: Mapped[bool] = mapped_column(Boolean, default=False)
    age: Mapped[float | None] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class TeamStatistic(Base):
    """Aggregated team statistics for a match."""

    __tablename__ = "team_statistics"

    __table_args__ = (
        UniqueConstraint("match_id", "team_id", name="uq_team_statistics_match_team"),
        Index("ix_team_statistics_match_home", "match_id", "is_home"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    match_id: Mapped[int] = mapped_column(ForeignKey("matches.id"), index=True)
    team_id: Mapped[int] = mapped_column(ForeignKey("teams.id"), index=True)
    is_home: Mapped[bool] = mapped_column(Boolean, index=True)
    goals: Mapped[int | None] = mapped_column(Integer)
    xg: Mapped[float | None] = mapped_column(Float)
    xga: Mapped[float | None] = mapped_column(Float)
    possession_pct: Mapped[float | None] = mapped_column(Float)
    shots: Mapped[int | None] = mapped_column(Integer)
    shots_on_target: Mapped[int | None] = mapped_column(Integer)
    corners: Mapped[int | None] = mapped_column(Integer)
    yellow_cards: Mapped[int | None] = mapped_column(Integer)
    red_cards: Mapped[int | None] = mapped_column(Integer)
    fouls: Mapped[int | None] = mapped_column(Integer)
    offsides: Mapped[int | None] = mapped_column(Integer)
    passes_completed: Mapped[int | None] = mapped_column(Integer)
    passes_attempted: Mapped[int | None] = mapped_column(Integer)
    pressing_intensity: Mapped[float | None] = mapped_column(Float)
    duels_won: Mapped[int | None] = mapped_column(Integer)
    saves: Mapped[int | None] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class PlayerStatistic(Base):
    """Per-player match statistics."""

    __tablename__ = "player_statistics"

    __table_args__ = (
        UniqueConstraint(
            "match_id", "team_id", "player_id", name="uq_player_statistics_match_player"
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    match_id: Mapped[int] = mapped_column(ForeignKey("matches.id"), index=True)
    team_id: Mapped[int] = mapped_column(ForeignKey("teams.id"), index=True)
    player_id: Mapped[int] = mapped_column(ForeignKey("players.id"), index=True)
    minutes_played: Mapped[int | None] = mapped_column(Integer)
    goals: Mapped[int | None] = mapped_column(Integer)
    assists: Mapped[int | None] = mapped_column(Integer)
    xg: Mapped[float | None] = mapped_column(Float)
    xa: Mapped[float | None] = mapped_column(Float)
    shots: Mapped[int | None] = mapped_column(Integer)
    shots_on_target: Mapped[int | None] = mapped_column(Integer)
    touches: Mapped[int | None] = mapped_column(Integer)
    passes_completed: Mapped[int | None] = mapped_column(Integer)
    key_passes: Mapped[int | None] = mapped_column(Integer)
    tackles: Mapped[int | None] = mapped_column(Integer)
    interceptions: Mapped[int | None] = mapped_column(Integer)
    saves: Mapped[int | None] = mapped_column(Integer)
    yellow_cards: Mapped[int | None] = mapped_column(Integer)
    red_cards: Mapped[int | None] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class MatchStatisticSource(Base):
    """Data Provenance Layer: which source supplied a statistic's value, and its status.

    One row per (match, statistic, source, fetch) -- deliberately NOT a
    single canonical value per (match, statistic). When two sources disagree
    on the same statistic for the same match, both rows are kept (per the
    project's "Source Priority i konflikty" rule: never drop a source,
    resolve the canonical value by priority and record the disagreement) --
    resolving the canonical value is a query-time concern (highest
    `confidence`/`license_status` wins), not something enforced by this
    table's shape.

    `license_status`/`confidence` reuse the exact vocabulary from the
    project's own Data Engine source audit (VERIFIED/RESEARCH_ONLY/UNKNOWN/
    FORBIDDEN; HIGH/MEDIUM_HIGH/LOW/UNKNOWN) so a row here means the same
    thing it means in that audit, not a parallel taxonomy. FORBIDDEN added
    2026-08-20 after confirming ESPN-sourced statistics (already loaded
    here at import time) are barred by Disney's Terms of Use even for
    non-commercial research use, not just production ML -- stronger than
    RESEARCH_ONLY, which still permits internal experimentation.
    """

    __tablename__ = "match_sources"

    __table_args__ = (
        UniqueConstraint(
            "match_id",
            "statistic_name",
            "source",
            "retrieved_at",
            name="uq_match_sources_match_stat_source_retrieved",
        ),
        CheckConstraint(
            "license_status IN ('VERIFIED', 'RESEARCH_ONLY', 'UNKNOWN', 'FORBIDDEN')",
            name="ck_match_sources_license_status",
        ),
        CheckConstraint(
            "confidence IN ('HIGH', 'MEDIUM_HIGH', 'LOW', 'UNKNOWN')",
            name="ck_match_sources_confidence",
        ),
        Index("ix_match_sources_match_stat", "match_id", "statistic_name"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    match_id: Mapped[int] = mapped_column(ForeignKey("matches.id"), index=True)
    statistic_name: Mapped[str] = mapped_column(String(64), index=True)
    home_value: Mapped[float | None] = mapped_column(Float)
    away_value: Mapped[float | None] = mapped_column(Float)
    source: Mapped[str] = mapped_column(String(64), index=True)
    source_url: Mapped[str | None] = mapped_column(String(255))
    source_dataset: Mapped[str | None] = mapped_column(String(120))
    source_license: Mapped[str] = mapped_column(String(64))
    license_status: Mapped[str] = mapped_column(String(16))
    retrieved_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    data_version: Mapped[str | None] = mapped_column(String(32))
    confidence: Mapped[str] = mapped_column(String(16))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class EloRating(Base):
    """Historical Elo ratings."""

    __tablename__ = "elo_ratings"

    __table_args__ = (
        UniqueConstraint("team_id", "rating_date", "source", name="uq_elo_team_date_source"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    team_id: Mapped[int] = mapped_column(ForeignKey("teams.id"), index=True)
    rating_date: Mapped[date] = mapped_column(Date, index=True)
    rating: Mapped[float] = mapped_column(Float, index=True)
    rank: Mapped[int | None] = mapped_column(Integer)
    source: Mapped[str] = mapped_column(String(32), default="world_football_elo")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class Injury(Base):
    """Player injury tracking."""

    __tablename__ = "injuries"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    team_id: Mapped[int] = mapped_column(ForeignKey("teams.id"), index=True)
    player_id: Mapped[int | None] = mapped_column(ForeignKey("players.id"), index=True)
    injury_date: Mapped[date] = mapped_column(Date, index=True)
    return_date: Mapped[date | None] = mapped_column(Date)
    severity: Mapped[str | None] = mapped_column(String(32))
    description: Mapped[str | None] = mapped_column(String(255))
    source: Mapped[str | None] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class Weather(Base):
    """Weather metadata linked to a match."""

    __tablename__ = "weather"

    __table_args__ = (
        UniqueConstraint("match_id", name="uq_weather_match"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    match_id: Mapped[int] = mapped_column(ForeignKey("matches.id"), index=True)
    temperature_c: Mapped[float | None] = mapped_column(Float)
    apparent_temperature_c: Mapped[float | None] = mapped_column(Float)
    precipitation_mm: Mapped[float | None] = mapped_column(Float)
    wind_speed_kph: Mapped[float | None] = mapped_column(Float)
    humidity_pct: Mapped[float | None] = mapped_column(Float)
    pressure_hpa: Mapped[float | None] = mapped_column(Float)
    cloud_cover_pct: Mapped[float | None] = mapped_column(Float)
    weather_code: Mapped[int | None] = mapped_column(Integer)
    source: Mapped[str] = mapped_column(String(32), default="open_meteo")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class ModelVersion(Base):
    """Model registry entry."""

    __tablename__ = "model_versions"

    __table_args__ = (
        UniqueConstraint("version", "model_name", "target_name", name="uq_model_version_name"),
        Index("ix_model_versions_active", "is_active", "target_name"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    version: Mapped[str] = mapped_column(String(64), index=True)
    model_name: Mapped[str] = mapped_column(String(64), index=True)
    target_name: Mapped[str] = mapped_column(String(64), index=True)
    task_type: Mapped[str] = mapped_column(String(32))
    metrics: Mapped[dict[str, float] | None] = mapped_column(JSON)
    feature_names: Mapped[list[str] | None] = mapped_column(JSON)
    artifact_path: Mapped[str] = mapped_column(String(255))
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    trained_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class Prediction(Base):
    """Persisted prediction payloads and responses."""

    __tablename__ = "predictions"

    __table_args__ = (
        Index("ix_predictions_match_model", "match_id", "model_version_id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    match_id: Mapped[int | None] = mapped_column(ForeignKey("matches.id"), index=True)
    model_version_id: Mapped[int] = mapped_column(
        ForeignKey("model_versions.id"), index=True
    )
    request_payload: Mapped[dict[str, object]] = mapped_column(JSON)
    outcome_probabilities: Mapped[dict[str, float] | None] = mapped_column(JSON)
    predicted_scores: Mapped[dict[str, float] | None] = mapped_column(JSON)
    expected_statistics: Mapped[dict[str, float] | None] = mapped_column(JSON)
    raw_prediction: Mapped[dict[str, object] | None] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
