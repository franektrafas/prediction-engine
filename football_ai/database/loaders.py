"""Loads already-fetched real data frames into the canonical model + provenance layer.

Faza 6 (Canonical Match Model) + Faza 7 (Data Provenance Layer) of the Data
Engine plan. Deliberately NOT a live-PostgreSQL import job -- that's Faza
9-10, a separate, explicit step. This module's job is to prove the ORM
schema actually fits the real data already sitting in
`data/processed/*.parquet` (martj42's results/goalscorers/shootouts), and
to give Faza 9 a tested function to call rather than hand-written SQL.

Team-name canonicalization is not re-done here -- `records_frame`'s
`home_team_name`/`away_team_name` are already canonicalized by
`international_results.py`'s `normalize_results`, and `TeamRepository`
keys teams on that same canonical name, so this module only has to look
teams up, never resolve aliases itself.
"""

from __future__ import annotations

import logging
from datetime import datetime

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from football_ai.database.models import Match, MatchEvent, MatchStatisticSource
from football_ai.database.repositories import TeamRepository

LOGGER = logging.getLogger(__name__)

MARTJ42_SOURCE = "martj42"
MARTJ42_LICENSE = "CC0"
MARTJ42_URL = "https://github.com/martj42/international_results"

# Keyed (penalty, own_goal) -- verified for real (2026-08-20, all 47,914
# rows of the live goalscorers.csv) that "own_goal and penalty both True"
# never occurs; a row hitting that gap is a genuine schema surprise, not
# routine messiness, so `load_match_events` warns and skips it rather than
# silently guessing which event type it is.
_GOAL_EVENT_TYPES = {
    (True, False): "penalty",
    (False, True): "own_goal",
    (False, False): "goal",
}


def _to_python_value(value: object) -> object | None:
    """Convert a pandas NA/NaN sentinel to a real `None`, pass real values through."""
    return None if pd.isna(value) else value


def load_matches(session: Session, records_frame: pd.DataFrame) -> dict[int, Match]:
    """Insert (or find existing) `Match` rows from `records_frame`.

    Keyed on (source="martj42", external_id=str(records_frame's own
    `match_id`)) -- that `match_id` is already this project's stable
    identifier for a martj42 row (see `international_results.py`'s
    `normalize_results`), so it's reused as-is rather than inventing a new
    external id scheme.

    Returns `{records_frame match_id -> ORM Match}` so callers (events,
    provenance) can attach to the same rows without re-querying.
    """
    team_repository = TeamRepository(session)
    matches_by_id: dict[int, Match] = {}

    for row in records_frame.itertuples(index=False):
        external_id = str(row.match_id)
        existing = session.execute(
            select(Match).where(
                Match.source == MARTJ42_SOURCE, Match.external_id == external_id
            )
        ).scalar_one_or_none()
        if existing is not None:
            matches_by_id[row.match_id] = existing
            continue

        home_team = team_repository.get_or_create(row.home_team_name)
        away_team = team_repository.get_or_create(row.away_team_name)
        match_date = row.match_date
        if isinstance(match_date, pd.Timestamp):
            match_date = match_date.to_pydatetime()

        match = Match(
            external_id=external_id,
            source=MARTJ42_SOURCE,
            competition_type="international",
            tournament=_to_python_value(row.tournament),
            match_date=match_date,
            stadium_name=_to_python_value(getattr(row, "stadium_name", None)),
            city=_to_python_value(row.city),
            country=_to_python_value(row.country),
            neutral_venue=bool(row.neutral_venue),
            home_team_id=home_team.id,
            away_team_id=away_team.id,
            home_score=_to_python_value(row.home_score),
            away_score=_to_python_value(row.away_score),
        )
        session.add(match)
        session.flush()
        matches_by_id[row.match_id] = match

    return matches_by_id


def load_match_events(
    session: Session, goalscorers_frame: pd.DataFrame, matches_by_id: dict[int, Match]
) -> int:
    """Insert `MatchEvent` rows from `goalscorers_frame` (goal/own_goal/penalty).

    Requires `matches_by_id` (from `load_matches`) rather than re-resolving
    matches by name/date -- `goalscorers_frame` already carries the same
    `match_id` records_frame does (see `international_results.py`'s
    `normalize_goalscorers`, which joins on it), so this is a direct lookup,
    not a fuzzy match. Rows whose `match_id` isn't in `matches_by_id` are
    skipped (that match wasn't loaded in this run) rather than raising.

    Returns the number of events inserted.
    """
    team_repository = TeamRepository(session)
    inserted = 0
    for row in goalscorers_frame.itertuples(index=False):
        match = matches_by_id.get(row.match_id)
        if match is None:
            continue
        event_type = _GOAL_EVENT_TYPES.get((bool(row.penalty), bool(row.own_goal)))
        if event_type is None:
            LOGGER.warning(
                "Unexpected own_goal/penalty combination for match_id=%s "
                "scorer=%r; skipping.",
                row.match_id,
                row.scorer,
            )
            continue
        team = team_repository.get_or_create(row.team_name)
        session.add(
            MatchEvent(
                match_id=match.id,
                team_id=team.id,
                event_type=event_type,
                minute=_to_python_value(row.minute),
                player_name=_to_python_value(row.scorer),
                event_source=MARTJ42_SOURCE,
            )
        )
        inserted += 1
    session.flush()
    return inserted


def load_penalty_shootouts(
    session: Session, shootouts_frame: pd.DataFrame, matches_by_id: dict[int, Match]
) -> int:
    """Flag `Match.penalty_shootout`/`penalty_shootout_winner_team_id`.

    Matched by (home_team_name, away_team_name, match_date) against
    `matches_by_id`'s already-loaded matches, mirroring how
    `international_results.py` itself matches goalscorers to `match_id` --
    `shootouts.csv` has no shared numeric id to join on, only these three
    fields (see its real columns: date/home_team/away_team/winner/
    first_shooter). Returns the number of matches updated.
    """
    team_repository = TeamRepository(session)
    lookup = {
        (match.home_team_id, match.away_team_id, match.match_date.date()): match
        for match in matches_by_id.values()
    }
    updated = 0
    for row in shootouts_frame.itertuples(index=False):
        home_team = team_repository.get_or_create(row.home_team_name)
        away_team = team_repository.get_or_create(row.away_team_name)
        match_date = row.match_date
        if isinstance(match_date, pd.Timestamp):
            match_date = match_date.date()
        match = lookup.get((home_team.id, away_team.id, match_date))
        if match is None:
            continue
        winner_team = team_repository.get_or_create(row.winner_team_name)
        match.penalty_shootout = True
        match.penalty_shootout_winner_team_id = winner_team.id
        updated += 1
    session.flush()
    return updated


def record_provenance(
    session: Session,
    *,
    match: Match,
    statistic_name: str,
    home_value: float | None,
    away_value: float | None,
    source: str,
    source_license: str,
    license_status: str,
    confidence: str,
    retrieved_at: datetime,
    source_url: str | None = None,
    source_dataset: str | None = None,
    data_version: str | None = None,
) -> MatchStatisticSource:
    """Record one (match, statistic, source) provenance row.

    Does not overwrite or deduplicate against an existing row for the same
    (match, statistic) from a DIFFERENT source -- per the project's "keep
    every source, resolve canonical by priority" rule, conflicting values
    are preserved side by side, not merged here.
    """
    provenance = MatchStatisticSource(
        match_id=match.id,
        statistic_name=statistic_name,
        home_value=home_value,
        away_value=away_value,
        source=source,
        source_url=source_url,
        source_dataset=source_dataset,
        source_license=source_license,
        license_status=license_status,
        retrieved_at=retrieved_at,
        data_version=data_version,
        confidence=confidence,
    )
    session.add(provenance)
    session.flush()
    return provenance
