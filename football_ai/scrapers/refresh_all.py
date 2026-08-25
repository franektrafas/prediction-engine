"""Daily refresh wrapper for results-freshness sources (PROMPT_DATA_REFRESH.md).

Chains the sources that actually keep match RESULTS current -- not squads
(deliberately deferred, see PROMPT_FOOTBALLSQUADS.md), not statsbomb/fbref/
api-football.com paid tiers (decided out of scope, PLAN_SILNIKA.md's 25
sierpnia 2026 note). One source's failure is logged and skipped, never
aborting the rest -- the same resilience pattern `football_api.py` already
uses per-request. A summary is appended as one JSON line to
`refresh_log.jsonl` per run, so a run is checkable afterward without
re-running anything.

Real network access from this execution environment was verified directly
(not assumed) before writing this: `raw.githubusercontent.com`,
`v3.football.api-sports.io`, and `api.football-data.org` all returned real
HTTP responses, not a connection refusal -- unlike a prior Cowork-sandbox
session's finding (PROMPT_DATA_REFRESH.md's own warning), which does not
apply here.

Three sources, each with a genuinely different freshness story:

1. `international_results` (martj42) -- rebuilds `records_frame`/
   `training_frame`/`goalscorers_frame` from the live CSVs. This is the
   source actually driving the staleness this task exists to fix (e.g. a
   live tournament's matches missing from `records_frame`).
2. `elo` -- the current ratings snapshot, ingested into the database
   (`elo_ratings`/`teams.current_elo`). NOT a rebuild of the historical
   per-match Elo file (`international_results_modern_with_elo.parquet`) --
   that file is a static, one-time-built base with no existing
   regeneration pipeline, and building one would be new scope this task
   explicitly excludes ("Czego nie robić"). `streamlit_app._load_elo_ratings`
   already re-fetches this same snapshot live on every call (1h cache),
   independent of this job.
3. `football_api` -- a connectivity/credentials smoke check, not a batch
   write. CONFIRMED (2026-08-25, grepped every real caller): every
   consumer of `FootballApiSource` already fetches live, on demand, with
   its own short cache (`streamlit_app._football_api_fixtures`,
   `predictions/live_prediction.py`'s lineup fetch) -- there is no static
   "match results" file it feeds that a daily batch job could refresh.
   Still run daily: catches an expired `FOOTBALL_API_KEY` or a changed API
   contract early, logged, never fatal to the other two sources.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path

from sqlalchemy.orm import Session

from football_ai.config.settings import get_settings
from football_ai.data.io import read_table, write_table
from football_ai.database.session import ENGINE
from football_ai.scrapers.sources.elo import WorldFootballEloSource
from football_ai.scrapers.sources.football_api import FootballApiSource
from football_ai.scrapers.sources.international_results import (
    InternationalResultsSource,
)

LOGGER = logging.getLogger(__name__)

_DEFAULT_LOG_PATH = Path("football_ai/data/processed/refresh_log.jsonl")
_DEFAULT_RECORDS_OUTPUT = (
    "football_ai/data/processed/international_results_full_history.parquet"
)
_DEFAULT_TRAINING_OUTPUT = (
    "football_ai/data/processed/international_results_training_history.parquet"
)
_DEFAULT_GOALSCORERS_OUTPUT = (
    "football_ai/data/processed/international_results_goalscorers.parquet"
)
_DEFAULT_ESPN_SOURCE = "football_ai/data/processed/espn_international_enriched.parquet"


@dataclass(slots=True)
class SourceRefreshResult:
    """Outcome of refreshing one source -- always produced, success or failure."""

    source: str
    ok: bool
    detail: str
    rows_before: int | None = None
    rows_after: int | None = None


def refresh_international_results(
    records_output: str = _DEFAULT_RECORDS_OUTPUT,
    training_output: str = _DEFAULT_TRAINING_OUTPUT,
    goalscorers_output: str = _DEFAULT_GOALSCORERS_OUTPUT,
    elo_source: str | None = None,
    espn_source: str | None = _DEFAULT_ESPN_SOURCE,
) -> SourceRefreshResult:
    """Re-fetch martj42's live CSVs and rewrite records/training/goalscorers frames."""
    records_path = Path(records_output)
    rows_before = len(read_table(records_path)) if records_path.exists() else None
    try:
        source = InternationalResultsSource()
        resolved_elo_source = elo_source or str(get_settings().inference_data_path)
        records_frame = source.build_records_frame(
            elo_source_path=resolved_elo_source, espn_source_path=espn_source or None
        )
        write_table(records_frame, records_output)
        training_frame = source.build_training_frame(records_frame)
        write_table(training_frame, training_output)
        goalscorers_frame = source.normalize_goalscorers(
            source.fetch_goalscorers(), records_frame
        )
        write_table(goalscorers_frame, goalscorers_output)
        date_range = (
            f"{records_frame['match_date'].min().date()} -> "
            f"{records_frame['match_date'].max().date()}"
        )
        return SourceRefreshResult(
            source="international_results",
            ok=True,
            detail=f"{len(records_frame)} matches, {date_range}",
            rows_before=rows_before,
            rows_after=len(records_frame),
        )
    except Exception as error:  # noqa: BLE001 -- one source's failure must not abort the run
        LOGGER.exception("international_results refresh failed.")
        return SourceRefreshResult(
            source="international_results",
            ok=False,
            detail=str(error),
            rows_before=rows_before,
        )


def refresh_elo() -> SourceRefreshResult:
    """Fetch the current Elo ratings snapshot and persist it to the database."""
    try:
        source = WorldFootballEloSource()
        snapshot = source.fetch_current_snapshot()
        with Session(ENGINE) as session:
            source.ingest_snapshot_to_database(session, snapshot)
        return SourceRefreshResult(
            source="elo",
            ok=True,
            detail=f"{len(snapshot)} team ratings ingested",
            rows_after=len(snapshot),
        )
    except Exception as error:  # noqa: BLE001
        LOGGER.exception("elo refresh failed.")
        return SourceRefreshResult(source="elo", ok=False, detail=str(error))


def check_football_api() -> SourceRefreshResult:
    """Connectivity/credentials smoke check -- see module docstring, source 3.

    Every real consumer of `FootballApiSource` already fetches live,
    on-demand -- this exists only to catch an expired key or a broken API
    contract early, not to write any file.
    """
    if not get_settings().football_api_key:
        return SourceRefreshResult(
            source="football_api", ok=False, detail="FOOTBALL_API_KEY not configured"
        )
    try:
        source = FootballApiSource()
        fixtures = source.fetch_fixtures(fixture_date=date.today().isoformat())
        return SourceRefreshResult(
            source="football_api",
            ok=True,
            detail=f"{len(fixtures)} fixtures reachable for today",
            rows_after=len(fixtures),
        )
    except Exception as error:  # noqa: BLE001
        LOGGER.exception("football_api smoke check failed.")
        return SourceRefreshResult(source="football_api", ok=False, detail=str(error))


def run_refresh(log_path: str | Path = _DEFAULT_LOG_PATH) -> list[SourceRefreshResult]:
    """Refresh every results-freshness source and append one summary line to the log."""
    results = [refresh_international_results(), refresh_elo(), check_football_api()]
    summary = {
        "timestamp": datetime.now(UTC).isoformat(),
        "results": [
            {
                "source": result.source,
                "ok": result.ok,
                "detail": result.detail,
                "rows_before": result.rows_before,
                "rows_after": result.rows_after,
            }
            for result in results
        ],
    }
    log_file = Path(log_path)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(summary) + "\n")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Daily refresh of results-freshness sources "
        "(martj42/elo/football_api)."
    )
    parser.add_argument("--log-path", default=str(_DEFAULT_LOG_PATH))
    arguments = parser.parse_args()

    _LOG_FORMAT = "%(levelname)s %(name)s: %(message)s"
    logging.basicConfig(level=logging.INFO, format=_LOG_FORMAT)
    outcome = run_refresh(log_path=arguments.log_path)
    for one_result in outcome:
        status = "OK" if one_result.ok else "FAILED"
        print(f"[{status}] {one_result.source}: {one_result.detail}")
