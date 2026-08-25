"""Streamlit demo -- for marketing/cold-email outreach.

Direct in-process calls to PredictionService/insights/simulation/narrative,
the exact same building blocks `api/routes/matches.py` and
`api/routes/competitions.py` call over HTTP -- no FastAPI, no database. Run
with: `streamlit run football_ai/streamlit_app.py`.

Presentation layer only -- nothing here reaches into `features/`,
`training/`, `models/` or `predictions/service.py`. The audience is a sports
journalist, not an analyst: every screen is built around "can I publish this
under my own name", which is why `fact.provenance` (the real source matches
behind each sentence) is a first-class UI element rather than a debug
detail, and why targets falling back to a heuristic are labelled as such
instead of being quietly presented as model output.
"""

from __future__ import annotations

import math
import sys
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

# CONFIRMED REAL (2026-08-05): `streamlit run football_ai/streamlit_app.py`
# fails with `ModuleNotFoundError: No module named 'football_ai'` on a bare
# `pip install -e .` -- this venv's `.pth` file is never actually processed
# (verified: even a throwaway marker `.pth` dropped into site-packages
# never appears in `sys.path`, editable-install mode aside), so the package
# is only importable by accident when the *current directory* happens to be
# the repo root. Streamlit's script runner points `sys.path[0]` at this
# file's own directory (`football_ai/`) instead, which breaks that accident.
# Fixed at the source rather than by requiring callers to set `PYTHONPATH`
# -- `streamlit run football_ai/streamlit_app.py` then works from anywhere,
# regardless of this machine's editable-install state.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ruff: noqa: E402 -- imports below must follow the sys.path fix above.
import matplotlib

# Headless rendering for the PDF's matplotlib charts -- must be set before
# `pyplot` is imported, and before Streamlit (or anything else) gets a
# chance to pick a GUI backend that assumes a display exists.
matplotlib.use("Agg")

import matplotlib.colors
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

from football_ai.config.settings import get_settings
from football_ai.data.geocoding import (
    coordinates_for_city_country,
    coordinates_for_stadium,
)
from football_ai.data.io import read_table
from football_ai.data.normalization import canonicalize_team_name
from football_ai.insights import h2h, streaks
from football_ai.insights.detectors import facts_for_match
from football_ai.insights.schemas import Fact
from football_ai.narrative.renderer import render_fact
from football_ai.narrative.team_names_pl import nominative_pl
from football_ai.predictions.schemas import MatchContextPayload, PredictionResponse
from football_ai.predictions.service import PredictionService
from football_ai.scrapers.sources.elo import WorldFootballEloSource
from football_ai.scrapers.sources.football_api import FootballApiSource
from football_ai.scrapers.sources.team_recent_results_wikipedia import (
    TeamRecentResultsWikipediaSource,
    empty_recent_results,
)
from football_ai.scrapers.sources.uefa_nations_league import UefaNationsLeagueSource
from football_ai.simulation.competition import (
    compute_standings,
    simulate_final_positions,
    simulate_group_qualification,
)
from football_ai.simulation.fixture_input import SimulationInput, build_simulation_input
from football_ai.simulation.scenarios import what_if_fixed_result

_APP_NAME = "StatsDesk"
# One Wikipedia fetch returns the team's whole 2020-present page anyway, so
# parsing all of it costs nothing extra; the *displayed* enrichment stays
# inside the form window, and the full set is used only to tell whether a
# referee is a repeat visitor (see `_referee_appearance_count`).
_WIKIPEDIA_PAGE_LIMIT = 500
# Below this many matches together, "this referee has taken charge of them
# before" is an anecdote, not a pattern -- so it isn't shown at all.
_MINIMUM_REFEREE_APPEARANCES = 3
_APP_TAGLINE = (
    "Prognozy i konteksty meczów reprezentacji narodowych, liczone na realnych "
    "danych historycznych — z możliwością sprawdzenia każdego faktu."
)

# Brand assets live as SVG and are rasterised once by
# `assets/branding/rasterize_logos.py` -- the PNGs are what Streamlit, Plotly
# and fpdf2 all consume, and none of them needs the SVG.
_BRANDING_DIRECTORY = Path(__file__).resolve().parent / "assets" / "branding"
_LOCKUP_DARK = _BRANDING_DIRECTORY / "statsdesk_lockup_dark.png"
_LOCKUP_LIGHT = _BRANDING_DIRECTORY / "statsdesk_lockup_light.png"
_MARK = _BRANDING_DIRECTORY / "statsdesk_mark.png"
_WATERMARK_DARK = _BRANDING_DIRECTORY / "statsdesk_watermark_dark.png"
_WATERMARK_LIGHT = _BRANDING_DIRECTORY / "statsdesk_watermark_light.png"

st.set_page_config(
    page_title=_APP_NAME,
    page_icon=str(_MARK) if _MARK.exists() else "⚽",
    layout="wide",
)

# Kept in sync by hand with .streamlit/config.toml -- Streamlit exposes the
# active theme only through st.get_option(), which returns None when a value
# is inherited rather than set, so mirroring the few colours the Plotly
# figures need is more reliable than reading them back at runtime.
_TEXT_COLOUR = "#e8eaed"
_MUTED_TEXT_COLOUR = "#9aa3b2"
_GRID_COLOUR = "#2b3345"
# 1X2 is three different semantic categories, so it gets three distinct hues
# rather than one colour at three lengths -- muted, not a betting board. The
# percentage is always drawn as text too, so the chart never depends on
# colour alone to be readable.
_HOME_COLOUR = "#4a7c9e"
_DRAW_COLOUR = "#8a8f98"
_AWAY_COLOUR = "#b5714e"

_RECORDS_FRAME_COLUMNS = [
    "match_id",
    "match_date",
    "home_team_name",
    "away_team_name",
    "home_score",
    "away_score",
    "tournament",
]
_GOALSCORERS_FRAME_COLUMNS = [
    "match_id",
    "match_date",
    "home_team_name",
    "away_team_name",
    "team_name",
    "scorer",
    "own_goal",
]

_FACT_TYPE_LABELS_PL: dict[str, str] = {
    "win_streak": "Seria zwycięstw",
    "unbeaten_streak": "Seria bez porażki",
    "clean_sheet_streak": "Seria czystych kont",
    "scoreless_streak": "Seria bez gola",
    "biggest_win": "Najwyższe zwycięstwo",
    "longest_win_streak_ever": "Rekord: seria zwycięstw",
    "longest_unbeaten_streak_ever": "Rekord: seria bez porażki",
    "match_count_milestone": "Jubileusz występów",
    "goal_count_milestone": "Jubileusz bramkowy",
    "player_goal_milestone": "Jubileusz zawodnika",
    "h2h_never_won": "Bilans bezpośredni",
    "h2h_first_win_since": "Bilans bezpośredni",
}

# Every target `PredictionService.predict()` fills in, so the diagnostic
# panel can distinguish "no model registered, hardcoded default used" from
# "target absent from this response" -- a target that silently fell back to
# a default would otherwise be indistinguishable from a real model output.
_DIAGNOSTIC_TARGETS: tuple[tuple[str, str], ...] = (
    ("match_outcome", "Wynik meczu (1X2)"),
    ("home_goals", "Bramki gospodarza"),
    ("away_goals", "Bramki gościa"),
    ("home_xg", "xG gospodarza"),
    ("away_xg", "xG gościa"),
    ("btts", "Obie drużyny strzelą"),
    ("over_1_5", "Powyżej 1.5 bramki"),
    ("over_2_5", "Powyżej 2.5 bramki"),
    ("over_3_5", "Powyżej 3.5 bramki"),
    ("home_corners", "Rzuty rożne gospodarza"),
    ("away_corners", "Rzuty rożne gościa"),
    ("home_yellow_cards", "Żółte kartki gospodarza"),
    ("away_yellow_cards", "Żółte kartki gościa"),
    ("home_shots", "Strzały gospodarza"),
    ("away_shots", "Strzały gościa"),
    ("home_shots_on_target", "Strzały celne gospodarza"),
    ("away_shots_on_target", "Strzały celne gościa"),
    ("home_possession", "Posiadanie gospodarza"),
    ("away_possession", "Posiadanie gościa"),
)

# Facts already presented as structured context (head-to-head + current
# streaks) are filtered out of the ranked "unusual facts" list, so the same
# information can't appear twice on one screen.
_STRUCTURAL_FACT_TYPES: frozenset[str] = frozenset(
    {
        "h2h_first_win_since",
        "h2h_never_won",
        "win_streak",
        "unbeaten_streak",
        "scoreless_streak",
        "clean_sheet_streak",
    }
)

_WEEKDAYS_PL: tuple[str, ...] = (
    "Poniedziałek",
    "Wtorek",
    "Środa",
    "Czwartek",
    "Piątek",
    "Sobota",
    "Niedziela",
)
_MONTHS_GENITIVE_PL: tuple[str, ...] = (
    "stycznia",
    "lutego",
    "marca",
    "kwietnia",
    "maja",
    "czerwca",
    "lipca",
    "sierpnia",
    "września",
    "października",
    "listopada",
    "grudnia",
)

# (background, text) per result marker. The letter carries the meaning and
# colour only reinforces it, so the row still reads in greyscale or with
# colour-vision deficiency. Text is a dark shade of the marker's own hue
# rather than black; every pair clears 6.8:1, well past the 4.5:1 AA floor.
_RESULT_MARKER_STYLES: dict[str, tuple[str, str]] = {
    "W": ("#a8c4b0", "#1d3326"),
    "R": ("#c2c6cc", "#262a30"),
    "P": ("#d4a99a", "#3d2019"),
}

# Numeric rows for the per-team metrics heat map (label, records_frame stat,
# cell text format) -- shown only where a majority of the rendered matches
# actually have the data (see `_row_has_majority_coverage`). "Gole
# strzelone"/"Gole stracone" aren't here because they're core to every
# played match and always shown; "cards" is synthetic (yellow + red summed).
_TEAM_METRIC_ROW_SPECS: tuple[tuple[str, str, str], ...] = (
    ("Posiadanie piłki (%)", "possession_pct", "{:.0f}%"),
    ("Strzały celne", "shots_on_target", "{:.0f}"),
    ("Kartki (Ż+Cz)", "cards", "{:.0f}"),
)
_MISSING_CELL_TEXT = "brak"

# A darker top stop than `_heatmap_figure`'s shared gold scale (#5c4a12):
# this grid pairs cell text with pure black/white (`_readable_cell_text_color`)
# rather than the app's branded off-white/off-black. MEASURED (2026-08-05):
# the branded pair (#f2ede0/#20242c) falls as low as 3.65:1 mid-scale --
# neither is extreme enough to guarantee 4.5:1 against a colour whose
# luminance sits near the black/white "crossover" point. Pure black/white
# is: the worst point achievable by *any* choice of the better of the two,
# at any luminance, is a fixed 4.583:1 (solving contrast-to-black ==
# contrast-to-white), and this ramp's true worst point measures 4.590:1 --
# essentially that theoretical floor, not a coincidence of this particular
# endpoint. No darker/lighter endpoint choice can do meaningfully better
# with a binary black/white pairing; it clears WCAG's 4.5:1 with real
# margin regardless of exactly where the ramp's dark stop sits.
_METRIC_HEATMAP_COLORSCALE: tuple[tuple[float, str], ...] = (
    (0.0, "#f0ead9"),
    (0.5, "#c9a227"),
    (1.0, "#3a2c08"),
)

_MATCH_CONTEXT_LABELS_PL: dict[str, tuple[str, str]] = {
    "home_injured_market_value": ("Wartość kontuzjowanych (gospodarz)", "money"),
    "away_injured_market_value": ("Wartość kontuzjowanych (gość)", "money"),
    "home_lineup_market_value": ("Wartość składu (gospodarz)", "money"),
    "away_lineup_market_value": ("Wartość składu (gość)", "money"),
    "home_goalkeeper_form": ("Forma bramkarza (gospodarz)", "number"),
    "away_goalkeeper_form": ("Forma bramkarza (gość)", "number"),
    "home_forward_form": ("Forma napastnika (gospodarz)", "number"),
    "away_forward_form": ("Forma napastnika (gość)", "number"),
}


# --------------------------------------------------------------------------
# Data loading (cached once per server process)
# --------------------------------------------------------------------------


@st.cache_resource
def _load_prediction_service() -> PredictionService:
    """Load once per server process -- reads local model artifacts, no network."""
    return PredictionService.create()


@st.cache_data(ttl=3600, show_spinner=False)
def _load_elo_ratings() -> dict[str, float]:
    """Live Elo, cached for an hour -- same degrade-gracefully contract as routes."""
    try:
        source = WorldFootballEloSource()
        snapshot = source.fetch_current_snapshot()
        return WorldFootballEloSource.latest_ratings_by_team(snapshot)
    except Exception:
        return {}


@st.cache_data(show_spinner=False)
def _load_records_frame() -> pd.DataFrame:
    """Same degrade-to-empty-but-correctly-shaped contract as api/routes/matches.py."""
    path = get_settings().records_frame_path
    if not path.exists():
        return pd.DataFrame(columns=_RECORDS_FRAME_COLUMNS)
    return read_table(path)


@st.cache_data(show_spinner=False)
def _load_goalscorers_frame() -> pd.DataFrame:
    """Same degrade-to-empty-but-correctly-shaped contract as api/routes/matches.py."""
    path = get_settings().goalscorers_frame_path
    if not path.exists():
        return pd.DataFrame(columns=_GOALSCORERS_FRAME_COLUMNS)
    return read_table(path)


@st.cache_data(show_spinner=False)
def _known_team_names() -> list[str]:
    """Real team names from the records frame, for a dropdown not free text."""
    frame = _load_records_frame()
    if frame.empty:
        return []
    teams = (
        pd.concat([frame["home_team_name"], frame["away_team_name"]]).dropna().unique()
    )
    return sorted(str(team) for team in teams)


@st.cache_data(show_spinner=False)
def _predictable_team_names() -> list[str]:
    """Teams the engine can actually reason about: real history *and* a live Elo rating.

    CONFIRMED REAL BUG this exists to stop (2026-08-03): the picker used to
    offer every name in the historical record, including defunct states
    (Czechoslovakia, Yugoslavia), Crown dependencies (Jersey, Alderney) and
    non-FIFA sides (Panjab) -- 109 of 337 names have no rating on
    eloratings.net. With no Elo, `home_elo`/`away_elo` reach the model as
    NaN, its own imputer fills them with the training median, and the model
    then reasons about a non-FIFA XI as if it were an average national team
    -- which, plus home advantage, produced "Panjab 54% to beat Sweden, 3:1".
    The model itself is fine: San Marino, which *does* have a real rating of
    825, correctly comes out at 7% against Sweden. So this is a presentation
    bug (offering unanswerable questions), not a modelling one.

    Falls back to every known team if the Elo fetch failed, so a network
    hiccup degrades to the old behaviour instead of an empty dropdown.
    """
    elo_ratings = _load_elo_ratings()
    if not elo_ratings:
        return _known_team_names()
    return [team for team in _known_team_names() if team in elo_ratings]


@st.cache_data(ttl=1800, show_spinner=False)
def _football_api_fixtures() -> pd.DataFrame:
    """Upcoming fixtures from api-football.com, or an empty frame.

    MEASURED, not assumed (2026-08-03, this project's own key): the free
    plan rejects any date outside today +/- 1 ("try from 2026-08-02 to
    2026-08-04") and blocks the `next` parameter outright, and what it does
    return for those days is club football -- the sample came back as 1 Lyga,
    3. Division, A Lyga and similar. National-team windows are months apart,
    so on this plan the endpoint cannot see a single upcoming international.

    Kept wired up rather than deleted because the shape of the problem is a
    billing one: on a paid plan the same call starts returning real
    multi-competition fixtures and the home page picks them up with no code
    change. Everything is filtered against the engine's own known team names
    so club sides can never leak into a national-team product.
    """
    settings = get_settings()
    if not settings.football_api_key:
        return pd.DataFrame()
    national_team_names = set(_known_team_names())
    collected: list[pd.DataFrame] = []
    try:
        source = FootballApiSource()
        for day_offset in range(0, 3):
            fixture_date = (date.today() + timedelta(days=day_offset)).isoformat()
            try:
                fixtures = source.fetch_fixtures(fixture_date=fixture_date)
            except Exception:
                # A rejected date is the documented free-plan behaviour, not
                # a failure worth aborting the whole sweep for.
                continue
            if not fixtures.empty:
                collected.append(fixtures)
    except Exception:
        return pd.DataFrame()
    if not collected:
        return pd.DataFrame()

    fixtures = pd.concat(collected, ignore_index=True)
    fixtures["home_team_name"] = fixtures["home_team_name"].map(canonicalize_team_name)
    fixtures["away_team_name"] = fixtures["away_team_name"].map(canonicalize_team_name)
    is_international = fixtures["home_team_name"].isin(national_team_names) & fixtures[
        "away_team_name"
    ].isin(national_team_names)
    fixtures = fixtures.loc[is_international].copy()
    if fixtures.empty:
        return pd.DataFrame()
    fixtures["tournament"] = fixtures["competition_name"].fillna(fixtures["tournament"])
    fixtures["source_label"] = "football_api"
    return fixtures


@st.cache_data(ttl=1800, show_spinner=False)
def _nations_league_schedule(season: str) -> pd.DataFrame:
    """Every unplayed Nations League fixture across all four divisions.

    Sixteen Wikipedia section fetches, so it is cached for half an hour --
    the home page must never pay this per script re-run.
    """
    frames: list[pd.DataFrame] = []
    for division in ("A", "B", "C", "D"):
        for group in (1, 2, 3, 4):
            fixtures = _nations_league_fixtures(division, group, season)
            if fixtures.empty:
                continue
            unplayed = fixtures.loc[fixtures["home_score"].isna()].copy()
            if unplayed.empty:
                continue
            unplayed["division"] = division
            unplayed["group"] = group
            frames.append(unplayed)
    if not frames:
        return pd.DataFrame()
    schedule = pd.concat(frames, ignore_index=True)
    schedule["tournament"] = "UEFA Nations League"
    schedule["source_label"] = "wikipedia"
    return schedule.sort_values("match_date").reset_index(drop=True)


@st.cache_data(ttl=1800, show_spinner=False)
def _upcoming_fixtures(season: str) -> tuple[pd.DataFrame, bool]:
    """All known upcoming national-team fixtures, plus whether football_api contributed.

    Wikipedia's Nations League tables are the dependable source; football_api
    is layered on when its plan actually returns internationals. The flag
    lets the page say so discreetly instead of pretending the schedule is
    broader than it is.
    """
    schedule = _nations_league_schedule(season)
    api_fixtures = _football_api_fixtures()
    api_contributed = not api_fixtures.empty
    frames = [frame for frame in (schedule, api_fixtures) if not frame.empty]
    if not frames:
        return pd.DataFrame(), api_contributed
    columns = [
        "match_date",
        "home_team_name",
        "away_team_name",
        "tournament",
        "source_label",
    ]
    optional = ["stadium", "stadium_name", "division", "group"]
    combined = pd.concat(
        [frame.reindex(columns=columns + optional) for frame in frames],
        ignore_index=True,
    )
    combined["match_date"] = pd.to_datetime(combined["match_date"], utc=True)
    combined = combined.sort_values("match_date").reset_index(drop=True)
    return combined, api_contributed


@st.cache_data(show_spinner=False)
def _tournaments_by_recency() -> list[str]:
    """Real competition names, most-recently-played first.

    A journalist picks a competition they're covering, then a match inside it
    -- far closer to how the work actually starts than typing two team names
    into an alphabetical list of every national side that ever played.
    """
    frame = _load_records_frame()
    if frame.empty or "tournament" not in frame.columns:
        return []
    latest = (
        frame.groupby("tournament")["match_date"].max().sort_values(ascending=False)
    )
    return [str(tournament) for tournament in latest.index]


@st.cache_data(show_spinner=False)
def _major_recent_tournaments(limit: int = 8) -> list[str]:
    """The competitions worth putting on a filter row, busiest-first.

    Deliberately *not* "most recently played": that ordering is dominated by
    one-off invitational tournaments (Diamond Jubilee, Unity Cup, Tri-Nations
    Cup), and measuring it showed none of the eight it produced contained a
    single match with recorded statistics -- the stat coverage lives in World
    Cup qualifying and the Nations League, which the recency ordering pushed
    off the list entirely. Ranking by how many matches a competition has
    actually played in the last two years surfaces those instead.
    """
    frame = _load_records_frame()
    if frame.empty or "tournament" not in frame.columns:
        return []
    cutoff = frame["match_date"].max() - pd.Timedelta(days=730)
    recent = frame.loc[frame["match_date"] >= cutoff]
    if recent.empty:
        return _tournaments_by_recency()[:limit]
    return [str(name) for name in recent["tournament"].value_counts().head(limit).index]


@st.cache_data(show_spinner=False)
def _played_matches_in_tournament(tournament: str, limit: int = 500) -> pd.DataFrame:
    """The most recent real, played matches of one competition."""
    frame = _load_records_frame()
    if frame.empty:
        return pd.DataFrame(columns=_RECORDS_FRAME_COLUMNS)
    subset = frame.loc[frame["tournament"] == tournament].sort_values(
        "match_date", ascending=False
    )
    return subset.head(limit).reset_index(drop=True)


@st.cache_data(ttl=3600, show_spinner=False)
def _nations_league_fixtures(division: str, group: int, season: str) -> pd.DataFrame:
    """Real Nations League fixtures for one group, straight from Wikipedia.

    The only real source of *upcoming* fixtures this project has -- the
    historical records frame is played-matches-only (verified: zero rows with
    a null score), so a pre-match screen can't be driven from it.
    """
    try:
        source = UefaNationsLeagueSource()
        return source.fetch_group_matches(division, group, season=season)
    except Exception:
        return pd.DataFrame()


def _format_fixture(row: pd.Series) -> str:
    """One fixture as a picker label: date, both sides in Polish, score if played."""
    match_date = pd.Timestamp(row["match_date"]).date()
    home = nominative_pl(str(row["home_team_name"]))
    away = nominative_pl(str(row["away_team_name"]))
    home_score = row.get("home_score")
    away_score = row.get("away_score")
    if pd.isna(home_score) or pd.isna(away_score):
        return f"{match_date} · {home} — {away}"
    return f"{match_date} · {home} {int(home_score)}:{int(away_score)} {away}"


@st.cache_data(show_spinner=False)
def _match_lookup() -> dict[int, dict[str, object]]:
    """match_id -> that match's real identity, for rendering `fact.provenance`.

    Built once and cached: a `Fact` carries only match_ids, so every
    provenance expansion in the UI needs this to turn them back into the
    real matches ("2022-11-17, Polska 1-0 Chile") a journalist can check.
    """
    frame = _load_records_frame()
    if frame.empty or "match_id" not in frame.columns:
        return {}
    columns = [column for column in _RECORDS_FRAME_COLUMNS if column in frame.columns]
    lookup: dict[int, dict[str, object]] = {}
    for row in frame[columns].itertuples(index=False):
        record = dict(zip(columns, row, strict=True))
        lookup[int(record["match_id"])] = record
    return lookup


# --------------------------------------------------------------------------
# Cached engine calls -- keyed on plain values so a re-run is free
# --------------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def _cached_prediction(
    home: str, away: str, match_date: date, tournament: str
) -> PredictionResponse:
    """One real `PredictionService.predict()` per (home, away, date, tournament)."""
    elo_ratings = _load_elo_ratings()
    payload = MatchContextPayload(
        match_date=pd.Timestamp(match_date, tz="UTC"),
        home_team_name=home,
        away_team_name=away,
        tournament=tournament,
        home_elo=elo_ratings.get(canonicalize_team_name(home)),
        away_elo=elo_ratings.get(canonicalize_team_name(away)),
    )
    return _load_prediction_service().predict(payload)


@st.cache_data(show_spinner=False)
def _cached_head_to_head(home: str, away: str, as_of_iso: str) -> pd.DataFrame:
    """Full head-to-head history from the home team's perspective.

    `insights.h2h.h2h_matches` already derives `team_a_result` per meeting,
    so nothing here recomputes a result -- the app only counts and formats.
    """
    return h2h.h2h_matches(_load_records_frame(), home, away, pd.Timestamp(as_of_iso))


@st.cache_data(show_spinner=False)
def _cached_recent_form(team: str, as_of_iso: str, limit: int = 10) -> pd.DataFrame:
    """A team's most recent matches, oldest first, goals_for/against attached."""
    matches = streaks.team_matches(_load_records_frame(), team, pd.Timestamp(as_of_iso))
    return matches.tail(limit)


@st.cache_data(ttl=21600, show_spinner=False)
def _cached_wikipedia_results(team: str) -> pd.DataFrame:
    """Attendance/referee for recent matches; one Wikipedia fetch per six hours.

    Long TTL because this is the slowest thing the app touches (a real HTTP
    round trip) and the least volatile -- a played match's attendance never
    changes. Degrades to an empty frame on any failure, so every caller can
    treat "we have nothing for this team" and "the fetch broke" identically.
    """
    try:
        source = TeamRecentResultsWikipediaSource()
        return source.fetch_recent_results(team, limit=_WIKIPEDIA_PAGE_LIMIT)
    except Exception:
        return empty_recent_results()


@st.cache_data(ttl=21600, show_spinner=False)
def _wikipedia_details_by_date(team: str) -> dict[date, dict[str, object]]:
    """`{match date: {attendance, referee, goals_for, goals_against}}` for one team.

    Keyed on date alone: a national team plays at most once a day, and the
    opponent name is the one field that *can't* be relied on to match -- the
    FIFA code table here is UEFA-shaped, so Wikipedia's `{{fb|NGA}}` stays
    "NGA" where the records frame says "Nigeria". Callers verify the join by
    score instead (see `_wikipedia_match_detail`).
    """
    frame = _cached_wikipedia_results(team)
    if frame.empty:
        return {}
    return {
        pd.Timestamp(row["match_date"]).date(): {
            "attendance": row["attendance"],
            "referee": row["referee"],
            "goals_for": row["goals_for"],
            "goals_against": row["goals_against"],
        }
        for _, row in frame.iterrows()
    }


def _wikipedia_match_detail(
    team: str, match_date: date, goals_for: float, goals_against: float
) -> dict[str, object] | None:
    """One match's attendance/referee, if the score confirms it's that match.

    The date join is near-certain on its own; requiring the score to agree
    makes it self-checking, so a mismatch between the two sources shows up
    as missing enrichment rather than as a confidently-wrong attendance
    figure attached to the wrong game.
    """
    detail = _wikipedia_details_by_date(team).get(match_date)
    if detail is None:
        return None
    recorded = (int(goals_for), int(goals_against))
    if (detail["goals_for"], detail["goals_against"]) != recorded:
        return None
    return detail


def _format_match_detail(detail: dict[str, object] | None) -> str:
    """The muted `12 345 widzów · Sędzia (Kraj)` suffix, or an empty string.

    Empty is deliberate: a match with neither field shows nothing at all,
    not a placeholder. Attendance of 0 is a real figure (COVID
    behind-closed-doors) and must still render, which is why this tests for
    None rather than truthiness.
    """
    if detail is None:
        return ""
    parts: list[str] = []
    attendance = detail.get("attendance")
    if attendance is not None and not pd.isna(attendance):
        count = int(attendance)
        spectators = _plural_pl(count, "widz", "widzów", "widzów")
        parts.append(f"{count:,}".replace(",", " ") + f" {spectators}")
    referee = detail.get("referee")
    if referee is not None and not pd.isna(referee):
        parts.append(f"sędzia: {referee}")
    return " · ".join(parts)


@st.cache_data(ttl=21600, show_spinner=False)
def _referee_appearance_count(team: str, referee: str) -> int:
    """How many of this team's matches on record that referee has taken charge of.

    Counted over the whole page (2020-present), not the ten-match form
    window, because MEASURED (2026-08-04) the window is always empty: across
    Poland, Germany, England, Malta, Andorra and Japan, no referee appears
    even twice in a team's last 15 matches. Over the full page the same
    check finds only 1-5 such referees per team out of 48-65 distinct ones
    -- rare enough to be worth remarking on when it happens, which is
    exactly why it needs a floor rather than being shown for every match.
    """
    frame = _cached_wikipedia_results(team)
    if frame.empty or not referee:
        return 0
    return int((frame["referee"] == referee).sum())


@st.cache_data(show_spinner=False)
def _cached_goal_minute_profile(
    team: str, as_of_iso: str, n_matches: int = 20
) -> pd.DataFrame:
    """Goals scored and conceded per 15-minute band over a team's last N matches.

    Built because Krok 0 measured `minute` at 99.5% coverage overall and
    100% across the last five years -- unlike the per-match stat columns,
    which sit at 14.8% and are handled conditionally instead.
    """
    records = _load_records_frame()
    goalscorers = _load_goalscorers_frame()
    as_of = pd.Timestamp(as_of_iso)
    recent = streaks.team_matches(records, team, as_of).tail(n_matches)
    bands = ["1-15", "16-30", "31-45", "46-60", "61-75", "76-90+"]
    empty = pd.DataFrame({"band": bands, "scored": [0] * 6, "conceded": [0] * 6})
    if recent.empty or goalscorers.empty or "minute" not in goalscorers.columns:
        return empty

    canonical = canonicalize_team_name(team)
    goals = goalscorers.loc[
        goalscorers["match_id"].isin(set(recent["match_id"]))
    ].copy()
    goals = goals.loc[goals["minute"].notna()]
    if goals.empty:
        return empty

    # An own goal is credited to the scorer's own team in this source, so the
    # side that actually benefits is the opponent -- flip it before counting,
    # otherwise a team's "scored" profile quietly includes its own mistakes.
    scoring_team = goals.apply(
        lambda row: (
            (
                row["away_team_name"]
                if row["team_name"] == row["home_team_name"]
                else row["home_team_name"]
            )
            if bool(row.get("own_goal"))
            else row["team_name"]
        ),
        axis=1,
    )
    band_index = (
        (
            (pd.to_numeric(goals["minute"], errors="coerce").fillna(0).clip(1, 90) - 1)
            // 15
        )
        .clip(0, 5)
        .astype(int)
    )
    scored_counts = band_index.loc[scoring_team == canonical].value_counts()
    conceded_counts = band_index.loc[scoring_team != canonical].value_counts()
    return pd.DataFrame(
        {
            "band": bands,
            "scored": [int(scored_counts.get(index, 0)) for index in range(6)],
            "conceded": [int(conceded_counts.get(index, 0)) for index in range(6)],
        }
    )


@st.cache_data(show_spinner=False)
def _cached_current_streaks(team: str, as_of_iso: str) -> list[str]:
    """Every live streak for a team, already rendered to Polish sentences."""
    facts = streaks.all_current_streaks(
        _load_records_frame(), team, pd.Timestamp(as_of_iso)
    )
    return [render_fact(fact) for fact in facts]


@st.cache_data(show_spinner=False)
def _cached_facts(home: str, away: str, as_of_iso: str, top_n: int) -> list[Fact]:
    """Facts for a match, keyed on values -- the frames are cached separately."""
    return facts_for_match(
        _load_records_frame(),
        home,
        away,
        as_of_date=pd.Timestamp(as_of_iso),
        top_n=top_n,
        goalscorers_frame=_load_goalscorers_frame(),
    )


# --------------------------------------------------------------------------
# Shared rendering helpers
# --------------------------------------------------------------------------


@st.cache_resource
def _watermark_image(path_text: str) -> Image.Image | None:
    """Load a watermark PNG once per process -- Plotly takes a PIL image directly."""
    path = Path(path_text)
    if not path.exists():
        return None
    return Image.open(path)


def _add_watermark(figure: go.Figure, on_light_fill: bool = False) -> go.Figure:
    """Anchor the StatsDesk watermark in a figure's bottom-right corner.

    `on_light_fill` picks the dark-ink variant, for figures whose own marks
    are pale -- a heat map's bottom-right corner is a cream cell, not the
    page background, and the light-ink watermark meant for the dark theme
    all but disappears on it.

    Opacity is already baked into the PNG (the source SVG sets it), so no
    extra `opacity` here -- doubling it would wash the mark out completely.
    """
    source = _WATERMARK_LIGHT if on_light_fill else _WATERMARK_DARK
    watermark = _watermark_image(str(source))
    if watermark is None:
        return figure
    figure.add_layout_image(
        source=watermark,
        xref="paper",
        yref="paper",
        x=1,
        y=0,
        sizex=0.12,
        sizey=0.12,
        xanchor="right",
        yanchor="bottom",
        layer="above",
    )
    return figure


def _style_figure(
    figure: go.Figure, height: int, watermark_on_light_fill: bool = False
) -> go.Figure:
    """Strip Plotly's default chrome, match the app theme, and stamp the watermark.

    The watermark is applied here rather than at each `st.plotly_chart` call
    because every figure in the app already funnels through this function --
    so a chart added later cannot silently ship unbranded.
    """
    figure.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Georgia, serif", color=_TEXT_COLOUR, size=14),
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False),
        yaxis=dict(showgrid=False, zeroline=False),
    )
    return _add_watermark(figure, on_light_fill=watermark_on_light_fill)


def _outcome_chart(
    home_label: str, away_label: str, probabilities: dict[str, float]
) -> go.Figure:
    """1X2 as three semantically distinct bars, percentage always drawn as text."""
    labels = [f"Wygrana: {away_label}", "Remis", f"Wygrana: {home_label}"]
    values = [
        probabilities["away_win"],
        probabilities["draw"],
        probabilities["home_win"],
    ]
    figure = go.Figure(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker_color=[_AWAY_COLOUR, _DRAW_COLOUR, _HOME_COLOUR],
            text=[f"{value:.0%}" for value in values],
            # Outside the bar, in the theme's own text colour: contrast is
            # then against the page background (13.9:1), not against whichever
            # bar colour happens to sit behind it.
            textposition="outside",
            textfont=dict(color=_TEXT_COLOUR, size=15),
            cliponaxis=False,
        )
    )
    figure.update_xaxes(range=[0, 1.15], showticklabels=False)
    return _style_figure(figure, height=210)


def _team_comparison_chart(
    home_label: str, away_label: str, rows: list[tuple[str, float, float]]
) -> go.Figure:
    """Match stats as paired home-vs-away bars -- easier to scan than a table."""
    categories = [label for label, _, _ in rows]
    home_values = [home for _, home, _ in rows]
    away_values = [away for _, _, away in rows]
    figure = go.Figure()
    figure.add_bar(
        x=home_values,
        y=categories,
        orientation="h",
        name=home_label,
        marker_color=_HOME_COLOUR,
        text=[f"{value:.1f}" for value in home_values],
        textposition="outside",
        textfont=dict(color=_TEXT_COLOUR),
        cliponaxis=False,
    )
    figure.add_bar(
        x=away_values,
        y=categories,
        orientation="h",
        name=away_label,
        marker_color=_AWAY_COLOUR,
        text=[f"{value:.1f}" for value in away_values],
        textposition="outside",
        textfont=dict(color=_TEXT_COLOUR),
        cliponaxis=False,
    )
    figure.update_layout(
        barmode="group", showlegend=True, legend=dict(orientation="h", y=1.12, x=0)
    )
    figure.update_xaxes(
        showticklabels=False, range=[0, max([*home_values, *away_values]) * 1.25]
    )
    return _style_figure(figure, height=60 + 52 * len(rows))


def _probability_chart(labels: list[str], values: list[float]) -> go.Figure:
    """Generic ranked probability chart (qualification odds)."""
    figure = go.Figure(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker_color=_HOME_COLOUR,
            text=[f"{value:.0%}" for value in values],
            textposition="outside",
            textfont=dict(color=_TEXT_COLOUR),
            cliponaxis=False,
        )
    )
    figure.update_xaxes(range=[0, 1.15], showticklabels=False)
    return _style_figure(figure, height=90 + 46 * len(labels))


def _importance_meter(weight: float, maximum_weight: float) -> str:
    """Five blocks -- a fact's weight relative to the strongest fact on screen."""
    if maximum_weight <= 0:
        filled = 0
    else:
        filled = max(1, round(5 * weight / maximum_weight))
    return "▰" * filled + "▱" * (5 - filled)


def _format_provenance_row(record: dict[str, object]) -> str:
    """One source match, as a line a journalist can check against any archive."""
    match_date = pd.Timestamp(record["match_date"]).date()
    home = nominative_pl(str(record["home_team_name"]))
    away = nominative_pl(str(record["away_team_name"]))
    home_score = record.get("home_score")
    away_score = record.get("away_score")
    if pd.isna(home_score) or pd.isna(away_score):
        score = "brak wyniku"
    else:
        score = f"{int(home_score)}:{int(away_score)}"
    tournament = str(record.get("tournament") or "").strip()
    suffix = f" — {tournament}" if tournament else ""
    return f"{match_date} · {home} {score} {away}{suffix}"


def _format_date_header_pl(day: date) -> str:
    """Format a date as "Czwartek, 24 września" -- weekday plus genitive month.

    Not `locale`/`babel`: neither is a dependency here and `setlocale` is
    process-global, so a Streamlit worker changing it would leak into every
    other thread's formatting.
    """
    month = _MONTHS_GENITIVE_PL[day.month - 1]
    return f"{_WEEKDAYS_PL[day.weekday()]}, {day.day} {month}"


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two points, in kilometres.

    For the Travel Fatigue Index (see PROMPT_GEOCODING.md) -- feeds
    `_render_travel_fatigue` below. Sanity-checked against well-known
    reference distances before use (Warsaw-London ~1450 km, New York-Los
    Angeles ~3936 km, both within 1 km of this formula's output) -- that
    verifies the *formula*, not any specific coordinate this app will look
    up, which is a separate, sourced-data concern handled entirely in the
    geocoding module.
    """
    earth_radius_km = 6371.0088
    lat1_radians, lat2_radians = math.radians(lat1), math.radians(lat2)
    delta_lat = math.radians(lat2 - lat1)
    delta_lon = math.radians(lon2 - lon1)
    haversine_term = (
        math.sin(delta_lat / 2) ** 2
        + math.cos(lat1_radians) * math.cos(lat2_radians) * math.sin(delta_lon / 2) ** 2
    )
    return 2 * earth_radius_km * math.asin(math.sqrt(haversine_term))


def _timezone_offset_change_hours(
    origin_tz: str, destination_tz: str, as_of: pd.Timestamp
) -> float:
    """Real UTC-offset delta between two IANA zones on a given date.

    Computed at `as_of` specifically, not from a fixed offset table -- the
    real gap between two zones can change across the year wherever one
    observes DST and the other doesn't (or they switch on different dates),
    so the correct answer depends on which date is actually being asked
    about.
    """
    moment = as_of.tz_localize(None).to_pydatetime()
    origin_offset = ZoneInfo(origin_tz).utcoffset(moment)
    destination_offset = ZoneInfo(destination_tz).utcoffset(moment)
    if origin_offset is None or destination_offset is None:
        return 0.0
    return (destination_offset - origin_offset).total_seconds() / 3600.0


def _render_travel_fatigue(
    home: str, away: str, match_date: date, venue: str | None
) -> None:
    """Distance/rest/timezone context ahead of kickoff -- never a model feature.

    Explicitly NOT merged into `features/builder.py`/the training frame
    (see `PROMPT_ZAMKNIECIE_WATKOW.md`'s "Twarde zasady, które zostają") --
    unlike the Club Cohesion Index, this stays a log/insight-layer read, not
    a verified engineered feature: it's a per-fixture geocoding lookup with
    no historical backfill, so a model could never see it consistently
    across training rows anyway. Shows nothing it can't back with real
    geocoded data -- no venue string for this fixture, no resolvable
    coordinates for the venue, no prior match for a team, or an
    unresolvable city/country on that prior match, each just drop that team
    (or the whole section) rather than guessing a distance.
    """
    if not venue:
        return
    venue_coordinates = coordinates_for_stadium(venue)
    if venue_coordinates is None:
        return
    venue_lat, venue_lon, venue_tz = venue_coordinates

    as_of = pd.Timestamp(match_date, tz="UTC")
    rows: list[dict[str, object]] = []
    for team in (home, away):
        form = _cached_recent_form(team, as_of.isoformat())
        if form.empty:
            continue
        last_match = form.iloc[-1]
        last_city = last_match.get("city")
        last_country = last_match.get("country")
        if pd.isna(last_city) or pd.isna(last_country):
            continue
        origin = coordinates_for_city_country(str(last_city), str(last_country))
        if origin is None:
            continue
        origin_lat, origin_lon, origin_tz = origin
        row: dict[str, object] = {
            "team": nominative_pl(team),
            "distance_km": _haversine_km(origin_lat, origin_lon, venue_lat, venue_lon),
            "rest_days": _current_rest_days(form, as_of),
        }
        if origin_tz != venue_tz:
            row["timezone_change_hours"] = _timezone_offset_change_hours(
                origin_tz, venue_tz, as_of
            )
        rows.append(row)

    if not rows:
        return

    with st.container(border=True):
        st.subheader("Zmęczenie podróżą")
        st.caption(
            "Kontekst przedmeczowy, nie cecha modelu — dystans od miejsca "
            "poprzedniego meczu do miejsca tego meczu, liczony po linii "
            "prostej (geodezyjnie) na podstawie realnych, geokodowanych "
            "lokalizacji."
        )
        columns = st.columns(len(rows))
        for column, row in zip(columns, rows, strict=True):
            with column:
                st.metric(str(row["team"]), f"{row['distance_km']:.0f} km")
                details = []
                if row["rest_days"] is not None:
                    details.append(f"odpoczynek: {row['rest_days']:.0f} dni")
                if "timezone_change_hours" in row:
                    details.append(
                        f"zmiana strefy: {row['timezone_change_hours']:+.0f} h"
                    )
                if details:
                    st.caption(" · ".join(details))


def _relative_luminance(hex_color: str) -> float:
    """WCAG relative luminance of a `#rrggbb` colour, for contrast-ratio math."""

    def channel(value: int) -> float:
        fraction = value / 255
        if fraction <= 0.04045:
            return fraction / 12.92
        return ((fraction + 0.055) / 1.055) ** 2.4

    red, green, blue = (int(hex_color[i : i + 2], 16) for i in (1, 3, 5))
    return 0.2126 * channel(red) + 0.7152 * channel(green) + 0.0722 * channel(blue)


def _readable_cell_text_color(background_hex: str) -> str:
    """Pure black or white, whichever clears WCAG contrast against `background_hex`.

    `0.1791` is the luminance where contrast-to-black equals contrast-to-
    white (solving `(L + 0.05) / 0.05 == 1.05 / (L + 0.05)`), so this always
    picks whichever of the two is actually better for a given cell -- not a
    fixed light-half/dark-half split, which is what let a branded-colour
    version fall short of 4.5:1 partway along its own ramp (see
    `_METRIC_HEATMAP_COLORSCALE`'s comment).
    """
    return "#000000" if _relative_luminance(background_hex) > 0.1791 else "#ffffff"


def _lerp_hex_color(start_hex: str, end_hex: str, t: float) -> str:
    """Linear interpolation between two `#rrggbb` colours."""
    start = [int(start_hex[i : i + 2], 16) for i in (1, 3, 5)]
    end = [int(end_hex[i : i + 2], 16) for i in (1, 3, 5)]
    mixed = [round(s + (e - s) * t) for s, e in zip(start, end, strict=True)]
    return "#" + "".join(f"{value:02x}" for value in mixed)


def _metric_heatmap_cell_color(t: float) -> str:
    """The exact hex colour `_METRIC_HEATMAP_COLORSCALE` renders at position `t` (0..1).

    Plotly computes this internally for the chart itself, but a cell's text
    colour has to be picked in Python before the figure is built.
    """
    stops = _METRIC_HEATMAP_COLORSCALE
    clamped = min(max(t, 0.0), 1.0)
    for (start_t, start_color), (end_t, end_color) in zip(
        stops, stops[1:], strict=False
    ):
        if clamped <= end_t:
            span = end_t - start_t
            local_t = (clamped - start_t) / span if span > 0 else 0.0
            return _lerp_hex_color(start_color, end_color, local_t)
    return stops[-1][1]


def _heatmap_figure(
    matrix: list[list[float]],
    x_labels: list[str],
    y_labels: list[str],
    cell_texts: list[list[str]],
    height: int,
) -> go.Figure:
    """A single-hue heat map with every value also written into its cell.

    One colour family from light to dark rather than a rainbow scale: a
    rainbow implies unordered categories and is unreadable with common
    colour-vision deficiencies. The number is always drawn as text too, so
    colour never carries information on its own, and the text flips to a
    light shade above the midpoint so it stays legible on dark cells.
    """
    peak = max((value for row in matrix for value in row), default=0.0)
    annotations = []
    for row_index, row in enumerate(matrix):
        for column_index, value in enumerate(row):
            is_dark_cell = peak > 0 and value / peak > 0.55
            annotations.append(
                dict(
                    x=x_labels[column_index],
                    y=y_labels[row_index],
                    text=cell_texts[row_index][column_index],
                    showarrow=False,
                    font=dict(
                        color="#f2ede0" if is_dark_cell else "#20242c",
                        size=12,
                        family="Georgia, serif",
                    ),
                )
            )
    figure = go.Figure(
        go.Heatmap(
            z=matrix,
            x=x_labels,
            y=y_labels,
            colorscale=[[0.0, "#f0ead9"], [0.5, "#c9a227"], [1.0, "#5c4a12"]],
            showscale=False,
            hoverinfo="skip",
            xgap=2,
            ygap=2,
        )
    )
    figure.update_layout(annotations=annotations)
    figure.update_yaxes(autorange="reversed")
    return _style_figure(figure, height=height, watermark_on_light_fill=True)


def _score_grid_matrix(
    home_goals: float, away_goals: float, max_goals: int = 5
) -> tuple[list[list[float]], list[tuple[str, float]], float]:
    """The Poisson exact-scoreline matrix, shared by the on-screen chart and the PDF's.

    Split out from `_score_grid_figure` so the PDF's matplotlib version
    (which needs the raw probabilities, not a Plotly figure) computes from
    the exact same numbers the screen does, rather than a second
    independent implementation that could drift out of sync.
    """
    home_lambda = max(float(home_goals), 1e-6)
    away_lambda = max(float(away_goals), 1e-6)
    home_pmf = [_poisson_pmf(home_lambda, count) for count in range(max_goals + 1)]
    away_pmf = [_poisson_pmf(away_lambda, count) for count in range(max_goals + 1)]
    matrix = [
        [
            home_pmf[home_count] * away_pmf[away_count]
            for away_count in range(max_goals + 1)
        ]
        for home_count in range(max_goals + 1)
    ]
    inside_mass = sum(value for row in matrix for value in row)

    scorelines = [
        (f"{home_count}:{away_count}", matrix[home_count][away_count])
        for home_count in range(max_goals + 1)
        for away_count in range(max_goals + 1)
    ]
    top_scorelines = sorted(scorelines, key=lambda item: -item[1])[:3]
    return matrix, top_scorelines, 1.0 - inside_mass


def _score_grid_figure(
    home_label: str,
    away_label: str,
    home_goals: float,
    away_goals: float,
    max_goals: int = 5,
) -> tuple[go.Figure, list[tuple[str, float]], float]:
    """Exact-scoreline probabilities as a grid, from independent Poisson sides.

    The same distributional assumption `simulation/monte_carlo.py` already
    simulates from, just evaluated analytically instead of sampled. Replaces
    a single "predicted score", which reads as a confident call the model
    never actually makes -- the grid shows how flat the real distribution is.

    Also returns the three likeliest scorelines and the probability mass
    outside the grid, so the caption can state honestly how much isn't shown.
    """
    matrix, top_scorelines, tail_mass = _score_grid_matrix(
        home_goals, away_goals, max_goals
    )
    figure = _heatmap_figure(
        matrix=matrix,
        x_labels=[str(count) for count in range(max_goals + 1)],
        y_labels=[str(count) for count in range(max_goals + 1)],
        cell_texts=[
            [f"{value:.0%}" if value >= 0.005 else "" for value in row]
            for row in matrix
        ],
        height=420,
    )
    figure.update_layout(
        xaxis_title=f"Bramki: {away_label}",
        yaxis_title=f"Bramki: {home_label}",
        xaxis=dict(showgrid=False, zeroline=False, side="top"),
    )
    return figure, top_scorelines, tail_mass


def _poisson_pmf(expected: float, count: int) -> float:
    """P(exactly `count`) for Poisson(`expected`), without pulling in scipy."""
    return math.exp(-expected) * expected**count / math.factorial(count)


def _plural_pl(count: int, one: str, few: str, many: str) -> str:
    """Polish numeral agreement: 1 mecz, 2 mecze, 5 meczów.

    Polish takes three forms, not two, and the "few" form is skipped for the
    teens -- 22 mecze but 12 meczów. Written out here because the app builds
    counted phrases ("3 spotkania", "2 zwycięstwa") that read as broken
    Polish with naive English-style pluralisation.
    """
    if count == 1:
        return one
    if count % 10 in (2, 3, 4) and count % 100 not in (12, 13, 14):
        return few
    return many


def _result_letter(goals_for: float, goals_against: float) -> str:
    """W/R/P from one team's own goal columns -- same letters the group table uses."""
    if goals_for > goals_against:
        return "W"
    if goals_for == goals_against:
        return "R"
    return "P"


def _team_metric_value(row: pd.Series, base: str) -> float | None:
    """This team's own value for one stat column, picking home_*/away_* by `is_home`.

    `base="cards"` is synthetic -- yellow and red summed -- everything else
    maps straight onto a `home_<base>`/`away_<base>` records_frame column.
    """
    prefix = "home_" if bool(row["is_home"]) else "away_"
    if base == "cards":
        yellow = row.get(f"{prefix}yellow_cards")
        red = row.get(f"{prefix}red_cards")
        if pd.isna(yellow) and pd.isna(red):
            return None
        yellow_count = 0.0 if pd.isna(yellow) else float(yellow)
        red_count = 0.0 if pd.isna(red) else float(red)
        return yellow_count + red_count
    value = row.get(f"{prefix}{base}")
    return None if pd.isna(value) else float(value)


def _row_has_majority_coverage(form: pd.DataFrame, base: str) -> bool:
    """True when more than half of the rendered matches have real data for `base`.

    Checked against the actual window being shown (however many matches that
    is, not a fixed ten) -- a team with 4 matches on record and 3 with data
    still shows the row. Krok 0 found this stat family moves together as one
    block from one source and is frequently absent entirely for a given
    team/era, never worth half-showing.
    """
    present = sum(
        _team_metric_value(row, base) is not None for _, row in form.iterrows()
    )
    return present * 2 > len(form)


def _normalize_row(values: list[float | None]) -> list[float | None]:
    """Scale one row's values into 0..1 for a shared colorscale; `None` stays `None`.

    Normalized *per row*, deliberately not globally: intensity means "high
    relative to this team's own matches shown for this metric", not "high
    compared to some other metric or another team" -- those aren't
    comparable scales.
    """
    present = [value for value in values if value is not None]
    if not present:
        return [None for _ in values]
    low, high = min(present), max(present)
    if high == low:
        return [None if value is None else 0.5 for value in values]
    return [None if value is None else (value - low) / (high - low) for value in values]


def _team_metric_heatmap_rows(
    form: pd.DataFrame,
) -> list[tuple[str, list[float | None], str]]:
    """Label, per-match values (oldest first) and cell-text format, per covered row.

    "Gole strzelone"/"Gole stracone" always appear -- goals are core to
    every played match. The rest appear only with real per-team coverage in
    this window (see `_row_has_majority_coverage`).
    """
    def _numeric_column(column: str) -> list[float | None]:
        return [None if pd.isna(value) else float(value) for value in form[column]]

    rows: list[tuple[str, list[float | None], str]] = [
        ("Gole strzelone", _numeric_column("goals_for"), "{:.0f}"),
        ("Gole stracone", _numeric_column("goals_against"), "{:.0f}"),
    ]
    for label, base, cell_format in _TEAM_METRIC_ROW_SPECS:
        if not _row_has_majority_coverage(form, base):
            continue
        values = [_team_metric_value(row, base) for _, row in form.iterrows()]
        rows.append((label, values, cell_format))
    return rows


def _team_metric_heatmap_figure(form: pd.DataFrame) -> go.Figure:
    """The combined Wynik + metrics grid: columns are matches, rows are metrics.

    Drawn as explicit per-cell rectangle shapes, not a Heatmap trace's
    z/colorscale. CONFIRMED REAL (2026-08-05, live browser): two Heatmap
    traces sharing one categorical axis -- meant to combine "Wynik"'s fixed
    three-hue scale with the metric rows' continuous, per-row-normalized
    one -- rendered as a blank chart, not the merged grid a shared axis was
    assumed to produce. Shapes sidestep that: every cell's colour is
    computed in Python and drawn directly, nothing left for Plotly to merge.
    Columns use explicit numeric positions with custom tick text rather
    than date strings as `x` -- also confirmed real, separately, on the
    attendance chart below: Plotly auto-detects axis type from the data, and
    "26.03"-shaped strings get silently parsed as the *number* 26.03,
    scrambling both position and chronological order.
    """
    dates = [pd.Timestamp(value).date() for value in form["match_date"]]
    x_labels = [single_date.strftime("%d.%m") for single_date in dates]
    letters = [
        _result_letter(row["goals_for"], row["goals_against"])
        for _, row in form.iterrows()
    ]
    metric_rows = _team_metric_heatmap_rows(form)
    row_labels = ["Wynik"] + [label for label, _, _ in metric_rows]
    n_columns = len(x_labels)
    n_rows = len(row_labels)

    shapes: list[dict[str, object]] = []
    annotations: list[dict[str, object]] = []
    inset = 0.07  # a thin gap between cells, echoing the old xgap/ygap=2

    def add_cell(
        row_index: int,
        column_index: int,
        fill_color: str,
        text: str,
        text_color: str,
    ) -> None:
        # Always a filled rect, missing cells included: a "brak" label with
        # no cell boundary at all crowds visually into its neighbour,
        # especially with two missing cells side by side -- confirmed real,
        # 2026-08-05 -- and doesn't match the brief's own "puste/wyszarzone"
        # (empty/*greyed*) wording, which implies a visible muted fill, not
        # literally nothing.
        shapes.append(
            dict(
                type="rect",
                xref="x",
                yref="y",
                x0=column_index - 0.5 + inset,
                x1=column_index + 0.5 - inset,
                y0=row_index - 0.5 + inset,
                y1=row_index + 0.5 - inset,
                fillcolor=fill_color,
                line=dict(width=0),
            )
        )
        # "brak" (4 letters) is wider than almost every real value a cell
        # holds ("0", "73%") and visibly overflowed its column at size 12 --
        # confirmed real, 2026-08-05, live browser -- so longer labels drop
        # a size to fit ~10 columns' worth of cell width.
        font_size = 10 if len(text) > 3 else 12
        annotations.append(
            dict(
                x=column_index,
                y=row_index,
                text=text,
                showarrow=False,
                font=dict(color=text_color, size=font_size, family="Georgia, serif"),
            )
        )

    for column_index, letter in enumerate(letters):
        background, foreground = _RESULT_MARKER_STYLES[letter]
        add_cell(0, column_index, background, letter, foreground)

    for row_index, (_, values, cell_format) in enumerate(metric_rows, start=1):
        normalized = _normalize_row(values)
        paired = enumerate(zip(values, normalized, strict=True))
        for column_index, (raw_value, norm_value) in paired:
            if raw_value is None:
                add_cell(
                    row_index,
                    column_index,
                    _GRID_COLOUR,
                    _MISSING_CELL_TEXT,
                    _MUTED_TEXT_COLOUR,
                )
            else:
                text = cell_format.format(raw_value)
                cell_color = _metric_heatmap_cell_color(norm_value)
                text_color = _readable_cell_text_color(cell_color)
                add_cell(row_index, column_index, cell_color, text, text_color)

    figure = go.Figure()
    # An invisible anchor trace: shapes alone don't establish axis ranges,
    # so without this the plot area can size itself arbitrarily.
    figure.add_trace(
        go.Scatter(
            x=[-0.5, n_columns - 0.5],
            y=[-0.5, n_rows - 0.5],
            mode="markers",
            marker=dict(opacity=0),
            showlegend=False,
            hoverinfo="skip",
        )
    )
    figure.update_layout(shapes=shapes, annotations=annotations)
    height = 58 + 42 * n_rows
    figure = _style_figure(figure, height=height, watermark_on_light_fill=True)
    figure.update_xaxes(
        tickmode="array",
        tickvals=list(range(n_columns)),
        ticktext=x_labels,
        range=[-0.5, n_columns - 0.5],
    )
    figure.update_yaxes(
        tickmode="array",
        tickvals=list(range(n_rows)),
        ticktext=row_labels,
        range=[n_rows - 0.5, -0.5],
    )
    return figure


def _render_attendance_chart(team: str, form: pd.DataFrame) -> None:
    """One bar per match with attendance data; a gap, not a zero, where it's missing.

    A true zero (COVID, behind closed doors) and a missing figure are both
    "no visible bar" in an ordinary bar chart -- indistinguishable, which is
    exactly the ambiguity this round's brief calls out. Zero gets a small
    floor height plus its real "0" label so it stays visibly present; a
    missing match gets neither -- true absence from the plot, not a data
    point silently rendered as zero.
    """
    dates = [pd.Timestamp(value).date() for value in form["match_date"]]
    x_labels = [single_date.strftime("%d.%m") for single_date in dates]
    attendances: list[float | None] = []
    for (_, row), match_date in zip(form.iterrows(), dates, strict=True):
        detail = _wikipedia_match_detail(
            team, match_date, row["goals_for"], row["goals_against"]
        )
        value = detail.get("attendance") if detail else None
        attendances.append(None if value is None or pd.isna(value) else float(value))

    if not any(value is not None for value in attendances):
        return  # nothing from Wikipedia in this window -- silence, not ten empty gaps

    present = [value for value in attendances if value is not None]
    floor = max(present) * 0.03 if max(present) > 0 else 1.0
    # Numeric column index, not the "DD.MM" string, is what goes into `x` --
    # CONFIRMED REAL (2026-08-05, live browser): Plotly auto-detects axis
    # type from the data, and date-shaped strings like "26.03" get silently
    # parsed as the *number* 26.03, scrambling both bar position and
    # chronological order (ticks rendered as "5, 10, ... 30" instead of
    # dates). Custom tick text below relabels the numeric axis instead.
    indexed = list(enumerate(attendances))
    bar_x = [index for index, value in indexed if value is not None]
    bar_y = [floor if value == 0 else value for value in present]
    bar_text = [f"{int(value):,}".replace(",", " ") for value in present]

    figure = go.Figure(
        go.Bar(
            x=bar_x,
            y=bar_y,
            text=bar_text,
            textposition="outside",
            marker=dict(color="#c9a227"),
            hoverinfo="skip",
        )
    )
    figure = _style_figure(figure, height=110)
    figure.update_xaxes(
        tickmode="array",
        tickvals=list(range(len(x_labels))),
        ticktext=x_labels,
        range=[-0.5, len(x_labels) - 0.5],
        showgrid=False,
        zeroline=False,
    )
    figure.update_yaxes(showticklabels=False, showgrid=False, zeroline=False)
    st.plotly_chart(figure, width="stretch")
    st.caption(
        "Frekwencja ostatnich meczów. Przerwa w słupkach = brak danych "
        "źródłowych; słupek przy zerze = mecz naprawdę rozegrany bez publiczności."
    )


def _head_to_head_summary(home: str, away: str, as_of: pd.Timestamp) -> str:
    """One-sentence head-to-head record, shared by the screen and the PDF."""
    meetings = _cached_head_to_head(home, away, as_of.isoformat())
    home_label = nominative_pl(home)
    away_label = nominative_pl(away)
    if meetings.empty:
        return f"{home_label} i {away_label} nigdy dotąd ze sobą nie grały."
    counts = meetings["team_a_result"].value_counts()
    wins = int(counts.get("win", 0))
    draws = int(counts.get("draw", 0))
    losses = int(counts.get("loss", 0))
    return (
        f"{len(meetings)} "
        f"{_plural_pl(len(meetings), 'spotkanie', 'spotkania', 'spotkań')} · "
        f"z perspektywy {home_label}: "
        f"{wins} {_plural_pl(wins, 'zwycięstwo', 'zwycięstwa', 'zwycięstw')}, "
        f"{draws} {_plural_pl(draws, 'remis', 'remisy', 'remisów')}, "
        f"{losses} {_plural_pl(losses, 'porażka', 'porażki', 'porażek')}."
    )


def _render_goal_minute_profile(home: str, away: str, as_of: pd.Timestamp) -> None:
    """When each side scores and concedes, in 15-minute bands.

    Built on `goalscorers.minute`, measured at 99.5% coverage overall and
    100% over the last five years -- the one rich column in this dataset.
    """
    rows: list[list[float]] = []
    labels: list[str] = []
    texts: list[list[str]] = []
    for team in (home, away):
        profile = _cached_goal_minute_profile(team, as_of.isoformat())
        if profile[["scored", "conceded"]].to_numpy().sum() == 0:
            continue
        team_label = nominative_pl(team)
        for column, suffix in (("scored", "strzelone"), ("conceded", "stracone")):
            rows.append([float(value) for value in profile[column]])
            labels.append(f"{team_label} — {suffix}")
            texts.append([str(int(value)) for value in profile[column]])
    if not rows:
        st.caption("Brak danych o minutach goli dla tych drużyn.")
        return
    st.plotly_chart(
        _heatmap_figure(
            matrix=rows,
            x_labels=list(_cached_goal_minute_profile(home, as_of.isoformat())["band"]),
            y_labels=labels,
            cell_texts=texts,
            height=90 + 46 * len(rows),
        ),
        width="stretch",
    )
    st.caption(
        "Gole z ostatnich 20 meczów każdej drużyny, w przedziałach 15-minutowych."
    )


def _render_head_to_head(home: str, away: str, as_of: pd.Timestamp) -> None:
    """W/D/L balance from the home team's side, plus the last five meetings."""
    meetings = _cached_head_to_head(home, away, as_of.isoformat())
    home_label = nominative_pl(home)
    away_label = nominative_pl(away)
    st.markdown("**Bilans bezpośredni**")
    if meetings.empty:
        st.caption(
            f"{home_label} i {away_label} nigdy dotąd ze sobą nie grały — to byłoby "
            "ich pierwsze spotkanie."
        )
        return

    counts = meetings["team_a_result"].value_counts()
    wins = int(counts.get("win", 0))
    draws = int(counts.get("draw", 0))
    losses = int(counts.get("loss", 0))
    st.caption(
        f"{len(meetings)} "
        f"{_plural_pl(len(meetings), 'spotkanie', 'spotkania', 'spotkań')} · "
        f"z perspektywy {home_label}: "
        f"{wins} {_plural_pl(wins, 'zwycięstwo', 'zwycięstwa', 'zwycięstw')}, "
        f"{draws} {_plural_pl(draws, 'remis', 'remisy', 'remisów')}, "
        f"{losses} {_plural_pl(losses, 'porażka', 'porażki', 'porażek')}."
    )
    recent = meetings.tail(5).iloc[::-1]
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Data": str(pd.Timestamp(row["match_date"]).date()),
                    "Wynik": (
                        f"{nominative_pl(str(row['home_team_name']))} "
                        f"{int(row['home_score'])}:{int(row['away_score'])} "
                        f"{nominative_pl(str(row['away_team_name']))}"
                    ),
                    "Rozgrywki": str(row.get("tournament") or ""),
                }
                for _, row in recent.iterrows()
            ]
        ),
        width="stretch",
        hide_index=True,
        # Without an explicit width the long "Wynik" string squeezes the date
        # column until it renders as "2024-0".
        column_config={"Data": st.column_config.TextColumn(width="small")},
    )


def _current_win_loss_streak(form: pd.DataFrame) -> tuple[int, int]:
    """Current win/loss streak entering the next match.

    Same definition `features.builder._win_loss_streaks_for_team` trains the
    model's own `win_streak`/`loss_streak` features on -- consecutive
    same-sign goal differences, reset by a draw -- computed independently
    here (not imported from `features/`) rather than reused, because this
    file's own architecture is presentation-only and never reaches into the
    engine (see module docstring). `form` already only holds matches
    strictly before the target date (`insights.streaks.team_matches`'s own
    contract), so summing over all of it is exactly the shifted, leak-safe
    value a training row on that date would have seen -- not a different
    metric borrowing the same name.
    """
    win_streak = 0
    loss_streak = 0
    for _, row in form.iterrows():
        difference = row["goals_for"] - row["goals_against"]
        if difference > 0:
            win_streak += 1
            loss_streak = 0
        elif difference < 0:
            loss_streak += 1
            win_streak = 0
        else:
            win_streak = 0
            loss_streak = 0
    return win_streak, loss_streak


def _current_rest_days(form: pd.DataFrame, as_of: pd.Timestamp) -> float | None:
    """Days since the team's last known match -- same definition as `_add_recency_features`.

    `None` (not the engine's 14.0 imputation default) when there's no prior
    match at all -- that fallback exists to keep a training row numeric for
    a model that can't accept NaN, not to be asserted here as a real fact to
    a reader who has no such constraint.
    """
    if form.empty:
        return None
    last_match_date = pd.Timestamp(form.iloc[-1]["match_date"])
    return (as_of - last_match_date).total_seconds() / 86400.0


def _render_model_signals(team: str, as_of: pd.Timestamp, form: pd.DataFrame) -> None:
    """Real, currently-used model inputs for one team -- Elo, rest, current streak.

    Deliberately the opposite case from `_render_match_context` (live
    context the model provably does NOT use, e.g. lineup value/injuries):
    every number shown here genuinely feeds the trained model, so it
    carries no "does not enter the model" disclaimer -- adding one here
    would be just as dishonest as omitting one there. Silently drops any
    single part it can't compute (no Elo for this team, no prior match for
    rest days, no active streak) instead of showing a placeholder.
    """
    parts: list[str] = []
    elo = _load_elo_ratings().get(canonicalize_team_name(team))
    if elo is not None:
        parts.append(f"Elo: {elo:.0f}")
    rest_days = _current_rest_days(form, as_of)
    if rest_days is not None:
        parts.append(f"odpoczynek: {rest_days:.0f} dni")
    win_streak, loss_streak = _current_win_loss_streak(form)
    if win_streak > 0:
        word = _plural_pl(win_streak, "zwycięstwo", "zwycięstwa", "zwycięstw")
        parts.append(f"seria: {win_streak} {word}")
    elif loss_streak > 0:
        word = _plural_pl(loss_streak, "porażka", "porażki", "porażek")
        parts.append(f"seria: {loss_streak} {word}")
    if parts:
        st.caption(" · ".join(parts) + " — realne cechy modelu, nie dodatek")


def _render_team_form(team: str, as_of: pd.Timestamp) -> None:
    """Last ten matches as a metrics heat map and attendance chart.

    The full match-by-match table is one expander away, unchanged.
    """
    label = nominative_pl(team)
    form = _cached_recent_form(team, as_of.isoformat())
    st.markdown(f"**{label}**")
    if form.empty:
        st.caption("Brak meczów w zbiorze historycznym przed tą datą.")
        return

    _render_model_signals(team, as_of, form)
    st.plotly_chart(_team_metric_heatmap_figure(form), width="stretch")
    st.caption(
        "Intensywność koloru rośnie wraz z wartością, osobno w każdym "
        "wierszu — nie porównuj odcieni między wierszami ani między "
        "drużynami. Dla „gole stracone” i „kartki” wyższa wartość oznacza "
        "gorszy wynik drużyny."
    )
    # A team with fewer than ten matches shows however many it really has --
    # never padded to look like a full row.
    if len(form) < 10:
        matches_word = _plural_pl(len(form), "mecz", "mecze", "meczów")
        st.caption(f"Tylko {len(form)} {matches_word} w historii przed tą datą.")

    streak_sentences = _cached_current_streaks(team, as_of.isoformat())
    st.caption(
        " ".join(streak_sentences) if streak_sentences else "Brak aktualnej serii."
    )

    _render_attendance_chart(team, form)

    with st.expander(f"Pełna forma: {label}", expanded=False):
        # Columns appear only where the source actually has them: a team
        # whose page Wikipedia doesn't carry gets the original four columns,
        # not two empty ones implying we lost the data.
        st.dataframe(
            pd.DataFrame(_team_form_table_rows(team, form)),
            width="stretch",
            hide_index=True,
        )


def _team_form_table_rows(team: str, form: pd.DataFrame) -> list[dict[str, object]]:
    """One row per match, newest first -- shared by the on-screen expander and the PDF.

    "Frekwencja"/"Sędzia" are added per row only when Wikipedia actually has
    them for that exact match, so the two tables degrade identically: a
    team with no Wikipedia coverage at all just never gets those columns.
    """
    rows: list[dict[str, object]] = []
    for _, row in form.iloc[::-1].iterrows():
        entry: dict[str, object] = {
            "Data": pd.Timestamp(row["match_date"]).date(),
            "Przeciwnik": nominative_pl(str(row["opponent"])),
            "Wynik": f"{int(row['goals_for'])}:{int(row['goals_against'])}",
            "Gdzie": "dom" if bool(row["is_home"]) else "wyjazd",
        }
        detail = _wikipedia_match_detail(
            team,
            pd.Timestamp(row["match_date"]).date(),
            row["goals_for"],
            row["goals_against"],
        )
        if detail is not None:
            attendance = detail.get("attendance")
            if attendance is not None and not pd.isna(attendance):
                entry["Frekwencja"] = int(attendance)
            referee = detail.get("referee")
            if referee is not None and not pd.isna(referee):
                entry["Sędzia"] = str(referee)
        rows.append(entry)
    return rows


def _render_match_officials(home: str, away: str, match_date: date) -> None:
    """The match referee, when a source actually knows it -- otherwise nothing at all.

    MEASURED (2026-08-04): for a genuinely *upcoming* match no source here
    has the appointment. api-football carries no referee field on this plan,
    and Wikipedia's own not-yet-played blocks leave `referee` blank in 9 of
    9 cases on both Poland's and Germany's pages -- appointments land days
    before kickoff, after the article is written. So this line stays silent
    on previews and fills in for matches already played, which the app can
    equally be pointed at. If a schedule source ever starts publishing
    appointments, it lights up with no further change here.
    """
    referee = None
    for team in (home, away):
        detail = _wikipedia_details_by_date(team).get(match_date)
        if not detail:
            continue
        if detail.get("referee") is not None and not pd.isna(detail["referee"]):
            referee = str(detail["referee"])
            break
    if referee is None:
        return

    line = f"Sędzia: **{referee}**"
    # Phrased as a tally rather than a sentence on purpose: the team-name
    # table carries nominative/dative/accusative but no genitive, and
    # "prowadził mecze Polski" needs one. A label-and-count line reads
    # naturally with the nominative that does exist, instead of inflecting
    # by hand and getting it wrong for half the table.
    notes = []
    for team in (home, away):
        appearances = _referee_appearance_count(team, referee)
        if appearances >= _MINIMUM_REFEREE_APPEARANCES:
            matches_word = _plural_pl(appearances, "mecz", "mecze", "meczów")
            notes.append(f"{nominative_pl(team)} — {appearances} {matches_word}")
    if notes:
        line = f"{line} · prowadził już: {', '.join(notes)} (od 2020)"
    st.caption(line)


def _render_structural_context(home: str, away: str, as_of: pd.Timestamp) -> None:
    """Head-to-head and form -- always shown, never ranked against each other.

    Deliberately separate from the ranked `facts_for_match` list: `weight`
    compares incomparable scales (a fixed 0.3 for a match-count milestone vs.
    a computed 0.1 + 0.03/meeting for head-to-head), so a jubilee for one
    team could push the other team's unbeaten run out of the top ten and the
    whole list read as arbitrary. This block is the match context; the ranked
    list below the prediction is explicitly labelled as trivia.
    """
    with st.container(border=True):
        st.subheader("Bilans i forma")
        _render_head_to_head(home, away, as_of)
        st.divider()
        home_column, away_column = st.columns(2)
        with home_column:
            _render_team_form(home, as_of)
        with away_column:
            _render_team_form(away, as_of)
        st.divider()
        st.markdown("**Kiedy padają gole**")
        _render_goal_minute_profile(home, away, as_of)


def _render_facts(facts: list[Fact]) -> None:
    """Ranked facts, each expandable to the real matches it was derived from."""
    if not facts:
        st.info(
            "Brak faktów historycznych dla tej pary. To normalne, jeśli te "
            "drużyny nigdy ze sobą nie grały i żadna nie ma aktualnie serii "
            "ani jubileuszu — silnik nie zmyśla faktów, by zapełnić miejsce."
        )
        return

    lookup = _match_lookup()
    maximum_weight = max(fact.weight for fact in facts)
    for index, fact in enumerate(facts):
        type_label = _FACT_TYPE_LABELS_PL.get(fact.fact_type, fact.fact_type)
        meter = _importance_meter(fact.weight, maximum_weight)
        separator = "&nbsp;·&nbsp;"
        st.markdown(
            f"**{render_fact(fact)}**  \n"
            f"<span style='color:{_MUTED_TEXT_COLOUR};font-size:0.85em'>"
            f"{meter} {separator} {type_label} {separator} "
            f"waga {fact.weight:.2f}</span>",
            unsafe_allow_html=True,
        )
        source_count = len(fact.provenance)
        source_word = _plural_pl(source_count, "mecz", "mecze", "meczów")
        with st.expander(f"Źródło: {source_count} {source_word}", expanded=False):
            missing = [
                match_id for match_id in fact.provenance if match_id not in lookup
            ]
            for match_id in fact.provenance:
                record = lookup.get(match_id)
                if record is not None:
                    st.markdown(f"- {_format_provenance_row(record)}")
            if missing:
                missing_ids = ", ".join(str(match_id) for match_id in missing[:10])
                st.caption(
                    f"{len(missing)} "
                    f"{_plural_pl(len(missing), 'mecz', 'mecze', 'meczów')} "
                    f"spoza załadowanego zbioru historycznego "
                    f"(match_id: {missing_ids})."
                )
        if index < len(facts) - 1:
            st.markdown("")


def _render_diagnostics(prediction: PredictionResponse) -> None:
    """Per-target provenance: model, derived, or heuristic fallback -- never hidden."""
    with st.expander(
        "Panel diagnostyczny — skąd pochodzi każda liczba", expanded=False
    ):
        st.caption(
            "Każdy target jest osobnym, niezależnie rejestrowanym modelem. Tam, "
            "gdzie nie ma wytrenowanego modelu, silnik świadomie schodzi na "
            "heurystykę opartą na Elo albo na wartość domyślną — ta tabela "
            "pokazuje to wprost, zamiast prezentować wszystko jako wynik modelu."
        )
        rows: list[dict[str, str]] = []
        for target, label in _DIAGNOSTIC_TARGETS:
            version = prediction.model_versions.get(target)
            if version is None:
                source = "Wartość domyślna (brak modelu)"
                version_text = "—"
            elif "heuristic" in version:
                source = "Heurystyka Elo"
                version_text = version
            elif version.startswith("poisson["):
                source = "Wyprowadzone z modelu bramkowego (Poisson)"
                version_text = version
            else:
                source = "Wytrenowany model"
                version_text = version
            rows.append({"Target": label, "Źródło": source, "Wersja": version_text})
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)


_MATCH_STAT_LABELS: tuple[tuple[str, str], ...] = (
    ("corners", "Rzuty rożne"),
    ("cards", "Żółte kartki"),
    ("shots", "Strzały"),
    ("shots_on_target", "Strzały celne"),
    ("possession", "Posiadanie (%)"),
)


def _match_stat_rows(prediction: PredictionResponse) -> list[tuple[str, float, float]]:
    """Home/away stat pairs actually available for this prediction.

    `corners`/`cards`/`shots`/`shots_on_target`/`possession` are `None`
    on `PredictionResponse` whenever no model is registered for that target
    (see `predictions.service._predict_regression_optional`) -- e.g. the 10
    ESPN-derived stat models withdrawn 2026-08-20 for a confirmed Disney ToU
    violation (see `models/registry.py`'s `withdraw()` and
    `import_records.py`'s module docstring). Skips any stat that's `None`
    rather than indexing into it, per the explicit "prefer nothing over a
    flat fabricated value" decision -- shared by both the live Streamlit
    view and the PDF export so neither can silently drift out of sync with
    what the API actually returns.
    """
    rows: list[tuple[str, float, float]] = []
    for field_name, label in _MATCH_STAT_LABELS:
        pair = getattr(prediction, field_name)
        if pair is None:
            continue
        rows.append((label, pair["home"], pair["away"]))
    return rows


def _render_match_context(prediction: PredictionResponse) -> None:
    """Live context the model provably ignores -- labelled, kept off model numbers."""
    context = prediction.match_context.model_dump()
    populated = {key: value for key, value in context.items() if value is not None}
    if not populated:
        return
    with st.container(border=True):
        st.subheader("Kontekst informacyjny")
        st.caption(
            "Te dane **nie wchodzą do modelu** — mają potwierdzoną zerową "
            "ważność (ablacja 2026-07-24), bo w danych historycznych nie "
            "istnieje ich punktowy zapis. Podajemy je jako kontekst redakcyjny "
            "obok liczb modelu, nigdy jako coś, co model uwzględnił."
        )
        columns = st.columns(min(4, len(populated)))
        for index, (key, value) in enumerate(populated.items()):
            label, kind = _MATCH_CONTEXT_LABELS_PL.get(key, (key, "number"))
            if kind == "money":
                display = f"{float(value) / 1_000_000:.1f} mln €"
            else:
                display = f"{float(value):.2f}"
            columns[index % len(columns)].metric(label, display)


def _unicode_font_path() -> Path:
    """A TTF with full Polish coverage, taken from matplotlib's own bundle.

    matplotlib is already a hard dependency of this project, and it ships
    DejaVuSans -- so the PDF gets real "ś/ż/ó" on any machine that can run
    the app, with no system-font lookup (which would break the moment this
    is deployed to a Linux container) and no binary committed to the repo.
    fpdf's built-in fonts are Latin-1 only and would render them as blanks.
    """
    return Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSans.ttf"


def _matplotlib_png_bytes(figure: plt.Figure) -> bytes:
    """Render a matplotlib figure to PNG bytes and close it, so exports don't leak."""
    from io import BytesIO

    buffer = BytesIO()
    figure.savefig(buffer, format="png", facecolor="white", bbox_inches="tight")
    plt.close(figure)
    return buffer.getvalue()


def _pdf_outcome_chart_png(
    home_label: str, away_label: str, probabilities: dict[str, float]
) -> bytes:
    """1X2 probabilities as a horizontal bar chart, rendered for the PDF.

    matplotlib, not the app's own Plotly figure: rasterising the
    interactive chart would need `kaleido`, which needs a real Chrome --
    CONFIRMED REAL (2026-08-05) this machine has none anywhere worth
    depending on, and kaleido's own fallback (downloading one) fails here
    on a broken system CA trust store. matplotlib is already a hard
    dependency and needs neither, so the PDF's charts are built with it
    directly rather than trying to reuse the screen's Plotly objects.
    """
    labels = [f"Wygrana {away_label}", "Remis", f"Wygrana {home_label}"]
    values = [
        probabilities["away_win"],
        probabilities["draw"],
        probabilities["home_win"],
    ]
    colors = [_AWAY_COLOUR, _DRAW_COLOUR, _HOME_COLOUR]
    figure, axis = plt.subplots(figsize=(6.3, 2.1), dpi=200)
    bars = axis.barh(labels, values, color=colors)
    axis.set_xlim(0, 1.12)
    axis.invert_yaxis()
    for bar, value in zip(bars, values, strict=True):
        axis.text(
            value + 0.02,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.0%}",
            va="center",
            fontsize=11,
        )
    axis.set_xticks([])
    for spine in ("top", "right", "bottom"):
        axis.spines[spine].set_visible(False)
    figure.tight_layout()
    return _matplotlib_png_bytes(figure)


def _pdf_probability_chart_png(labels: list[str], values: list[float]) -> bytes:
    """Ranked N-team qualification-probability bars, rendered for the PDF.

    Generalizes `_pdf_outcome_chart_png` from a fixed three-category 1X2
    chart to an arbitrary number of teams -- same single-colour bar family
    as the on-screen `_probability_chart` (`_HOME_COLOUR` for every bar,
    since there's no home/away/draw split for a qualification race), same
    "value written past the bar end" label style as every other PDF chart
    in this module.
    """
    figure, axis = plt.subplots(
        figsize=(6.3, 0.5 * len(labels) + 1), dpi=200
    )
    bars = axis.barh(labels, values, color=_HOME_COLOUR)
    axis.set_xlim(0, 1.15)
    axis.invert_yaxis()
    for bar, value in zip(bars, values, strict=True):
        axis.text(
            value + 0.02,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.0%}",
            va="center",
            fontsize=10,
        )
    axis.set_xticks([])
    for spine in ("top", "right", "bottom"):
        axis.spines[spine].set_visible(False)
    figure.tight_layout()
    return _matplotlib_png_bytes(figure)


def _pdf_score_grid_png(
    home_label: str, away_label: str, matrix: list[list[float]]
) -> bytes:
    """The exact-scoreline probability grid, rendered for the PDF.

    Same single-hue gold family and same contrast logic as the app's own
    heat maps (`_metric_heatmap_cell_color`/`_readable_cell_text_color`),
    normalized against this grid's own peak cell -- consistent branding
    and a guaranteed-readable cell text colour, not a second colour scheme
    invented for print.
    """
    max_goals = len(matrix) - 1
    peak = max((value for row in matrix for value in row), default=0.0)
    colormap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "statsdesk_gold", [stop for _, stop in _METRIC_HEATMAP_COLORSCALE]
    )
    figure, axis = plt.subplots(figsize=(4.6, 4.2), dpi=200)
    axis.imshow(matrix, cmap=colormap, vmin=0, vmax=peak or 1.0, aspect="equal")
    for home_count in range(max_goals + 1):
        for away_count in range(max_goals + 1):
            value = matrix[home_count][away_count]
            if value < 0.005:
                continue
            normalized = value / peak if peak > 0 else 0.0
            cell_color = _metric_heatmap_cell_color(normalized)
            axis.text(
                away_count,
                home_count,
                f"{value:.0%}",
                ha="center",
                va="center",
                fontsize=9,
                color=_readable_cell_text_color(cell_color),
            )
    axis.set_xticks(range(max_goals + 1))
    axis.set_yticks(range(max_goals + 1))
    axis.set_xlabel(f"Bramki: {away_label}")
    axis.set_ylabel(f"Bramki: {home_label}")
    axis.xaxis.set_label_position("top")
    axis.xaxis.tick_top()
    axis.tick_params(length=0)
    for spine in axis.spines.values():
        spine.set_visible(False)
    figure.tight_layout()
    return _matplotlib_png_bytes(figure)


def _pdf_position_heatmap_png(
    team_labels: list[str], position_labels: list[str], matrix: list[list[float]]
) -> bytes:
    """Final-position probability heat map (teams x finishing places), for the PDF.

    Same logic as `_pdf_score_grid_png` -- same gold colormap
    (`_METRIC_HEATMAP_COLORSCALE`), same peak-normalized cell shading, same
    `_readable_cell_text_color` contrast pick -- just with the axes swapped
    for teams x final position instead of home goals x away goals, matching
    the on-screen `_heatmap_figure` used for "Gdzie kończy każda drużyna".
    """
    peak = max((value for row in matrix for value in row), default=0.0)
    colormap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "statsdesk_gold", [stop for _, stop in _METRIC_HEATMAP_COLORSCALE]
    )
    figure, axis = plt.subplots(
        figsize=(0.85 * len(position_labels) + 2, 0.6 * len(team_labels) + 1.6),
        dpi=200,
    )
    axis.imshow(matrix, cmap=colormap, vmin=0, vmax=peak or 1.0, aspect="auto")
    for row_index, row in enumerate(matrix):
        for column_index, value in enumerate(row):
            if value < 0.005:
                continue
            normalized = value / peak if peak > 0 else 0.0
            cell_color = _metric_heatmap_cell_color(normalized)
            axis.text(
                column_index,
                row_index,
                f"{value:.0%}",
                ha="center",
                va="center",
                fontsize=9,
                color=_readable_cell_text_color(cell_color),
            )
    axis.set_xticks(range(len(position_labels)))
    axis.set_xticklabels(position_labels, fontsize=8)
    axis.set_yticks(range(len(team_labels)))
    axis.set_yticklabels(team_labels, fontsize=9)
    axis.tick_params(length=0)
    for spine in axis.spines.values():
        spine.set_visible(False)
    figure.tight_layout()
    return _matplotlib_png_bytes(figure)


def _pdf_team_comparison_png(
    home_label: str, away_label: str, rows: list[tuple[str, float, float]]
) -> bytes:
    """Match-stat comparison as paired home/away bars, rendered for the PDF."""
    categories = [label for label, _, _ in rows]
    home_values = [home for _, home, _ in rows]
    away_values = [away for _, _, away in rows]
    y_positions = list(range(len(categories)))
    bar_height = 0.35
    peak_value = max([*home_values, *away_values], default=1.0) or 1.0

    figure, axis = plt.subplots(figsize=(6.3, 0.55 * len(categories) + 1), dpi=200)
    home_bars = axis.barh(
        [y + bar_height / 2 for y in y_positions],
        home_values,
        height=bar_height,
        color=_HOME_COLOUR,
        label=home_label,
    )
    away_bars = axis.barh(
        [y - bar_height / 2 for y in y_positions],
        away_values,
        height=bar_height,
        color=_AWAY_COLOUR,
        label=away_label,
    )
    for bar, value in zip(
        list(home_bars) + list(away_bars), home_values + away_values, strict=True
    ):
        axis.text(
            bar.get_width() + peak_value * 0.02,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.1f}",
            va="center",
            fontsize=9,
        )
    axis.set_xlim(0, peak_value * 1.15)
    axis.set_yticks(y_positions)
    axis.set_yticklabels(categories)
    axis.invert_yaxis()
    axis.set_xticks([])
    for spine in ("top", "right", "bottom"):
        axis.spines[spine].set_visible(False)
    # Above the plot, not inside it: CONFIRMED REAL (2026-08-06) an
    # in-axes legend placed in any corner collided with that row's own
    # value labels for some real prediction values (the last category's
    # bars happened to reach into the "lower right" corner).
    axis.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=2,
        frameon=False,
        fontsize=9,
    )
    figure.tight_layout()
    return _matplotlib_png_bytes(figure)


def _build_analysis_pdf(
    home_label: str,
    away_label: str,
    match_date: date,
    tournament: str,
    venue: str | None,
    prediction: PredictionResponse,
    facts: list[Fact],
    head_to_head_summary: str,
    score_matrix: list[list[float]],
    top_scorelines: list[tuple[str, float]],
    home_form_rows: list[dict[str, object]],
    away_form_rows: list[dict[str, object]],
) -> bytes:
    """PDF of the analysis on screen, for sending on to an editor.

    Match/date/venue sit at the top before anything else -- CONFIRMED REAL
    (2026-08-06): an editor picking a fixture off a stack of these needs
    that at a glance, not after reading past a logo. The prediction and
    stats sections are the same charts the screen shows (rendered with
    matplotlib rather than the app's own Plotly objects -- see
    `_pdf_outcome_chart_png`'s docstring for why), not the numbers spelled
    out as a sentence: a chart reads at a glance, and this format exists
    specifically to be handed to a working desk, not read start to finish.
    Only the head-to-head summary and the historical facts stay prose --
    provenance-bearing claims, not something a chart can carry. The two
    recent-form tables are the exact same rows as the on-screen "Pełna
    forma" expander (`_team_form_table_rows`), including the same
    per-match Frekwencja/Sędzia columns when Wikipedia has them.
    """
    from fpdf import FPDF
    from fpdf.fonts import FontFace

    class _BrandedPDF(FPDF):
        """A4 document that stamps the StatsDesk watermark on every page.

        Implemented as a `footer()` override rather than drawn once after the
        content: fpdf2 calls `footer()` for each page as it is closed, so a
        document that grows to two pages still carries the mark on both --
        which is the entire point of it being an anti-reuse measure.
        """

        def footer(self) -> None:
            """Draw the light-background watermark just above the bottom margin."""
            if not _WATERMARK_LIGHT.exists():
                return
            self.set_y(-18)
            self.image(str(_WATERMARK_LIGHT), x=self.w - 46, w=32)

    document = _BrandedPDF(format="A4")
    document.set_auto_page_break(auto=True, margin=22)
    document.add_page()
    font_path = str(_unicode_font_path())
    document.add_font("DejaVu", "", font_path)
    document.add_font(
        "DejaVu", "B", str(font_path).replace("DejaVuSans", "DejaVuSans-Bold")
    )

    def write(text: str, size: float = 10, style: str = "", gap: float = 5.5) -> None:
        document.set_font("DejaVu", style, size)
        # Explicit new_x/new_y: fpdf2's default leaves the cursor at the right
        # edge of the cell it just drew, so the next full-width multi_cell has
        # zero space left and raises "Not enough horizontal space".
        document.multi_cell(0, gap, text, new_x="LMARGIN", new_y="NEXT")

    usable_width = document.w - document.l_margin - document.r_margin

    def embed_chart(png_bytes: bytes, width: float, gap: float = 3) -> None:
        """Place a chart image at the left margin, then advance past it."""
        from io import BytesIO

        document.image(BytesIO(png_bytes), x=document.l_margin, w=width)
        document.ln(gap)

    def render_form_table(team_label: str, rows: list[dict[str, object]]) -> None:
        """The exact rows the "Pełna forma" expander shows, as a real PDF table.

        Skips entirely for a team with no matches on record before this
        date, rather than a heading over an empty table.
        """
        if not rows:
            return
        write(f"Forma: {team_label}", size=11, style="B", gap=6)
        frame = pd.DataFrame(rows)
        table_rows = [list(frame.columns)] + [
            ["" if pd.isna(value) else str(value) for value in record]
            for record in frame.itertuples(index=False)
        ]
        document.set_font("DejaVu", "", 8)
        with document.table(
            table_rows,
            text_align="LEFT",
            first_row_as_headings=True,
            headings_style=FontFace(family="DejaVu", emphasis="B", size_pt=8),
            line_height=5,
        ):
            pass
        document.ln(3)

    if _LOCKUP_LIGHT.exists():
        # The PDF page is white, so this is the `_light` variant -- the dark
        # one would be near-invisible on it.
        document.image(str(_LOCKUP_LIGHT), x=document.l_margin, y=10, w=52)
        document.set_y(28)
    else:
        write(_APP_NAME, size=9, style="B", gap=5)
    write(f"{home_label} — {away_label}", size=19, style="B", gap=10)
    write(f"{match_date}", size=11, style="B", gap=6)
    if venue:
        write(venue, size=10, gap=5.5)
    write(tournament, size=9, gap=5)
    document.ln(3)

    outcome = prediction.outcome_probabilities
    write("Prognoza wyniku", size=12, style="B", gap=7)
    embed_chart(
        _pdf_outcome_chart_png(
            home_label, away_label, outcome.model_dump()
        ),
        width=usable_width,
    )
    write(
        f"Oczekiwane bramki (xG): {prediction.xg['home']:.2f} — "
        f"{prediction.xg['away']:.2f}    "
        f"Obie drużyny strzelą: {prediction.btts:.0%}    "
        f"Powyżej 2.5: {prediction.over_under['over_2_5']:.0%}",
        size=9,
    )
    document.ln(3)

    write("Prawdopodobieństwo dokładnego wyniku", size=12, style="B", gap=7)
    embed_chart(
        _pdf_score_grid_png(home_label, away_label, score_matrix),
        width=min(usable_width, 100),
    )
    if top_scorelines:
        likeliest = "    ".join(
            f"{score} ({value:.0%})" for score, value in top_scorelines
        )
        write(f"Najbardziej prawdopodobne: {likeliest}", size=9)
    document.ln(3)

    write("Bilans bezpośredni", size=12, style="B", gap=7)
    write(head_to_head_summary)
    document.ln(3)

    render_form_table(home_label, home_form_rows)
    render_form_table(away_label, away_form_rows)

    write("Statystyki meczowe (prognoza)", size=12, style="B", gap=7)
    stat_rows = _match_stat_rows(prediction)
    if stat_rows:
        embed_chart(
            _pdf_team_comparison_png(home_label, away_label, stat_rows),
            width=usable_width,
        )
    else:
        write(
            "Brak dostępnych statystyk (rzuty rożne, kartki, strzały, "
            "posiadanie) -- źródło tych danych zostało wycofane z produkcji "
            "z powodu ograniczeń licencyjnych.",
            size=9,
        )
        document.ln(3)

    write("Nietypowe fakty", size=12, style="B", gap=7)
    if facts:
        lookup = _match_lookup()
        for fact in facts:
            write(f"• {render_fact(fact)}")
            sources = [
                _format_provenance_row(lookup[match_id])
                for match_id in fact.provenance
                if match_id in lookup
            ][:3]
            if sources:
                write("    źródło: " + "; ".join(sources), size=7.5, gap=4.5)
    else:
        write("• (brak faktów historycznych dla tej pary)")

    document.ln(4)
    generated_at = datetime.now(tz=UTC).strftime("%Y-%m-%d %H:%M UTC")
    write(
        f"Wygenerowano {generated_at} przez {_APP_NAME}. Liczby pochodzą z modelu "
        "statystycznego uczonego na historycznych wynikach — nie są kursem ani "
        "zakładem bukmacherskim.",
        size=7.5,
        gap=4,
    )
    return bytes(document.output())


def _build_outlook_pdf(
    division: str,
    group: str,
    season: str,
    standings_frame: pd.DataFrame,
    probability_labels: list[str],
    probability_values: list[float],
    position_team_labels: list[str],
    position_labels: list[str],
    position_matrix: list[list[float]],
    whatif: dict[str, object] | None,
) -> bytes:
    """PDF of the "Perspektywy grupy" tab, mirroring `_build_analysis_pdf`'s pattern.

    Same `_BrandedPDF` class (watermark on every page via `footer()`), same
    DejaVu font, same `write()`/`embed_chart()` helpers -- deliberately not
    reimplemented, just re-derived per call, since `_build_analysis_pdf`
    itself keeps them as function-local closures rather than module-level
    helpers.

    `standings_frame` is rendered as a real fpdf2 table
    (`document.table(...)`), not an image, mirroring `render_form_table`'s
    approach in `_build_analysis_pdf` -- an editor/reader needs to select
    and copy numbers out of it, which a rasterized table would prevent.
    """
    from fpdf import FPDF
    from fpdf.fonts import FontFace

    class _BrandedPDF(FPDF):
        """A4 document that stamps the StatsDesk watermark on every page."""

        def footer(self) -> None:
            """Draw the light-background watermark just above the bottom margin."""
            if not _WATERMARK_LIGHT.exists():
                return
            self.set_y(-18)
            self.image(str(_WATERMARK_LIGHT), x=self.w - 46, w=32)

    document = _BrandedPDF(format="A4")
    document.set_auto_page_break(auto=True, margin=22)
    document.add_page()
    font_path = str(_unicode_font_path())
    document.add_font("DejaVu", "", font_path)
    document.add_font(
        "DejaVu", "B", str(font_path).replace("DejaVuSans", "DejaVuSans-Bold")
    )

    def write(text: str, size: float = 10, style: str = "", gap: float = 5.5) -> None:
        document.set_font("DejaVu", style, size)
        document.multi_cell(0, gap, text, new_x="LMARGIN", new_y="NEXT")

    usable_width = document.w - document.l_margin - document.r_margin

    def embed_chart(png_bytes: bytes, width: float, gap: float = 3) -> None:
        """Place a chart image at the left margin, then advance past it."""
        from io import BytesIO

        document.image(BytesIO(png_bytes), x=document.l_margin, w=width)
        document.ln(gap)

    def render_table(frame: pd.DataFrame) -> None:
        """A pandas frame as a real fpdf2 table -- same pattern as `render_form_table`."""
        table_rows = [list(frame.columns)] + [
            ["" if pd.isna(value) else str(value) for value in record]
            for record in frame.itertuples(index=False)
        ]
        document.set_font("DejaVu", "", 8)
        with document.table(
            table_rows,
            text_align="LEFT",
            first_row_as_headings=True,
            headings_style=FontFace(family="DejaVu", emphasis="B", size_pt=8),
            line_height=5,
        ):
            pass
        document.ln(3)

    if _LOCKUP_LIGHT.exists():
        document.image(str(_LOCKUP_LIGHT), x=document.l_margin, y=10, w=52)
        document.set_y(28)
    else:
        write(_APP_NAME, size=9, style="B", gap=5)
    write(f"Dywizja {division}, grupa {group}", size=19, style="B", gap=10)
    write(f"Sezon {season}", size=10, gap=5.5)
    generated_at = datetime.now(tz=UTC).strftime("%Y-%m-%d %H:%M UTC")
    write(f"Wygenerowano {generated_at}", size=9, gap=5)
    document.ln(3)

    write("Aktualna tabela", size=12, style="B", gap=7)
    render_table(standings_frame)
    write(
        "⚠️ Uproszczenie: przy równej liczbie punktów tabela rozstrzyga kolejność "
        "wyłącznie przez bilans bramek i bramki strzelone. Regulaminy UEFA/FIFA "
        "uwzględniają najpierw bezpośrednie spotkania — w ściśle remisowych "
        "przypadkach kolejność może się różnić.",
        size=8,
        gap=4.5,
    )
    document.ln(3)

    write("Prawdopodobieństwo awansu", size=12, style="B", gap=7)
    embed_chart(
        _pdf_probability_chart_png(probability_labels, probability_values),
        width=usable_width,
    )
    document.ln(2)

    write("Gdzie kończy każda drużyna", size=12, style="B", gap=7)
    embed_chart(
        _pdf_position_heatmap_png(
            position_team_labels, position_labels, position_matrix
        ),
        width=min(usable_width, 150),
    )
    write(
        "Każdy wiersz sumuje się do 100% — drużyna kończy dokładnie na jednym miejscu.",
        size=8,
        gap=4.5,
    )

    if whatif is not None:
        document.ln(3)
        write("Scenariusz „co jeśli”", size=12, style="B", gap=7)
        write(
            f"Mecz: {whatif['match_label']}    Ustalony wynik: "
            f"{whatif['home_label']} {whatif['home_goals']}:{whatif['away_goals']} "
            f"{whatif['away_label']}",
            size=9,
            gap=5,
        )
        render_table(whatif["comparison_frame"])

    document.ln(4)
    write(
        f"Wygenerowano przez {_APP_NAME}. Procenty pochodzą z symulacji Monte "
        "Carlo opartej na prognozach modelu statystycznego dla pozostałych "
        "meczów grupy — nie są kursem ani zakładem bukmacherskim.",
        size=7.5,
        gap=4,
    )
    return bytes(document.output())


def _team_picker(
    key_prefix: str, default_home: str, default_away: str
) -> tuple[str, str] | None:
    """Two team dropdowns showing Polish names but returning canonical English ones.

    Restricted to `_predictable_team_names()` -- see that function for the
    real bug this prevents.
    """
    teams = _predictable_team_names()
    if not teams:
        st.warning(
            "Brak lokalnych danych historycznych. Uruchom najpierw "
            "`python -m football_ai.scrapers.sources.international_results`."
        )
        return None
    column_home, column_away = st.columns(2)
    with column_home:
        home = st.selectbox(
            "Gospodarz",
            teams,
            index=teams.index(default_home) if default_home in teams else 0,
            format_func=nominative_pl,
            key=f"{key_prefix}_home",
        )
    with column_away:
        away_options = [team for team in teams if team != home]
        away_index = (
            away_options.index(default_away) if default_away in away_options else 0
        )
        away = st.selectbox(
            "Gość",
            away_options,
            index=away_index,
            format_func=nominative_pl,
            key=f"{key_prefix}_away",
        )
    return home, away


# --------------------------------------------------------------------------
# Tab 1: pre-match preview
# --------------------------------------------------------------------------


def _select_recap(match_id: int) -> None:
    """Open one played match's recap -- `on_click` callback, see `_select_fixture`."""
    st.session_state["recap_result"] = match_id


def _clear_recap() -> None:
    """Return to the results list."""
    st.session_state.pop("recap_result", None)


def _clear_fixture() -> None:
    """Return to the fixture list."""
    st.session_state.pop("selected_fixture", None)


def _select_fixture(
    home: str, away: str, match_date: date, tournament: str, venue: str | None = None
) -> None:
    """Move to the analysis screen for one fixture.

    Used as an `on_click` callback, never called inline: a callback runs
    *before* the script re-executes, so the new screen renders on the very
    next pass without an explicit `st.rerun()`. That matters because
    `st.rerun()` resets `st.tabs` to the first tab -- clicking a match in the
    "Po meczu" tab used to bounce the user back to "Nadchodzące mecze".

    `venue` defaults to `None`: the manual-pair picker has no fixture row to
    read it from (it's a hypothetical matchup), so the PDF header simply
    omits that line rather than showing a placeholder.
    """
    st.session_state["selected_fixture"] = (home, away, match_date, tournament, venue)


def _fixture_venue(row: pd.Series) -> str | None:
    """The venue text a fixture row carries, whichever column name the source used.

    CONFIRMED REAL (2026-08-06): checking `not venue` to decide whether to
    fall back to `stadium_name` never falls back, because `not float("nan")`
    is `False` -- NaN is truthy under plain Python truthiness, so a row
    with `stadium=NaN` and a real `stadium_name` (e.g. Thailand-Singapore,
    `stadium_name="Rajamangala Stadium"`) silently returned no venue at
    all. Needs `pd.isna()`, not truthiness, to tell "empty" from "NaN".
    """
    venue = row.get("stadium") if "stadium" in row.index else None
    if (venue is None or pd.isna(venue)) and "stadium_name" in row.index:
        venue = row.get("stadium_name")
    return str(venue) if venue is not None and not pd.isna(venue) else None


def _fixture_left_label(row: pd.Series) -> str:
    """Kick-off time when the source really has one, otherwise the venue.

    Wikipedia's Nations League tables carry a date with no time (verified:
    every timestamp is exactly 00:00), so showing "00:00" there would be an
    invented kick-off. football_api rows do carry a real time.
    """
    timestamp = pd.Timestamp(row["match_date"])
    if timestamp.hour or timestamp.minute:
        return timestamp.strftime("%H:%M")
    return _fixture_venue(row) or "—"


def _render_fixture_card(
    row: pd.Series, elo_ratings: dict[str, float], key: str
) -> None:
    """One fixture as a full-width, visibly clickable card."""
    home = str(row["home_team_name"])
    away = str(row["away_team_name"])
    tournament = str(row.get("tournament") or "UEFA Nations League")
    with st.container(border=True):
        left_column, teams_column, action_column = st.columns([2, 3, 2])
        with left_column:
            st.caption(_fixture_left_label(row))
            st.caption(tournament)
        with teams_column:
            st.markdown(f"**{nominative_pl(home)}**  \n**{nominative_pl(away)}**")
        with action_column:
            home_elo = elo_ratings.get(canonicalize_team_name(home))
            away_elo = elo_ratings.get(canonicalize_team_name(away))
            st.caption(
                f"Elo {home_elo:.0f} — {away_elo:.0f}"
                if home_elo and away_elo
                else "Elo: brak"
            )
            st.button(
                "Analiza",
                key=key,
                width="stretch",
                on_click=_select_fixture,
                args=(
                    home,
                    away,
                    pd.Timestamp(row["match_date"]).date(),
                    tournament,
                    _fixture_venue(row),
                ),
            )


def _render_competition_filter(options: list[str], key: str) -> str:
    """The shared "Filtruj rozgrywki" control -- identical on both screens.

    The choice is mirrored into a plain session_state key because Streamlit
    discards a widget's own state on any run where the widget isn't rendered,
    and the detail screens render none of these -- without the mirror, going
    back to a list always snapped the filter to "Wszystkie".
    """
    memory_key = f"{key}_choice"
    choice = st.segmented_control(
        "Filtruj rozgrywki",
        ["Wszystkie", *options],
        default=st.session_state.get(memory_key, "Wszystkie"),
        key=key,
    )
    resolved = choice or "Wszystkie"
    st.session_state[memory_key] = resolved
    return resolved


def _render_result_card(row: pd.Series, key: str) -> None:
    """A played match: same shape as a fixture card, but the score replaces the time."""
    home = str(row["home_team_name"])
    away = str(row["away_team_name"])
    with st.container(border=True):
        score_column, teams_column, action_column = st.columns([2, 3, 2])
        with score_column:
            st.markdown(f"### {int(row['home_score'])} : {int(row['away_score'])}")
            st.caption(str(row.get("tournament") or ""))
        with teams_column:
            st.markdown(f"**{nominative_pl(home)}**  \n**{nominative_pl(away)}**")
        with action_column:
            st.caption(str(pd.Timestamp(row["match_date"]).date()))
            st.button(
                "Podsumowanie",
                key=key,
                width="stretch",
                on_click=_select_recap,
                args=(int(row["match_id"]),),
            )


def _render_manual_pair_expander() -> None:
    """The escape hatch for a pairing that isn't on any schedule."""
    with st.expander("Dowolna para (mecz spoza terminarza)", expanded=False):
        st.caption(
            "Lista obejmuje reprezentacje z aktualnym rankingiem Elo. Drużyny bez "
            "rankingu (nieistniejące państwa, terytoria zależne, zespoły spoza FIFA) "
            "są pominięte — silnik nie ma dla nich danych, na których mógłby oprzeć "
            "sensowną prognozę."
        )
        picked = _team_picker("preview", "Poland", "Sweden")
        if picked is None:
            return
        home, away = picked
        column_date, column_tournament = st.columns(2)
        with column_date:
            match_date = st.date_input(
                "Data meczu", value=date.today(), key="preview_date"
            )
        with column_tournament:
            tournament = st.text_input(
                "Turniej", value="UEFA Nations League", key="preview_tournament"
            )
        if st.button("Przygotuj analizę", type="primary", key="preview_manual_run"):
            _select_fixture(home, away, match_date, tournament)


def _render_fixture_list() -> None:
    """Screen 1: the next upcoming matches, newest competition filter on top."""
    with st.expander("Ustawienia", expanded=False):
        # Streamlit drops a widget's session_state entry on any run where the
        # widget isn't rendered, and the analysis screen renders none of
        # these -- mirroring into plain keys keeps the choice across screens.
        season = st.text_input(
            "Sezon Ligi Narodów",
            value=st.session_state.get("season_choice", "2026–27"),
            key="preview_season",
        )
    st.session_state["season_choice"] = season

    with st.spinner("Pobieram terminarz..."):
        schedule, api_contributed = _upcoming_fixtures(season)

    if schedule.empty:
        st.warning(
            f"Nie udało się pobrać żadnego terminarza (sezon {season}). "
            "Sprawdź sezon w ustawieniach albo użyj „Dowolna para”."
        )
        _render_manual_pair_expander()
        return

    competitions = sorted(schedule["tournament"].dropna().unique())
    selected = _render_competition_filter(list(competitions), key="home_competition")
    filtered = (
        schedule
        if selected == "Wszystkie"
        else schedule.loc[schedule["tournament"] == selected]
    )

    # Team filter drawn from the schedule itself, not from every team that ever
    # played: offering a side with no upcoming fixture would only ever lead to
    # an empty list.
    scheduled_teams = sorted(
        set(filtered["home_team_name"].dropna())
        | set(filtered["away_team_name"].dropna())
    )
    selected_team = st.selectbox(
        "Filtruj drużynę",
        ["Wszystkie", *scheduled_teams],
        format_func=lambda name: (
            "Wszystkie" if name == "Wszystkie" else nominative_pl(name)
        ),
        key="home_team_filter",
    )
    if selected_team != "Wszystkie":
        filtered = filtered.loc[
            (filtered["home_team_name"] == selected_team)
            | (filtered["away_team_name"] == selected_team)
        ]

    if filtered.empty:
        label = selected if selected != "Wszystkie" else "wybranych filtrów"
        if selected_team != "Wszystkie":
            label = nominative_pl(selected_team)
        st.info(f"Brak nadchodzących spotkań dla {label}.")
        _render_manual_pair_expander()
        return

    if not api_contributed:
        st.caption(
            "Terminarz pochodzi z Ligi Narodów (Wikipedia). Rozszerzenie o inne "
            "rozgrywce wymaga płatnego planu api-football — obecny plan nie "
            "udostępnia dat poza dzisiejszą."
        )

    upcoming = filtered.head(10)
    elo_ratings = _load_elo_ratings()
    fixtures_by_day = upcoming.groupby(upcoming["match_date"].dt.date, sort=True)
    for day, day_fixtures in fixtures_by_day:
        st.markdown(f"##### {_format_date_header_pl(day)}")
        for _, row in day_fixtures.iterrows():
            fixture_key = f"{row['home_team_name']}_{row['away_team_name']}_{day}"
            _render_fixture_card(row, elo_ratings, key=f"fixture_{fixture_key}")

    _render_manual_pair_expander()


def _render_match_preview_tab() -> None:
    """Master-detail: a group's fixture list, or one fixture's full analysis."""
    stored = st.session_state.get("selected_fixture")
    if stored is None:
        _render_fixture_list()
        return

    # Only the selection is cleared -- the season lives under its own mirrored
    # key, so the list comes back exactly as it was.
    st.button("← Wszystkie mecze", key="preview_back", on_click=_clear_fixture)
    if st.session_state.get("selected_fixture") is None:
        return

    home, away, match_date, tournament, venue = stored

    with st.spinner("Liczę predykcję..."):
        prediction = _cached_prediction(home, away, match_date, tournament)
        facts = _cached_facts(
            home, away, pd.Timestamp(match_date, tz="UTC").isoformat(), 10
        )

    home_label = nominative_pl(home)
    away_label = nominative_pl(away)

    st.header(f"{home_label} — {away_label}")
    caption = f"{match_date} · {tournament}"
    st.caption(f"{caption} · {venue}" if venue else caption)
    _render_match_officials(home, away, match_date)

    _warn_on_missing_elo(home, away)
    _render_structural_context(home, away, pd.Timestamp(match_date, tz="UTC"))

    score_figure, top_scorelines, tail_mass = _score_grid_figure(
        home_label, away_label, prediction.goals["home"], prediction.goals["away"]
    )

    with st.container(border=True):
        st.subheader("Prognoza wyniku")
        st.plotly_chart(
            _outcome_chart(
                home_label, away_label, prediction.outcome_probabilities.model_dump()
            ),
            width="stretch",
        )
        st.metric(
            "Oczekiwane bramki (xG)",
            f"{prediction.xg['home']:.2f} : {prediction.xg['away']:.2f}",
        )

    with st.container(border=True):
        # Replaces a single "predicted score". One scoreline reads as a
        # confident call the model never makes -- the likeliest cell here is
        # typically only ~10%, which the grid shows and a metric hides.
        st.subheader("Prawdopodobieństwo dokładnego wyniku")
        st.plotly_chart(score_figure, width="stretch")
        likeliest = " · ".join(
            f"**{score}** {value:.0%}" for score, value in top_scorelines
        )
        st.markdown(f"Najbardziej prawdopodobne: {likeliest}")
        st.caption(
            f"Siatka pokrywa 0–5 bramek każdej drużyny; poza nią pozostaje "
            f"{tail_mass:.1%} prawdopodobieństwa (wyniki z 6+ bramkami)."
        )

    with st.container(border=True):
        st.subheader("Bramki")
        over_under = sorted(prediction.over_under.items())
        # Column count derived from the response rather than hardcoded, so a
        # future extra threshold can't silently drop off the end of the row.
        goal_columns = st.columns(1 + len(over_under))
        goal_columns[0].metric("Obie drużyny strzelą", f"{prediction.btts:.0%}")
        for column, (label, value) in zip(goal_columns[1:], over_under, strict=True):
            threshold = label.replace("over_", "").replace("_", ".")
            column.metric(f"Powyżej {threshold}", f"{value:.0%}")

    with st.container(border=True):
        st.subheader("Statystyki meczowe")
        stat_rows = _match_stat_rows(prediction)
        if stat_rows:
            st.plotly_chart(
                _team_comparison_chart(home_label, away_label, stat_rows),
                width="stretch",
            )
        else:
            st.caption(
                "Brak dostępnych statystyk meczowych (rzuty rożne, kartki, "
                "strzały, posiadanie) — źródło tych danych zostało wycofane "
                "z produkcji z powodu ograniczeń licencyjnych; silnik nie "
                "zgaduje wartości, których nie ma czym potwierdzić."
            )

    # Head-to-head and streak facts already appear as structured context
    # above, so they're dropped here rather than repeated -- what's left is
    # genuinely the trivia this section claims to be.
    trivia = [fact for fact in facts if fact.fact_type not in _STRUCTURAL_FACT_TYPES]
    with st.container(border=True):
        st.subheader("Nietypowe fakty")
        st.caption(
            "Ciekawostki historyczne, nie kontekst meczu — uszeregowane wagą. "
            "Każde zdanie rozwija się do listy meczów, z których powstało."
        )
        _render_facts(trivia)

    _render_match_context(prediction)
    _render_travel_fatigue(home, away, match_date, venue)
    _render_group_outlook_link(home, away, tournament)
    _render_diagnostics(prediction)


def _warn_on_missing_elo(home: str, away: str) -> None:
    """Say so when a side has no Elo rating, instead of quietly guessing.

    The manual picker already excludes ratingless teams, but a real fixture
    from the schedule can still involve one (e.g. an ASEAN Championship side
    eloratings.net doesn't track). Without a rating the model receives NaN,
    its imputer substitutes the training median, and it then reasons about
    that team as an average national side -- the same mechanism that once
    produced "Panjab 54% to beat Sweden". Better to flag it on the screen
    than to present the number as if it were on equal footing.
    """
    elo_ratings = _load_elo_ratings()
    if not elo_ratings:
        return
    missing = [
        nominative_pl(team)
        for team in (home, away)
        if canonicalize_team_name(team) not in elo_ratings
    ]
    if not missing:
        return
    st.warning(
        f"Brak aktualnego rankingu Elo dla: {', '.join(missing)}. Model zastępuje "
        "go wartością typową dla przeciętnej reprezentacji, więc tę prognozę należy "
        "traktować z dużą ostrożnością."
    )


def _render_group_outlook_link(home: str, away: str, tournament: str) -> None:
    """Jump to the group simulation, but only when this match is really in a group.

    A friendly belongs to no group, so rather than showing a disabled
    control the button simply isn't rendered.
    """
    if "nations league" not in tournament.lower():
        return
    location = _find_nations_league_group(
        home, away, st.session_state.get("season_choice", "2026–27")
    )
    if location is None:
        return
    division, group = location
    if st.button(
        f"Zobacz perspektywy grupy (dywizja {division}, grupa {group})",
        key="preview_to_outlook",
    ):
        st.session_state["outlook_division"] = division
        st.session_state["outlook_group"] = group
        st.session_state["outlook_autorun"] = True
        st.rerun()


@st.cache_data(ttl=1800, show_spinner=False)
def _find_nations_league_group(
    home: str, away: str, season: str
) -> tuple[str, int] | None:
    """Which Nations League division/group this pairing belongs to, if any."""
    schedule = _nations_league_schedule(season)
    if schedule.empty or "division" not in schedule.columns:
        return None
    match = schedule.loc[
        (schedule["home_team_name"] == home) & (schedule["away_team_name"] == away)
    ]
    if match.empty:
        return None
    row = match.iloc[0]
    return str(row["division"]), int(row["group"])


# --------------------------------------------------------------------------
# Tab 2: post-match recap
# --------------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def _recent_results(
    tournament: str | None, team: str | None, limit: int = 10
) -> pd.DataFrame:
    """The most recent played matches, optionally narrowed by competition and/or team.

    Both filters are independent and optional: neither gives the latest
    matches overall, one gives that slice, both give the intersection. A
    plain filter over `records_frame` -- no new data source involved.
    """
    frame = _load_records_frame()
    if frame.empty:
        return frame
    if tournament:
        frame = frame.loc[frame["tournament"] == tournament]
    if team:
        canonical = canonicalize_team_name(team)
        frame = frame.loc[
            (frame["home_team_name"] == canonical)
            | (frame["away_team_name"] == canonical)
        ]
    return (
        frame.sort_values("match_date", ascending=False)
        .head(limit)
        .reset_index(drop=True)
    )


def _render_recent_results_list(tournaments: list[str]) -> None:
    """Screen 1 of the recap tab: recently played matches, with two optional filters."""
    # The historical record spans ~100 competitions, and offering every one
    # buried the match list under ten rows of buttons. Capped to the most
    # recently played, and said out loud rather than silently truncated --
    # the team filter still reaches everything older.
    shortlist = _major_recent_tournaments() or tournaments[:8]
    selected_tournament = _render_competition_filter(
        shortlist, key="recap_competition"
    )
    if len(tournaments) > len(shortlist):
        st.caption(
            f"Pokazano {len(shortlist)} najnowszych rozgrywek z {len(tournaments)} "
            "w zbiorze — starsze znajdziesz przez filtr drużyny."
        )
    teams = _known_team_names()
    selected_team = st.selectbox(
        "Filtruj drużynę",
        ["Wszystkie", *teams],
        format_func=lambda name: (
            "Wszystkie" if name == "Wszystkie" else nominative_pl(name)
        ),
        key="recap_team",
    )

    matches = _recent_results(
        None if selected_tournament == "Wszystkie" else selected_tournament,
        None if selected_team == "Wszystkie" else selected_team,
    )
    if matches.empty:
        label = (
            selected_tournament
            if selected_tournament != "Wszystkie"
            else "wybranych filtrów"
        )
        st.info(f"Brak rozegranych spotkań dla {label}.")
        return

    results_by_day = matches.groupby(matches["match_date"].dt.date, sort=False)
    for day, day_matches in results_by_day:
        st.markdown(f"##### {_format_date_header_pl(day)}")
        for _, row in day_matches.iterrows():
            # Keyed on match_id, never on position: the list changes shape when
            # a filter changes, and a positional key would then bind the same
            # widget to a different match -- clicking one row opened another.
            _render_result_card(row, key=f"result_{int(row['match_id'])}")


def _render_match_statistics(
    match_row: pd.Series, home_label: str, away_label: str
) -> None:
    """Real recorded stats for this match -- only the ones that actually exist.

    Krok 0 measured these columns at 14.8% coverage over the last five years
    (1.7% overall), so they are rendered per match and never as a fixed block
    of "brak danych" rows: a match either has them or the section says so in
    one sentence.
    """
    available = [
        (label, float(match_row[f"home_{column}"]), float(match_row[f"away_{column}"]))
        for label, column in (
            ("Strzały", "shots"),
            ("Strzały celne", "shots_on_target"),
            ("Rzuty rożne", "corners"),
            ("Posiadanie (%)", "possession_pct"),
            ("Żółte kartki", "yellow_cards"),
            ("Czerwone kartki", "red_cards"),
            ("Faule", "fouls_committed"),
            ("Spalone", "offsides"),
            ("Obrony", "saves"),
            ("Podania", "total_passes"),
        )
        if f"home_{column}" in match_row.index
        and pd.notna(match_row.get(f"home_{column}"))
        and pd.notna(match_row.get(f"away_{column}"))
    ]
    with st.container(border=True):
        st.subheader("Statystyki meczowe")
        if not available:
            st.caption(
                "Brak zapisanych statystyk dla tego meczu — szczegółowe dane "
                "statystyczne są dostępne głównie dla nowszych spotkań."
            )
            return
        st.plotly_chart(
            _team_comparison_chart(home_label, away_label, available), width="stretch"
        )


def _render_goalscorers(match_id: int, home_team: str, away_team: str) -> None:
    """Chronological goal list with minute, penalty and own-goal markers."""
    goalscorers = _load_goalscorers_frame()
    if goalscorers.empty or "match_id" not in goalscorers.columns:
        return
    goals = goalscorers.loc[goalscorers["match_id"] == match_id].copy()
    if goals.empty:
        return
    goals = goals.sort_values("minute", na_position="last")
    with st.container(border=True):
        st.subheader("Strzelcy")
        for _, goal in goals.iterrows():
            minute = goal.get("minute")
            minute_text = f"{int(minute)}'" if pd.notna(minute) else "—"
            markers = []
            if bool(goal.get("penalty")):
                markers.append("k.")
            if bool(goal.get("own_goal")):
                markers.append("sam.")
            marker_text = f" ({', '.join(markers)})" if markers else ""
            team_label = nominative_pl(str(goal["team_name"]))
            st.markdown(
                f"**{minute_text}** {goal['scorer']} — {team_label}{marker_text}"
            )
    _ = (home_team, away_team)


def _render_recap_tab() -> None:
    """The real result plus the facts as they stood right after the match.

    Mirrors `api/routes/matches.py::match_recap`, including its `+1s` cutoff:
    an exclusive `as_of` on the match's own date would exclude every fact
    this match itself created (its own streak extension, its own milestone).
    """
    tournaments = _tournaments_by_recency()
    if not tournaments:
        st.warning(
            "Brak lokalnych danych historycznych. Uruchom najpierw "
            "`python -m football_ai.scrapers.sources.international_results`."
        )
        return

    stored_match_id = st.session_state.get("recap_result")
    if stored_match_id is None:
        _render_recent_results_list(tournaments)
        return

    st.button("← Wszystkie mecze", key="recap_back", on_click=_clear_recap)
    if st.session_state.get("recap_result") is None:
        return

    records_frame = _load_records_frame()
    match_rows = records_frame.loc[records_frame["match_id"] == stored_match_id]
    if match_rows.empty:
        st.warning("Nie znaleziono tego meczu w danych historycznych.")
        return
    result_row = match_rows.iloc[0]
    home = str(result_row["home_team_name"])
    away = str(result_row["away_team_name"])
    match_date = pd.Timestamp(result_row["match_date"]).date()
    home_label = nominative_pl(home)
    away_label = nominative_pl(away)

    as_of = pd.Timestamp(match_date, tz="UTC") + pd.Timedelta(seconds=1)
    with st.spinner("Zbieram fakty..."):
        facts = _cached_facts(home, away, as_of.isoformat(), 10)

    st.header(f"{home_label} — {away_label}")
    tournament = str(result_row.get("tournament") or "").strip()
    st.caption(f"{match_date}{f' · {tournament}' if tournament else ''}")

    with st.container(border=True):
        st.subheader("Wynik")
        st.markdown(
            f"<div style='font-size:2.6em;font-weight:600'>"
            f"{int(result_row['home_score'])} : {int(result_row['away_score'])}</div>",
            unsafe_allow_html=True,
        )
        # Silent when Wikipedia has nothing for this match -- no placeholder,
        # no "brak danych", just the score on its own as before.
        detail_text = _format_match_detail(
            _wikipedia_match_detail(
                home,
                match_date,
                int(result_row["home_score"]),
                int(result_row["away_score"]),
            )
        )
        if detail_text:
            st.caption(detail_text)

    _render_match_statistics(result_row, home_label, away_label)
    _render_goalscorers(int(stored_match_id), home, away)

    with st.container(border=True):
        st.subheader("Fakty historyczne")
        st.caption(
            "Stan na moment tuż po tym meczu — z faktami, które stworzył on sam."
        )
        _render_facts(facts)


# --------------------------------------------------------------------------
# Tab 3: competition outlook + what-if scenarios
# --------------------------------------------------------------------------


def _render_competition_outlook_tab() -> None:
    """Real remaining Nations League fixtures + predictions, fed to Monte Carlo."""
    st.markdown(
        "Rozgrywamy pozostałe mecze grupy dziesiątki tysięcy razy. Za każdym razem "
        "losujemy wyniki zgodnie z prognozą modelu dla konkretnego spotkania, "
        "układamy tabelę końcową i zapisujemy, kto gdzie skończył. Procent obok "
        "drużyny to po prostu udział rozegrań, w których zajęła miejsce dające awans."
    )
    teams = _predictable_team_names()
    column_team, column_spots = st.columns(2)
    with column_team:
        team_query = (
            st.selectbox(
                "Drużyna",
                teams,
                index=teams.index("Poland") if "Poland" in teams else 0,
                format_func=nominative_pl,
                key="outlook_team",
            )
            if teams
            else st.text_input("Drużyna", value="Poland", key="outlook_team_text")
        )
    with column_spots:
        n_qualifying_spots = st.number_input(
            "Liczba awansujących miejsc",
            min_value=1,
            max_value=4,
            value=1,
            key="outlook_spots",
        )
    with st.expander("Ustawienia", expanded=False):
        season = st.text_input("Sezon", value="2026–27", key="outlook_season")
        n_simulations = st.number_input(
            "Liczba rozegrań (symulacji Monte Carlo)",
            min_value=100,
            max_value=50_000,
            value=10_000,
            step=1000,
            key="outlook_simulations",
            help="Więcej rozegrań = stabilniejsze procenty, ale dłuższe liczenie.",
        )

    autorun = st.session_state.pop("outlook_autorun", False)
    if st.button("Symuluj grupę", type="primary", key="outlook_run") or autorun:
        uefa_source = UefaNationsLeagueSource()
        try:
            # A jump from the pre-match screen already knows the group, so it
            # skips the four-page Wikipedia search that finding it requires.
            requested_division = st.session_state.pop("outlook_division", None)
            requested_group = st.session_state.pop("outlook_group", None)
            if requested_division and requested_group:
                division, group = requested_division, requested_group
            else:
                with st.spinner("Szukam grupy drużyny..."):
                    division, group = uefa_source.find_team_group(
                        team_query, season=season
                    )
        except ValueError as error:
            st.session_state["outlook_result"] = {"error": str(error)}
        else:
            with st.spinner("Pobieram terminarz i liczę predykcje..."):
                fixtures = uefa_source.fetch_group_matches(
                    division, group, season=season
                )
                if fixtures.empty:
                    st.session_state["outlook_result"] = {
                        "error": f"Brak meczów w grupie {group} dywizji {division}."
                    }
                else:
                    simulation_input = build_simulation_input(
                        fixtures,
                        _load_prediction_service(),
                        elo_ratings=_load_elo_ratings(),
                    )
                    st.session_state["outlook_result"] = {
                        "division": division,
                        "group": group,
                        "simulation_input": simulation_input,
                    }

    stored = st.session_state.get("outlook_result")
    if stored is None:
        return
    if "error" in stored:
        st.warning(stored["error"])
        return

    simulation_input = stored["simulation_input"]
    st.header(f"Dywizja {stored['division']}, grupa {stored['group']}")

    standings = compute_standings(
        simulation_input.teams, simulation_input.played_matches, []
    )
    probabilities = simulate_group_qualification(
        simulation_input.teams,
        simulation_input.played_matches,
        simulation_input.remaining_matches,
        n_qualifying_spots=int(n_qualifying_spots),
        n_simulations=int(n_simulations),
    )

    with st.container(border=True):
        st.subheader("Aktualna tabela")
        standings_frame = pd.DataFrame(
            [
                {
                    "Drużyna": nominative_pl(standing.team),
                    "Pkt": standing.points,
                    "Bilans": standing.goal_difference,
                    "Strzelone": standing.goals_for,
                    "Stracone": standing.goals_against,
                    "W": standing.wins,
                    "R": standing.draws,
                    "P": standing.losses,
                }
                for standing in standings
            ]
        )
        st.dataframe(standings_frame, width="stretch", hide_index=True)
        st.caption(
            "⚠️ Uproszczenie: przy równej liczbie punktów tabela rozstrzyga "
            "kolejność wyłącznie przez bilans bramek i bramki strzelone. "
            "Regulaminy UEFA/FIFA uwzględniają najpierw bezpośrednie "
            "spotkania — w ściśle remisowych przypadkach kolejność może się różnić."
        )

    with st.container(border=True):
        st.subheader("Prawdopodobieństwo awansu")
        ranked = sorted(probabilities.items(), key=lambda item: -item[1])
        st.plotly_chart(
            _probability_chart(
                [nominative_pl(team) for team, _ in ranked],
                [value for _, value in ranked],
            ),
            width="stretch",
        )
        st.caption(
            f"Pozostałe mecze do rozegrania: {len(simulation_input.remaining_matches)}"
        )

    with st.container(border=True):
        # Far richer than the single qualification percentage above: it
        # separates "reliably second" from "either first or last", which the
        # one-number view collapses into the same figure.
        st.subheader("Gdzie kończy każda drużyna")
        positions = simulate_final_positions(
            simulation_input.teams,
            simulation_input.played_matches,
            simulation_input.remaining_matches,
            n_simulations=int(n_simulations),
        )
        ordered_teams = sorted(positions, key=lambda team: -positions[team][0])
        matrix = [positions[team] for team in ordered_teams]
        st.plotly_chart(
            _heatmap_figure(
                matrix=matrix,
                x_labels=[
                    f"{place}. miejsce" for place in range(1, len(ordered_teams) + 1)
                ],
                y_labels=[nominative_pl(team) for team in ordered_teams],
                cell_texts=[[f"{value:.0%}" for value in row] for row in matrix],
                height=90 + 46 * len(ordered_teams),
            ),
            width="stretch",
        )
        st.caption(
            "Każdy wiersz sumuje się do 100% — drużyna kończy "
            "dokładnie na jednym miejscu."
        )

    whatif_data = _render_what_if(
        simulation_input, int(n_qualifying_spots), int(n_simulations)
    )

    with st.container(border=True):
        st.subheader("Eksport")
        try:
            probability_labels = [nominative_pl(team) for team, _ in ranked]
            probability_values = [value for _, value in ranked]
            pdf_bytes = _build_outlook_pdf(
                division=stored["division"],
                group=stored["group"],
                season=season,
                standings_frame=standings_frame,
                probability_labels=probability_labels,
                probability_values=probability_values,
                position_team_labels=[nominative_pl(team) for team in ordered_teams],
                position_labels=[
                    f"{place}. miejsce" for place in range(1, len(ordered_teams) + 1)
                ],
                position_matrix=matrix,
                whatif=whatif_data,
            )
        except Exception:
            st.caption("PDF niedostępny")
        else:
            file_name = (
                f"{stored['division']}-{stored['group']}-{season}.pdf"
            ).replace(" ", "_")
            st.download_button(
                "Eksportuj do PDF",
                data=pdf_bytes,
                file_name=file_name,
                mime="application/pdf",
                width="stretch",
                key="outlook_pdf_export",
            )


def _render_what_if(
    simulation_input: SimulationInput, n_qualifying_spots: int, n_simulations: int
) -> dict[str, object] | None:
    """Fix one remaining match and show qualification odds before vs. after.

    Returns the same match/goals/comparison data just rendered on screen (or
    `None` when there are no remaining matches to build a scenario from) so
    `_render_competition_outlook_tab`'s PDF export can reuse it verbatim
    instead of recomputing the Monte Carlo scenario a second time -- and so
    the PDF only ever gets a "co jeśli" section under the exact same
    condition the screen does, never fabricated for a group with nothing
    left to play.
    """
    remaining = simulation_input.remaining_matches
    if not remaining:
        return None

    with st.container(border=True):
        st.subheader("Scenariusz „co jeśli”")
        st.caption(
            "Ustal wynik jednego nierozegranego meczu i zobacz, jak zmieniają "
            "się szanse awansu."
        )

        def _match_label(index: int) -> str:
            """Both sides of one remaining fixture, in Polish."""
            match = remaining[index]
            return (
                f"{nominative_pl(match.home_team)} — {nominative_pl(match.away_team)}"
            )

        match_index = st.selectbox(
            "Mecz",
            range(len(remaining)),
            format_func=_match_label,
            key="whatif_match",
        )
        fixed_match = remaining[match_index]
        column_home, column_away = st.columns(2)
        with column_home:
            home_goals = st.number_input(
                f"Bramki: {nominative_pl(fixed_match.home_team)}",
                min_value=0,
                max_value=10,
                value=1,
                key="whatif_home_goals",
            )
        with column_away:
            away_goals = st.number_input(
                f"Bramki: {nominative_pl(fixed_match.away_team)}",
                min_value=0,
                max_value=10,
                value=0,
                key="whatif_away_goals",
            )

        with st.spinner("Liczę scenariusz..."):
            baseline, scenario = what_if_fixed_result(
                simulation_input.teams,
                simulation_input.played_matches,
                remaining,
                n_qualifying_spots=n_qualifying_spots,
                fixed_match_index=int(match_index),
                fixed_home_goals=int(home_goals),
                fixed_away_goals=int(away_goals),
                n_simulations=n_simulations,
                # Same seed for both runs, so the only difference between them
                # is the fixed result rather than Monte Carlo noise.
                seed=20260803,
            )

        comparison = pd.DataFrame(
            [
                {
                    "Drużyna": nominative_pl(team),
                    "Przed": baseline.get(team, 0.0),
                    "Po": scenario.get(team, 0.0),
                    "Zmiana": scenario.get(team, 0.0) - baseline.get(team, 0.0),
                }
                for team in sorted(baseline, key=lambda name: -scenario.get(name, 0.0))
            ]
        )
        st.dataframe(
            comparison.style.format(
                {"Przed": "{:.0%}", "Po": "{:.0%}", "Zmiana": "{:+.1%}"}
            ),
            width="stretch",
            hide_index=True,
        )

    comparison_display = comparison.assign(
        Przed=comparison["Przed"].map(lambda value: f"{value:.0%}"),
        Po=comparison["Po"].map(lambda value: f"{value:.0%}"),
        Zmiana=comparison["Zmiana"].map(lambda value: f"{value:+.1%}"),
    )
    return {
        "match_label": _match_label(match_index),
        "home_label": nominative_pl(fixed_match.home_team),
        "home_goals": int(home_goals),
        "away_label": nominative_pl(fixed_match.away_team),
        "away_goals": int(away_goals),
        "comparison_frame": comparison_display,
    }


# --------------------------------------------------------------------------


def _render_header() -> None:
    """Centred brand lockup, one thin line of description, PDF export on the right."""
    title_column, export_column = st.columns([5, 1])
    with title_column:
        if _LOCKUP_DARK.exists():
            # Centred via columns rather than CSS: st.image has no alignment
            # option, and the surrounding page is a dark theme, so this is the
            # `_dark` variant of the lockup.
            left_pad, logo_column, right_pad = st.columns([1, 2, 1])
            with logo_column:
                st.image(str(_LOCKUP_DARK), width=280)
            _ = (left_pad, right_pad)
        else:
            st.markdown(
                f"<div style='text-align:center;font-size:2.6rem;font-weight:700'>"
                f"{_APP_NAME}</div>",
                unsafe_allow_html=True,
            )
        st.markdown(
            f"<div style='text-align:center;font-size:0.85rem;"
            f"color:{_MUTED_TEXT_COLOUR};margin-top:0.2rem'>"
            f"{_APP_TAGLINE}</div>",
            unsafe_allow_html=True,
        )
    with export_column:
        _render_pdf_export()
    st.markdown("")


def _render_pdf_export() -> None:
    """Offer the analysis currently on screen as an editorial-ready PDF.

    Rebuilt from `selected_fixture` rather than from a payload stashed by the
    analysis screen: the header renders *before* the tabs do, so a stashed
    payload is always one interaction stale and the button never shows up on
    the run that first opens an analysis. Everything needed here is already
    memoised, so recomputing costs nothing.
    """
    stored = st.session_state.get("selected_fixture")
    if stored is None:
        return
    home, away, match_date, tournament, venue = stored
    try:
        prediction = _cached_prediction(home, away, match_date, tournament)
        as_of = pd.Timestamp(match_date, tz="UTC")
        facts = [
            fact
            for fact in _cached_facts(home, away, as_of.isoformat(), 10)
            if fact.fact_type not in _STRUCTURAL_FACT_TYPES
        ]
        home_label = nominative_pl(home)
        away_label = nominative_pl(away)
        score_matrix, top_scorelines, _ = _score_grid_matrix(
            prediction.goals["home"], prediction.goals["away"]
        )
        pdf_bytes = _build_analysis_pdf(
            home_label=home_label,
            away_label=away_label,
            match_date=match_date,
            tournament=tournament,
            venue=venue,
            prediction=prediction,
            facts=facts,
            head_to_head_summary=_head_to_head_summary(home, away, as_of),
            score_matrix=score_matrix,
            top_scorelines=top_scorelines,
            home_form_rows=_team_form_table_rows(
                home, _cached_recent_form(home, as_of.isoformat())
            ),
            away_form_rows=_team_form_table_rows(
                away, _cached_recent_form(away, as_of.isoformat())
            ),
        )
    except Exception:
        st.caption("PDF niedostępny")
        return
    st.download_button(
        "Eksportuj do PDF",
        data=pdf_bytes,
        file_name=f"{home}-{away}-{match_date}.pdf".replace(" ", "_"),
        mime="application/pdf",
        width="stretch",
    )


def main() -> None:
    """Render the demo app."""
    _render_header()
    preview_tab, recap_tab, outlook_tab = st.tabs(
        ["Nadchodzące mecze", "Po meczu", "Perspektywy grupy"]
    )
    with preview_tab:
        _render_match_preview_tab()
    with recap_tab:
        _render_recap_tab()
    with outlook_tab:
        _render_competition_outlook_tab()


if __name__ == "__main__":
    main()
