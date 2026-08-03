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
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import matplotlib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

from football_ai.config.settings import get_settings
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
from football_ai.scrapers.sources.uefa_nations_league import UefaNationsLeagueSource
from football_ai.simulation.competition import (
    compute_standings,
    simulate_final_positions,
    simulate_group_qualification,
)
from football_ai.simulation.fixture_input import SimulationInput, build_simulation_input
from football_ai.simulation.scenarios import what_if_fixed_result

_APP_NAME = "StatsDesk"
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
    return figure, top_scorelines, 1.0 - inside_mass


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


def _form_markers_html(letters: list[str]) -> str:
    """A row of W/R/P squares, oldest on the left."""
    markers = []
    for letter in letters:
        background, foreground = _RESULT_MARKER_STYLES[letter]
        markers.append(
            f"<span style='display:inline-flex;align-items:center;"
            f"justify-content:center;width:22px;height:22px;margin-right:4px;"
            f"border-radius:4px;background:{background};color:{foreground};"
            f"font-weight:700;font-size:12px;font-family:sans-serif'>{letter}</span>"
        )
    return f"<div style='display:flex;flex-wrap:wrap'>{''.join(markers)}</div>"


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


def _render_team_form(team: str, as_of: pd.Timestamp) -> None:
    """Last ten results as W/R/P markers, with the full table one expander away."""
    label = nominative_pl(team)
    form = _cached_recent_form(team, as_of.isoformat())
    st.markdown(f"**{label}**")
    if form.empty:
        st.caption("Brak meczów w zbiorze historycznym przed tą datą.")
        return

    letters = [
        _result_letter(row["goals_for"], row["goals_against"])
        for _, row in form.iterrows()
    ]
    st.markdown(_form_markers_html(letters), unsafe_allow_html=True)
    # A team with fewer than ten matches shows however many it really has --
    # never padded to look like a full row.
    if len(letters) < 10:
        matches_word = _plural_pl(len(letters), "mecz", "mecze", "meczów")
        st.caption(f"Tylko {len(letters)} {matches_word} w historii przed tą datą.")

    streak_sentences = _cached_current_streaks(team, as_of.isoformat())
    st.caption(
        " ".join(streak_sentences) if streak_sentences else "Brak aktualnej serii."
    )

    with st.expander(f"Pełna forma: {label}", expanded=False):
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Data": pd.Timestamp(row["match_date"]).date(),
                        "Przeciwnik": nominative_pl(str(row["opponent"])),
                        "Wynik": f"{int(row['goals_for'])}:{int(row['goals_against'])}",
                        "Gdzie": "dom" if bool(row["is_home"]) else "wyjazd",
                    }
                    for _, row in form.iloc[::-1].iterrows()
                ]
            ),
            width="stretch",
            hide_index=True,
        )


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


def _build_analysis_pdf(
    home_label: str,
    away_label: str,
    match_date: date,
    tournament: str,
    prediction: PredictionResponse,
    facts: list[Fact],
    head_to_head_summary: str,
    top_scorelines: list[tuple[str, float]],
) -> bytes:
    """One-page PDF of the analysis on screen, for sending on to an editor."""
    from fpdf import FPDF

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

    if _LOCKUP_LIGHT.exists():
        # The PDF page is white, so this is the `_light` variant -- the dark
        # one would be near-invisible on it.
        document.image(str(_LOCKUP_LIGHT), x=document.l_margin, y=10, w=52)
        document.set_y(28)
    else:
        write(_APP_NAME, size=9, style="B", gap=5)
    write(f"{home_label} — {away_label}", size=17, style="B", gap=9)
    write(f"{match_date} · {tournament}", size=9)
    document.ln(2)

    outcome = prediction.outcome_probabilities
    write("Prognoza", size=12, style="B", gap=7)
    write(
        f"Wygrana {home_label}: {outcome.home_win:.0%}    "
        f"Remis: {outcome.draw:.0%}    "
        f"Wygrana {away_label}: {outcome.away_win:.0%}"
    )
    write(
        f"Oczekiwane bramki (xG): {prediction.xg['home']:.2f} — "
        f"{prediction.xg['away']:.2f}    "
        f"Obie drużyny strzelą: {prediction.btts:.0%}    "
        f"Powyżej 2.5: {prediction.over_under['over_2_5']:.0%}"
    )
    if top_scorelines:
        likeliest = "    ".join(
            f"{score} ({value:.0%})" for score, value in top_scorelines
        )
        write(f"Najbardziej prawdopodobne wyniki: {likeliest}")
    document.ln(2)

    write("Bilans bezpośredni", size=12, style="B", gap=7)
    write(head_to_head_summary)
    document.ln(2)

    write("Statystyki meczowe (prognoza)", size=12, style="B", gap=7)
    for label, home_value, away_value in (
        ("Rzuty rożne", prediction.corners["home"], prediction.corners["away"]),
        ("Żółte kartki", prediction.cards["home"], prediction.cards["away"]),
        ("Strzały", prediction.shots["home"], prediction.shots["away"]),
        (
            "Strzały celne",
            prediction.shots_on_target["home"],
            prediction.shots_on_target["away"],
        ),
        (
            "Posiadanie (%)",
            prediction.possession["home"],
            prediction.possession["away"],
        ),
    ):
        write(f"{label}: {home_value:.1f} — {away_value:.1f}")
    document.ln(2)

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


def _select_fixture(home: str, away: str, match_date: date, tournament: str) -> None:
    """Move to the analysis screen for one fixture.

    Used as an `on_click` callback, never called inline: a callback runs
    *before* the script re-executes, so the new screen renders on the very
    next pass without an explicit `st.rerun()`. That matters because
    `st.rerun()` resets `st.tabs` to the first tab -- clicking a match in the
    "Po meczu" tab used to bounce the user back to "Nadchodzące mecze".
    """
    st.session_state["selected_fixture"] = (home, away, match_date, tournament)


def _fixture_left_label(row: pd.Series) -> str:
    """Kick-off time when the source really has one, otherwise the venue.

    Wikipedia's Nations League tables carry a date with no time (verified:
    every timestamp is exactly 00:00), so showing "00:00" there would be an
    invented kick-off. football_api rows do carry a real time.
    """
    timestamp = pd.Timestamp(row["match_date"])
    if timestamp.hour or timestamp.minute:
        return timestamp.strftime("%H:%M")
    venue = row.get("stadium") if "stadium" in row.index else None
    if not venue and "stadium_name" in row.index:
        venue = row.get("stadium_name")
    return str(venue) if venue and not pd.isna(venue) else "—"


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
                args=(home, away, pd.Timestamp(row["match_date"]).date(), tournament),
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

    home, away, match_date, tournament = stored

    with st.spinner("Liczę predykcję..."):
        prediction = _cached_prediction(home, away, match_date, tournament)
        facts = _cached_facts(
            home, away, pd.Timestamp(match_date, tz="UTC").isoformat(), 10
        )

    home_label = nominative_pl(home)
    away_label = nominative_pl(away)

    st.header(f"{home_label} — {away_label}")
    st.caption(f"{match_date} · {tournament}")

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
        st.plotly_chart(
            _team_comparison_chart(
                home_label,
                away_label,
                [
                    (
                        "Rzuty rożne",
                        prediction.corners["home"],
                        prediction.corners["away"],
                    ),
                    (
                        "Żółte kartki",
                        prediction.cards["home"],
                        prediction.cards["away"],
                    ),
                    ("Strzały", prediction.shots["home"], prediction.shots["away"]),
                    (
                        "Strzały celne",
                        prediction.shots_on_target["home"],
                        prediction.shots_on_target["away"],
                    ),
                    (
                        "Posiadanie (%)",
                        prediction.possession["home"],
                        prediction.possession["away"],
                    ),
                ],
            ),
            width="stretch",
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

    _render_what_if(simulation_input, int(n_qualifying_spots), int(n_simulations))


def _render_what_if(
    simulation_input: SimulationInput, n_qualifying_spots: int, n_simulations: int
) -> None:
    """Fix one remaining match and show qualification odds before vs. after."""
    remaining = simulation_input.remaining_matches
    if not remaining:
        return

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
    """Offer the analysis currently on screen as a one-page PDF.

    Rebuilt from `selected_fixture` rather than from a payload stashed by the
    analysis screen: the header renders *before* the tabs do, so a stashed
    payload is always one interaction stale and the button never shows up on
    the run that first opens an analysis. Everything needed here is already
    memoised, so recomputing costs nothing.
    """
    stored = st.session_state.get("selected_fixture")
    if stored is None:
        return
    home, away, match_date, tournament = stored
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
        _, top_scorelines, _ = _score_grid_figure(
            home_label, away_label, prediction.goals["home"], prediction.goals["away"]
        )
        pdf_bytes = _build_analysis_pdf(
            home_label=home_label,
            away_label=away_label,
            match_date=match_date,
            tournament=tournament,
            prediction=prediction,
            facts=facts,
            head_to_head_summary=_head_to_head_summary(home, away, as_of),
            top_scorelines=top_scorelines,
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
