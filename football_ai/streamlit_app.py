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

from datetime import date

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

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
from football_ai.scrapers.sources.uefa_nations_league import UefaNationsLeagueSource
from football_ai.simulation.competition import (
    compute_standings,
    simulate_group_qualification,
)
from football_ai.simulation.fixture_input import SimulationInput, build_simulation_input
from football_ai.simulation.scenarios import what_if_fixed_result

st.set_page_config(page_title="Football AI", page_icon="⚽", layout="wide")

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
    matches = streaks.team_matches(
        _load_records_frame(), team, pd.Timestamp(as_of_iso)
    )
    return matches.tail(limit)


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


def _style_figure(figure: go.Figure, height: int) -> go.Figure:
    """Strip Plotly's default chrome and match the app theme."""
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
    return figure


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
    """"Czwartek, 24 września" -- Polish weekday + genitive month, built by hand.

    Not `locale`/`babel`: neither is a dependency here and `setlocale` is
    process-global, so a Streamlit worker changing it would leak into every
    other thread's formatting.
    """
    month = _MONTHS_GENITIVE_PL[day.month - 1]
    return f"{_WEEKDAYS_PL[day.weekday()]}, {day.day} {month}"


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
                    "Data": pd.Timestamp(row["match_date"]).date(),
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


def _prediction_export_text(
    home_label: str,
    away_label: str,
    match_date: date,
    prediction: PredictionResponse,
    facts: list[Fact],
) -> str:
    """Ready-to-paste Markdown -- the product is sentences, so they must copy."""
    outcome = prediction.outcome_probabilities
    score = f"{prediction.exact_score['home']}:{prediction.exact_score['away']}"
    expected_goals = f"{prediction.xg['home']:.2f} — {prediction.xg['away']:.2f}"
    lines = [
        f"# {home_label} — {away_label} ({match_date})",
        "",
        "## Prognoza",
        f"- Wygrana {home_label}: {outcome.home_win:.0%}",
        f"- Remis: {outcome.draw:.0%}",
        f"- Wygrana {away_label}: {outcome.away_win:.0%}",
        f"- Przewidywany wynik: {score}",
        f"- Oczekiwane bramki (xG): {expected_goals}",
        f"- Obie drużyny strzelą: {prediction.btts:.0%}",
        f"- Powyżej 2.5 bramki: {prediction.over_under['over_2_5']:.0%}",
        "",
        "## Fakty historyczne",
    ]
    if facts:
        lines.extend(f"- {render_fact(fact)}" for fact in facts)
    else:
        lines.append("- (brak faktów historycznych dla tej pary)")
    return "\n".join(lines)


def _render_export(file_stem: str, text: str) -> None:
    """st.code has a built-in copy button; download covers the paste-into-CMS case."""
    with st.container(border=True):
        st.subheader("Tekst do publikacji")
        st.code(text, language="markdown")
        st.download_button(
            "Pobierz jako .md",
            data=text.encode("utf-8"),
            file_name=f"{file_stem}.md",
            mime="text/markdown",
        )


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


def _select_fixture(home: str, away: str, match_date: date, tournament: str) -> None:
    """Move to the analysis screen for one fixture."""
    st.session_state["selected_fixture"] = (home, away, match_date, tournament)
    st.rerun()


def _render_fixture_card(
    row: pd.Series, elo_ratings: dict[str, float], key: str
) -> None:
    """One fixture as a full-width, visibly clickable card."""
    home = str(row["home_team_name"])
    away = str(row["away_team_name"])
    with st.container(border=True):
        venue_column, teams_column, action_column = st.columns([2, 3, 2])
        with venue_column:
            # The Wikipedia fixture tables carry a date but no kick-off time
            # (verified: every timestamp is exactly 00:00), so this shows the
            # real venue rather than inventing an hour.
            st.caption(str(row.get("stadium") or "—"))
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
            if st.button("Analiza", key=key, width="stretch"):
                _select_fixture(
                    home,
                    away,
                    pd.Timestamp(row["match_date"]).date(),
                    "UEFA Nations League",
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
    """Screen 1: every scheduled match of one group, grouped by date."""
    # Streamlit drops a widget's session_state entry on any run where that
    # widget isn't rendered -- and the analysis screen renders none of these.
    # Without mirroring the choice into plain (non-widget) keys, coming back
    # from an analysis always snapped the list to the defaults instead of the
    # group the user was actually looking at.
    division = st.segmented_control(
        "Dywizja",
        ["A", "B", "C", "D"],
        default=st.session_state.get("division_choice", "B"),
        key="preview_division",
    )
    group = st.segmented_control(
        "Grupa",
        [1, 2, 3, 4],
        default=st.session_state.get("group_choice", 4),
        key="preview_group",
    )
    with st.expander("Ustawienia", expanded=False):
        season = st.text_input(
            "Sezon",
            value=st.session_state.get("season_choice", "2026–27"),
            key="preview_season",
        )
    if division is None or group is None:
        st.caption("Wybierz dywizję i grupę.")
        return
    st.session_state["division_choice"] = division
    st.session_state["group_choice"] = group
    st.session_state["season_choice"] = season

    with st.spinner("Pobieram terminarz..."):
        fixtures = _nations_league_fixtures(str(division), int(group), season)
    if fixtures.empty:
        st.warning(
            f"Nie udało się pobrać terminarza dla dywizji {division}, grupy {group} "
            f"({season}). Sprawdź sezon albo użyj „Dowolna para”."
        )
        _render_manual_pair_expander()
        return

    upcoming = fixtures.loc[fixtures["home_score"].isna()]
    if upcoming.empty:
        st.info(
            "W tej grupie wszystkie mecze zostały już rozegrane "
            "— zobacz zakładkę „Po meczu”."
        )
        _render_manual_pair_expander()
        return

    elo_ratings = _load_elo_ratings()
    fixtures_by_day = upcoming.groupby(upcoming["match_date"].dt.date, sort=True)
    for day, day_fixtures in fixtures_by_day:
        st.markdown(f"##### {_format_date_header_pl(day)}")
        for position, (_, row) in enumerate(day_fixtures.iterrows()):
            _render_fixture_card(row, elo_ratings, key=f"fixture_{day}_{position}")

    _render_manual_pair_expander()


def _render_match_preview_tab() -> None:
    """Master-detail: a group's fixture list, or one fixture's full analysis."""
    stored = st.session_state.get("selected_fixture")
    if stored is None:
        _render_fixture_list()
        return

    if st.button("← Wszystkie mecze grupy", key="preview_back"):
        # Only the selection is cleared -- division/group/season live under
        # their own widget keys, so the list comes back exactly as it was.
        del st.session_state["selected_fixture"]
        st.rerun()

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

    _render_structural_context(home, away, pd.Timestamp(match_date, tz="UTC"))

    with st.container(border=True):
        st.subheader("Prognoza wyniku")
        st.plotly_chart(
            _outcome_chart(
                home_label, away_label, prediction.outcome_probabilities.model_dump()
            ),
            width="stretch",
        )
        score_column, xg_column = st.columns(2)
        score_column.metric(
            "Przewidywany wynik",
            f"{prediction.exact_score['home']} : {prediction.exact_score['away']}",
        )
        xg_column.metric(
            "Oczekiwane bramki (xG)",
            f"{prediction.xg['home']:.2f} : {prediction.xg['away']:.2f}",
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
    trivia = [
        fact for fact in facts if fact.fact_type not in _STRUCTURAL_FACT_TYPES
    ]
    with st.container(border=True):
        st.subheader("Nietypowe fakty")
        st.caption(
            "Ciekawostki historyczne, nie kontekst meczu — uszeregowane wagą. "
            "Każde zdanie rozwija się do listy meczów, z których powstało."
        )
        _render_facts(trivia)

    _render_match_context(prediction)
    _render_export(
        f"{home}-{away}-{match_date}",
        _prediction_export_text(home_label, away_label, match_date, prediction, trivia),
    )
    _render_diagnostics(prediction)


# --------------------------------------------------------------------------
# Tab 2: post-match recap
# --------------------------------------------------------------------------


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

    tournament = st.selectbox("Rozgrywki", tournaments, key="recap_tournament")
    matches = _played_matches_in_tournament(str(tournament))
    if matches.empty:
        st.info("Brak rozegranych meczów w tych rozgrywkach.")
        return
    # Only real, already-played matches are offered, so the old "guess a date
    # and hope the match exists" failure mode can't happen any more.
    index = st.selectbox(
        "Mecz",
        range(len(matches)),
        format_func=lambda position: _format_fixture(matches.iloc[position]),
        key="recap_match",
    )

    if st.button("Pokaż podsumowanie", type="primary", key="recap_run"):
        st.session_state["recap_result"] = int(matches.iloc[int(index)]["match_id"])

    stored_match_id = st.session_state.get("recap_result")
    if stored_match_id is None:
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

    with st.container(border=True):
        st.subheader("Fakty historyczne")
        st.caption(
            "Stan na moment tuż po tym meczu — z faktami, które stworzył on sam."
        )
        _render_facts(facts)

    export_lines = [
        f"# {home_label} — {away_label} ({match_date})",
        "",
        f"Wynik: {int(result_row['home_score'])}:{int(result_row['away_score'])}",
        "",
        "## Fakty historyczne",
    ]
    if facts:
        export_lines.extend(f"- {render_fact(fact)}" for fact in facts)
    else:
        export_lines.append("- (brak faktów historycznych dla tej pary)")
    _render_export(f"{home}-{away}-{match_date}-recap", "\n".join(export_lines))


# --------------------------------------------------------------------------
# Tab 3: competition outlook + what-if scenarios
# --------------------------------------------------------------------------


def _render_competition_outlook_tab() -> None:
    """Real remaining Nations League fixtures + predictions, fed to Monte Carlo."""
    teams = _predictable_team_names()
    column_team, column_season = st.columns(2)
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
    with column_season:
        season = st.text_input("Sezon", value="2026–27", key="outlook_season")
    column_spots, column_simulations = st.columns(2)
    with column_spots:
        n_qualifying_spots = st.number_input(
            "Liczba awansujących miejsc",
            min_value=1,
            max_value=4,
            value=1,
            key="outlook_spots",
        )
    with column_simulations:
        n_simulations = st.number_input(
            "Liczba symulacji Monte Carlo",
            min_value=100,
            max_value=50_000,
            value=10_000,
            step=1000,
            key="outlook_simulations",
        )

    if st.button("Symuluj grupę", type="primary", key="outlook_run"):
        uefa_source = UefaNationsLeagueSource()
        try:
            with st.spinner("Szukam grupy drużyny..."):
                division, group = uefa_source.find_team_group(team_query, season=season)
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
                f"{nominative_pl(match.home_team)} — "
                f"{nominative_pl(match.away_team)}"
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


def main() -> None:
    """Render the demo app."""
    st.title("Football AI")
    st.caption(
        "Predykcje meczów reprezentacji narodowych z modeli uczonych na "
        "realnych danych historycznych — każdy fakt z możliwością "
        "sprawdzenia meczów źródłowych."
    )
    preview_tab, recap_tab, outlook_tab = st.tabs(
        ["Przed meczem", "Po meczu", "Perspektywy grupy"]
    )
    with preview_tab:
        _render_match_preview_tab()
    with recap_tab:
        _render_recap_tab()
    with outlook_tab:
        _render_competition_outlook_tab()


if __name__ == "__main__":
    main()
