"""Streamlit demo -- for marketing/cold-email outreach.

Direct in-process calls to PredictionService/insights/simulation/narrative,
the exact same building blocks `api/routes/matches.py` and
`api/routes/competitions.py` call over HTTP -- no FastAPI, no database. Run
with: `streamlit run football_ai/streamlit_app.py`.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from football_ai.config.settings import get_settings
from football_ai.data.io import read_table
from football_ai.data.normalization import canonicalize_team_name
from football_ai.insights.detectors import facts_for_match
from football_ai.narrative.renderer import render_fact
from football_ai.predictions.schemas import MatchContextPayload
from football_ai.predictions.service import PredictionService
from football_ai.scrapers.sources.elo import WorldFootballEloSource
from football_ai.scrapers.sources.uefa_nations_league import UefaNationsLeagueSource
from football_ai.simulation.competition import compute_standings, simulate_group_qualification
from football_ai.simulation.fixture_input import build_simulation_input


st.set_page_config(page_title="Football AI -- Demo", page_icon="⚽", layout="wide")

_RECORDS_FRAME_COLUMNS = [
    "match_id", "match_date", "home_team_name", "away_team_name", "home_score", "away_score", "tournament",
]
_GOALSCORERS_FRAME_COLUMNS = [
    "match_id", "match_date", "home_team_name", "away_team_name", "team_name", "scorer", "own_goal",
]


@st.cache_resource
def _load_prediction_service() -> PredictionService:
    """Load once per server process -- reads local model artifacts, no network."""
    return PredictionService.create()


@st.cache_data(ttl=3600)
def _load_elo_ratings() -> dict[str, float]:
    """Live Elo, cached for an hour -- same graceful-degradation contract as the API routes."""
    try:
        source = WorldFootballEloSource()
        snapshot = source.fetch_current_snapshot()
        return WorldFootballEloSource.latest_ratings_by_team(snapshot)
    except Exception:
        return {}


@st.cache_data
def _load_records_frame() -> pd.DataFrame:
    """Same degrade-to-empty-but-correctly-shaped contract as api/routes/matches.py."""
    path = get_settings().records_frame_path
    if not path.exists():
        return pd.DataFrame(columns=_RECORDS_FRAME_COLUMNS)
    return read_table(path)


@st.cache_data
def _load_goalscorers_frame() -> pd.DataFrame:
    """Same degrade-to-empty-but-correctly-shaped contract as api/routes/matches.py."""
    path = get_settings().goalscorers_frame_path
    if not path.exists():
        return pd.DataFrame(columns=_GOALSCORERS_FRAME_COLUMNS)
    return read_table(path)


@st.cache_data
def _known_team_names() -> list[str]:
    """Real team names from the records frame, for a searchable dropdown instead of free text."""
    frame = _load_records_frame()
    if frame.empty:
        return []
    teams = pd.concat([frame["home_team_name"], frame["away_team_name"]]).dropna().unique()
    return sorted(str(team) for team in teams)


def _probability_bar_chart(labels: list[str], values: list[float]) -> go.Figure:
    """A horizontal percentage bar chart, shared by both tabs."""
    figure = go.Figure(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            text=[f"{value:.0%}" for value in values],
            textposition="auto",
            marker_color="#2b6cb0",
        )
    )
    figure.update_layout(
        xaxis_tickformat=".0%",
        xaxis_range=[0, 1],
        height=90 + 45 * len(labels),
        margin=dict(l=10, r=10, t=10, b=10),
    )
    return figure


def _render_match_preview_tab() -> None:
    """Pick two teams + a date, get a real prediction plus historical facts."""
    teams = _known_team_names()
    if not teams:
        st.warning(
            "Brak lokalnych danych historycznych (records_frame). Uruchom najpierw "
            "`python -m football_ai.scrapers.sources.international_results`."
        )
        return

    column_home, column_away, column_date = st.columns(3)
    with column_home:
        home = st.selectbox("Gospodarz", teams, index=teams.index("Poland") if "Poland" in teams else 0)
    with column_away:
        away_options = [team for team in teams if team != home]
        default_away_index = away_options.index("Sweden") if "Sweden" in away_options else 0
        away = st.selectbox("Gość", away_options, index=default_away_index)
    with column_date:
        match_date = st.date_input("Data meczu", value=date.today())
    tournament = st.text_input("Turniej", value="UEFA Nations League")

    if not st.button("Przewidź mecz", type="primary"):
        return

    service = _load_prediction_service()
    elo_ratings = _load_elo_ratings()
    payload = MatchContextPayload(
        match_date=pd.Timestamp(match_date, tz="UTC"),
        home_team_name=home,
        away_team_name=away,
        tournament=tournament,
        home_elo=elo_ratings.get(canonicalize_team_name(home)),
        away_elo=elo_ratings.get(canonicalize_team_name(away)),
    )
    prediction = service.predict(payload)

    st.subheader(f"{home} vs {away}")
    outcome = prediction.outcome_probabilities
    st.plotly_chart(
        _probability_bar_chart(
            [f"{away} wygrywa", "Remis", f"{home} wygrywa"],
            [outcome.away_win, outcome.draw, outcome.home_win],
        ),
        width="stretch",
    )

    metric_columns = st.columns(4)
    metric_columns[0].metric(
        "Przewidywany wynik", f"{prediction.exact_score['home']} : {prediction.exact_score['away']}"
    )
    metric_columns[1].metric(
        "Oczekiwane bramki (xG)", f"{prediction.xg['home']:.2f} : {prediction.xg['away']:.2f}"
    )
    metric_columns[2].metric("Obie drużyny strzelą (BTTS)", f"{prediction.btts:.0%}")
    metric_columns[3].metric("Model", prediction.model_version)

    st.markdown("**Powyżej / poniżej liczby bramek**")
    over_under_columns = st.columns(len(prediction.over_under))
    for column, (label, value) in zip(over_under_columns, sorted(prediction.over_under.items())):
        threshold = label.replace("over_", "").replace("_", ".")
        column.metric(f"Powyżej {threshold}", f"{value:.0%}")

    st.markdown("**Statystyki meczowe**")
    stats_table = pd.DataFrame(
        {
            home: [
                prediction.corners["home"],
                prediction.cards["home"],
                prediction.shots["home"],
                prediction.shots_on_target["home"],
                prediction.possession["home"],
            ],
            away: [
                prediction.corners["away"],
                prediction.cards["away"],
                prediction.shots["away"],
                prediction.shots_on_target["away"],
                prediction.possession["away"],
            ],
        },
        index=["Rożne", "Kartki żółte", "Strzały", "Strzały celne", "Posiadanie piłki (%)"],
    )
    st.dataframe(stats_table.style.format("{:.1f}"), width="stretch")

    st.markdown("**Fakty historyczne**")
    facts = facts_for_match(
        _load_records_frame(),
        home,
        away,
        as_of_date=pd.Timestamp(match_date, tz="UTC"),
        top_n=10,
        goalscorers_frame=_load_goalscorers_frame(),
    )
    if not facts:
        st.info("Brak istotnych faktów historycznych dla tego meczu.")
    for fact in facts:
        st.markdown(f"- {render_fact(fact)}")


def _render_competition_outlook_tab() -> None:
    """Real remaining UEFA Nations League fixtures + model predictions, fed into a Monte Carlo sim."""
    column_team, column_season = st.columns(2)
    with column_team:
        team_query = st.text_input("Drużyna", value="Poland")
    with column_season:
        season = st.text_input("Sezon", value="2026–27")
    column_spots, column_sims = st.columns(2)
    with column_spots:
        n_qualifying_spots = st.number_input("Liczba awansujących miejsc", min_value=1, max_value=4, value=1)
    with column_sims:
        n_simulations = st.number_input(
            "Liczba symulacji Monte Carlo", min_value=100, max_value=50_000, value=10_000, step=1000
        )

    if not st.button("Symuluj grupę", type="primary"):
        return

    uefa_source = UefaNationsLeagueSource()
    try:
        with st.spinner("Szukam grupy drużyny..."):
            division, group = uefa_source.find_team_group(team_query, season=season)
    except ValueError as error:
        st.error(str(error))
        return
    st.success(f"{team_query}: Dywizja {division}, grupa {group}")

    fixtures = uefa_source.fetch_group_matches(division, group, season=season)
    if fixtures.empty:
        st.warning("Brak meczów w tej grupie.")
        return

    service = _load_prediction_service()
    elo_ratings = _load_elo_ratings()
    simulation_input = build_simulation_input(fixtures, service, elo_ratings=elo_ratings)
    standings = compute_standings(simulation_input.teams, simulation_input.played_matches, [])
    probabilities = simulate_group_qualification(
        simulation_input.teams,
        simulation_input.played_matches,
        simulation_input.remaining_matches,
        n_qualifying_spots=int(n_qualifying_spots),
        n_simulations=int(n_simulations),
    )

    st.markdown("**Aktualna tabela grupy**")
    standings_frame = pd.DataFrame(
        [
            {
                "Drużyna": standing.team,
                "Pkt": standing.points,
                "Bilans bramek": standing.goal_difference,
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

    st.markdown("**Prawdopodobieństwo awansu**")
    ranked_probabilities = sorted(probabilities.items(), key=lambda item: -item[1])
    st.plotly_chart(
        _probability_bar_chart([team for team, _ in ranked_probabilities], [value for _, value in ranked_probabilities]),
        width="stretch",
    )
    st.caption(f"Pozostałe mecze do rozegrania: {len(simulation_input.remaining_matches)}")


def main() -> None:
    """Render the demo app."""
    st.title("⚽ Football AI — Silnik Predykcyjny (Demo)")
    st.caption(
        "Predykcje meczów reprezentacji narodowych z modeli ML trenowanych na realnych "
        "danych historycznych -- 1X2, bramki, kartki, rożne, over/under, BTTS, plus fakty "
        "historyczne i symulacje szans na awans."
    )
    preview_tab, outlook_tab = st.tabs(["Podgląd meczu", "Perspektywy Ligi Narodów"])
    with preview_tab:
        _render_match_preview_tab()
    with outlook_tab:
        _render_competition_outlook_tab()


if __name__ == "__main__":
    main()
