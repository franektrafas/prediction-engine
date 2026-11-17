"""Manual, non-Streamlit driver for `streamlit_app._build_outlook_pdf`.

Generates a real "Perspektywy grupy" PDF for one Nations League division/
group without going through the running app -- useful for verifying the
Round 6 PDF export (see `PROMPT_STREAMLIT_V6.md`) while a training run is
using the machine, or for a quick regression check after touching any of
the PDF-building code.

Run from the repo root, in the project's real venv (needs `fpdf2`,
`streamlit`, `pydantic`, `joblib` and the rest of `requirements.txt` --
this will NOT run in a lightweight `requirements-local.txt`-only
environment):

    python -m football_ai.scripts.generate_outlook_pdf_manual
    python -m football_ai.scripts.generate_outlook_pdf_manual --division B --group 4
    python -m football_ai.scripts.generate_outlook_pdf_manual --team Poland
    python -m football_ai.scripts.generate_outlook_pdf_manual --no-whatif

Defaults to Division B, Group 4 -- Poland's real 2026-27 group, per the
worked example already in `scrapers/sources/uefa_nations_league.py`'s
`fetch_group_matches` docstring (`("B", 4, "2026-27")`), which is almost
certainly what "grupa B ligi narodow" meant.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ruff: noqa: E402 -- import must follow the sys.path fix above.


def main() -> None:
    """Fetch one real group, run the simulation, and write a PDF to disk."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--division", default=None, help='e.g. "B" (League A/B/C/D)')
    parser.add_argument("--group", type=int, default=None, help="e.g. 4")
    parser.add_argument(
        "--team",
        default="Poland",
        help="Used to look up division/group only when --division/--group are omitted.",
    )
    parser.add_argument("--season", default="2026–27")
    parser.add_argument("--spots", type=int, default=1, help="Qualifying spots.")
    parser.add_argument(
        "--simulations", type=int, default=10_000, help="Monte Carlo draws."
    )
    parser.add_argument(
        "--whatif",
        dest="whatif",
        action="store_true",
        default=True,
        help="Include a 'co jesli' section (default).",
    )
    parser.add_argument(
        "--no-whatif",
        dest="whatif",
        action="store_false",
        help="Generate the PDF WITHOUT a 'co jesli' section, to check that path too.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output PDF path (default: <division>-<group>-<season>.pdf in cwd).",
    )
    args = parser.parse_args()

    # Imported here, not at module scope: these pull in streamlit, fpdf2,
    # pydantic, joblib and (transitively) the trained model artifacts --
    # only worth failing loudly on once argparse has already validated the
    # command line.
    from football_ai.narrative.team_names_pl import nominative_pl
    from football_ai.predictions.service import PredictionService
    from football_ai.scrapers.sources.elo import WorldFootballEloSource
    from football_ai.scrapers.sources.uefa_nations_league import UefaNationsLeagueSource
    from football_ai.simulation.competition import (
        compute_standings,
        simulate_final_positions,
        simulate_group_qualification,
    )
    from football_ai.simulation.fixture_input import build_simulation_input
    from football_ai.simulation.scenarios import what_if_fixed_result
    from football_ai.streamlit_app import _build_outlook_pdf

    uefa_source = UefaNationsLeagueSource()
    if args.division and args.group:
        division, group = args.division, args.group
    else:
        print(f"Looking up {args.team!r}'s division/group for {args.season}...")
        division, group = uefa_source.find_team_group(args.team, season=args.season)
    print(f"Division {division}, group {group}, season {args.season}")

    fixtures = uefa_source.fetch_group_matches(division, group, season=args.season)
    if fixtures.empty:
        raise SystemExit(f"No fixtures found for division {division}, group {group}.")

    prediction_service = PredictionService.create()
    try:
        elo_ratings = WorldFootballEloSource.latest_ratings_by_team(
            WorldFootballEloSource().fetch_current_snapshot()
        )
    except Exception as error:  # noqa: BLE001 -- same degrade-gracefully contract as the app
        print(f"Elo fetch failed ({error!r}); continuing with no live Elo.")
        elo_ratings = {}

    simulation_input = build_simulation_input(
        fixtures, prediction_service, elo_ratings=elo_ratings
    )

    standings = compute_standings(
        simulation_input.teams, simulation_input.played_matches, []
    )
    standings_frame_rows = [
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
    import pandas as pd

    standings_frame = pd.DataFrame(standings_frame_rows)

    probabilities = simulate_group_qualification(
        simulation_input.teams,
        simulation_input.played_matches,
        simulation_input.remaining_matches,
        n_qualifying_spots=args.spots,
        n_simulations=args.simulations,
    )
    ranked = sorted(probabilities.items(), key=lambda item: -item[1])
    probability_labels = [nominative_pl(team) for team, _ in ranked]
    probability_values = [value for _, value in ranked]

    positions = simulate_final_positions(
        simulation_input.teams,
        simulation_input.played_matches,
        simulation_input.remaining_matches,
        n_simulations=args.simulations,
    )
    ordered_teams = sorted(positions, key=lambda team: -positions[team][0])
    position_matrix = [positions[team] for team in ordered_teams]
    position_team_labels = [nominative_pl(team) for team in ordered_teams]
    position_labels = [f"{place}. miejsce" for place in range(1, len(ordered_teams) + 1)]

    whatif_data = None
    remaining = simulation_input.remaining_matches
    if args.whatif and remaining:
        fixed_match = remaining[0]
        baseline, scenario = what_if_fixed_result(
            simulation_input.teams,
            simulation_input.played_matches,
            remaining,
            n_qualifying_spots=args.spots,
            fixed_match_index=0,
            fixed_home_goals=1,
            fixed_away_goals=0,
            n_simulations=args.simulations,
            seed=20260803,
        )
        comparison = pd.DataFrame(
            [
                {
                    "Drużyna": nominative_pl(team),
                    "Przed": f"{baseline.get(team, 0.0):.0%}",
                    "Po": f"{scenario.get(team, 0.0):.0%}",
                    "Zmiana": f"{scenario.get(team, 0.0) - baseline.get(team, 0.0):+.1%}",
                }
                for team in sorted(baseline, key=lambda name: -scenario.get(name, 0.0))
            ]
        )
        whatif_data = {
            "match_label": (
                f"{nominative_pl(fixed_match.home_team)} — "
                f"{nominative_pl(fixed_match.away_team)}"
            ),
            "home_label": nominative_pl(fixed_match.home_team),
            "home_goals": 1,
            "away_label": nominative_pl(fixed_match.away_team),
            "away_goals": 0,
            "comparison_frame": comparison,
        }
        print(f"Included 'co jesli' section for: {whatif_data['match_label']}")
    elif args.whatif:
        print("No remaining matches -- 'co jesli' section skipped (nothing to fix).")
    else:
        print("'co jesli' section deliberately omitted (--no-whatif).")

    pdf_bytes = _build_outlook_pdf(
        division=division,
        group=group,
        season=args.season,
        standings_frame=standings_frame,
        probability_labels=probability_labels,
        probability_values=probability_values,
        position_team_labels=position_team_labels,
        position_labels=position_labels,
        position_matrix=position_matrix,
        whatif=whatif_data,
    )

    out_path = Path(
        args.out or f"{division}-{group}-{args.season}.pdf".replace(" ", "_")
    )
    out_path.write_bytes(pdf_bytes)
    print(f"Wrote {len(pdf_bytes)} bytes -> {out_path.resolve()}")


if __name__ == "__main__":
    main()
