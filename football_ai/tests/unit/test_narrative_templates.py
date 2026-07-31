"""Tests for football_ai/narrative/templates_pl.py.

Exact-string assertions against hand-built Facts -- this is Krok 4's own
version of a golden test: PLAN_SILNIKA.md's completion criterion is
"poprawne językowo zdania" (grammatically correct sentences), which no
generic test can verify automatically, so each expected string here has
been proofread by hand against real Polish grammar (numeral agreement,
verb conjugation by gender/number, case selection) -- if a template
changes, the diff against these strings is exactly what should be
re-proofread.
"""

from __future__ import annotations

import unittest

from football_ai.insights.schemas import Fact
from football_ai.narrative import templates_pl


def _fact(fact_type: str, subject_team: str, data: dict, **kwargs) -> Fact:
    """Build a minimal Fact for template testing -- provenance is irrelevant here, just non-empty."""
    return Fact(
        fact_type=fact_type,
        subject_team=subject_team,
        summary="irrelevant to template rendering",
        data=data,
        weight=0.5,
        provenance=(1,),
        **kwargs,
    )


class StreakTemplateTestCase(unittest.TestCase):
    """win/unbeaten/scoreless/clean_sheet streaks -- verb-gender agreement across team classes."""

    def test_win_streak_feminine_subject(self) -> None:
        fact = _fact("win_streak", "Poland", {"streak_length": 5, "venue": None})
        self.assertEqual(templates_pl.win_streak(fact), "Polska wygrała 5 meczów z rzędu.")

    def test_win_streak_plural_subject(self) -> None:
        """"Niemcy" (Germany) is grammatically plural in Polish -- verb must be plural too."""
        fact = _fact("win_streak", "Germany", {"streak_length": 3, "venue": None})
        self.assertEqual(templates_pl.win_streak(fact), "Niemcy wygrały 3 mecze z rzędu.")

    def test_win_streak_masculine_subject_with_venue(self) -> None:
        fact = _fact("win_streak", "Israel", {"streak_length": 1, "venue": "home"})
        self.assertEqual(templates_pl.win_streak(fact), "Izrael wygrał 1 mecz z rzędu u siebie.")

    def test_win_streak_teen_exception(self) -> None:
        """13 is in the 12-14 exception range -- "meczów", not the "few" form "mecze"."""
        fact = _fact("win_streak", "Poland", {"streak_length": 13, "venue": "away"})
        self.assertEqual(templates_pl.win_streak(fact), "Polska wygrała 13 meczów z rzędu na wyjeździe.")

    def test_unbeaten_streak(self) -> None:
        fact = _fact("unbeaten_streak", "Poland", {"streak_length": 2, "venue": None})
        self.assertEqual(templates_pl.unbeaten_streak(fact), "Polska rozegrała 2 mecze z rzędu bez porażki.")

    def test_scoreless_streak_singular(self) -> None:
        """n=1 uses genitive singular "meczu", not genitive plural "meczów"."""
        fact = _fact("scoreless_streak", "Poland", {"streak_length": 1, "venue": None})
        self.assertEqual(templates_pl.scoreless_streak(fact), "Polska nie strzeliła gola od 1 meczu.")

    def test_scoreless_streak_plural(self) -> None:
        fact = _fact("scoreless_streak", "Poland", {"streak_length": 4, "venue": None})
        self.assertEqual(templates_pl.scoreless_streak(fact), "Polska nie strzeliła gola od 4 meczów.")

    def test_clean_sheet_streak(self) -> None:
        fact = _fact("clean_sheet_streak", "Poland", {"streak_length": 2, "venue": "home"})
        self.assertEqual(
            templates_pl.clean_sheet_streak(fact), "Polska nie straciła gola od 2 meczów u siebie."
        )


class RecordTemplateTestCase(unittest.TestCase):
    """biggest_win, longest_*_ever -- opponent dative case, real historical example."""

    def test_biggest_win_matches_polands_real_record(self) -> None:
        """Poland's real biggest win on record: 10-0 vs San Marino, 2009-04-01."""
        fact = _fact(
            "biggest_win",
            "Poland",
            {"opponent": "San Marino", "goals_for": 10, "goals_against": 0, "margin": 10, "date": "2009-04-01"},
            related_teams=("San Marino",),
        )
        self.assertEqual(
            templates_pl.biggest_win(fact),
            "Polska zanotowała swoje największe zwycięstwo w historii: 10:0 przeciwko San Marino (2009-04-01).",
        )

    def test_longest_win_streak_ever_genitive_plural(self) -> None:
        fact = _fact("longest_win_streak_ever", "Poland", {"streak_length": 7})
        self.assertEqual(
            templates_pl.longest_win_streak_ever(fact),
            "Polska zanotowała najdłuższą w historii serię 7 zwycięstw z rzędu.",
        )

    def test_longest_unbeaten_streak_ever_few_form(self) -> None:
        fact = _fact("longest_unbeaten_streak_ever", "Poland", {"streak_length": 13})
        self.assertEqual(
            templates_pl.longest_unbeaten_streak_ever(fact),
            "Polska zanotowała najdłuższą w historii serię 13 meczów bez porażki.",
        )


class MilestoneTemplateTestCase(unittest.TestCase):
    """match_count_milestone, goal_count_milestone -- real historical example."""

    def test_match_count_milestone_matches_polands_real_800th(self) -> None:
        """Poland's real 800th international was 2018-06-12 vs Lithuania."""
        fact = _fact(
            "match_count_milestone",
            "Poland",
            {"milestone_number": 800, "opponent": "Lithuania"},
            as_of_match_id=1,
        )
        self.assertEqual(
            templates_pl.match_count_milestone(fact),
            "Polska rozegrała swój 800. mecz w historii reprezentacji, przeciwko Litwie.",
        )

    def test_goal_count_milestone(self) -> None:
        fact = _fact("goal_count_milestone", "Poland", {"milestone_number": 500, "opponent": "Hungary"})
        self.assertEqual(
            templates_pl.goal_count_milestone(fact),
            "Polska przekroczyła granicę 500 goli w historii w meczu przeciwko Węgrom.",
        )


class PlayerGoalMilestoneTemplateTestCase(unittest.TestCase):
    """Real historical example: Robert Lewandowski's real 50th goal for Poland, vs. Sweden."""

    def test_player_goal_milestone_matches_lewandowskis_real_50th(self) -> None:
        fact = _fact(
            "player_goal_milestone",
            "Poland",
            {"player_name": "Robert Lewandowski", "milestone_number": 50, "opponent": "Sweden"},
            related_teams=("Sweden",),
        )
        self.assertEqual(
            templates_pl.player_goal_milestone(fact),
            "Robert Lewandowski (Polska) zdobył swój 50. gol w kadrze narodowej, w meczu przeciwko Szwecji.",
        )


class HeadToHeadTemplateTestCase(unittest.TestCase):
    """h2h_never_won, h2h_first_win_since -- accusative case, singular/plural years."""

    def test_h2h_never_won_singular_meeting(self) -> None:
        fact = _fact(
            "h2h_never_won", "Poland", {"opponent": "Croatia", "meeting_count": 1}, related_teams=("Croatia",)
        )
        self.assertEqual(
            templates_pl.h2h_never_won(fact),
            "Polska rozegrała 1 spotkanie przeciwko Chorwacji bez ani jednego zwycięstwa.",
        )

    def test_h2h_never_won_few_form(self) -> None:
        fact = _fact(
            "h2h_never_won", "Poland", {"opponent": "Croatia", "meeting_count": 3}, related_teams=("Croatia",)
        )
        self.assertEqual(
            templates_pl.h2h_never_won(fact),
            "Polska rozegrała 3 spotkania przeciwko Chorwacji bez ani jednego zwycięstwa.",
        )

    def test_h2h_first_win_since_with_gap(self) -> None:
        """Poland's real 2023 win over Germany, first since 2014 -- a 9-year gap."""
        fact = _fact(
            "h2h_first_win_since",
            "Poland",
            {"opponent": "Germany", "win_year": 2023, "gap_years": 9},
            related_teams=("Germany",),
        )
        self.assertEqual(
            templates_pl.h2h_first_win_since(fact),
            "Polska pokonała Niemcy w 2023 roku, pierwszy raz od 9 lat.",
        )

    def test_h2h_first_win_since_one_year_gap_uses_singular_genitive(self) -> None:
        fact = _fact(
            "h2h_first_win_since",
            "Poland",
            {"opponent": "Germany", "win_year": 2023, "gap_years": 1},
            related_teams=("Germany",),
        )
        self.assertEqual(
            templates_pl.h2h_first_win_since(fact),
            "Polska pokonała Niemcy w 2023 roku, pierwszy raz od 1 roku.",
        )

    def test_h2h_first_win_since_no_prior_win_is_first_ever(self) -> None:
        fact = _fact(
            "h2h_first_win_since",
            "Poland",
            {"opponent": "Brazil", "win_year": 1974, "gap_years": None},
            related_teams=("Brazil",),
        )
        self.assertEqual(
            templates_pl.h2h_first_win_since(fact),
            "Polska pokonała Brazil w 1974 roku, pierwszy raz w historii.",
        )


if __name__ == "__main__":
    unittest.main()
