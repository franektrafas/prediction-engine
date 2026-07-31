"""Deterministic Polish sentence templates, one function per `insights.Fact.fact_type`.

Every template reads only from `Fact.data` (never `Fact.summary`, which is
English and explicitly documented as "not the polished, template-rendered
narrative sentence Krok 4 produces" -- see `insights.schemas.Fact`), so a
rendered sentence is exactly as reproducible/verifiable as the `Fact` it
came from. Deliberately no LLM anywhere in this module.

Grammar helpers here handle exactly what these 11 templates need and no
more: 3rd-person past-tense verb agreement (`_past_tense`), Polish's 3-way
nominative/accusative numeral agreement (`_count_form`: 1 / 2-4 / 5+), and
2-way genitive agreement for "od N X" constructions (`_genitive_count`:
singular / plural, where the plural form is always identical to
`_count_form`'s "many" form -- that's not a coincidence, Polish numerals
5+ grammatically require a genitive-plural noun, which is exactly what
the "many" form is). Verbs were deliberately chosen to avoid Polish's
ó/o stem alternation (e.g. "odniósł"/"odniosła") -- a plain
`stem + suffix` append is NOT safe for verbs like "odnieść"/"nieść", so
templates use "zanotował" instead of "odniósł" for "achieved/recorded",
where suffix-appending is safe for every gender class.
"""

from __future__ import annotations

from football_ai.insights.schemas import Fact
from football_ai.narrative.team_names_pl import accusative_pl, dative_pl, gender_pl, nominative_pl


_PAST_TENSE_SUFFIXES = {"m": "", "f": "a", "n": "o", "pl": "y"}


def _past_tense(masculine_stem: str, gender: str) -> str:
    """3rd-person past-tense form of a Polish verb for a team's grammatical gender/number."""
    return masculine_stem + _PAST_TENSE_SUFFIXES[gender]


def _count_form(n: int, singular: str, few: str, many: str) -> str:
    """Polish nominative/accusative count agreement: 1 -> singular, 2-4 -> few, else -> many.

    `many` is used for 0, 5+, and any count ending in 11-19 (e.g. 15 like
    5, not like "few") -- standard Polish numeral-noun agreement.
    """
    if n == 1:
        return singular
    last_two_digits = n % 100
    last_digit = n % 10
    if 12 <= last_two_digits <= 14:
        return many
    if 2 <= last_digit <= 4:
        return few
    return many


def _genitive_count(n: int, singular_genitive: str, many_form: str) -> str:
    """Polish genitive count agreement for "od N X"/"przez N X": 1 -> genitive singular, else -> many-form."""
    return singular_genitive if n == 1 else many_form


_VENUE_CLAUSES = {"home": " u siebie", "away": " na wyjeździe"}


def win_streak(fact: Fact) -> str:
    """"[Team] won N matches in a row [(home/away)]."""
    nom = nominative_pl(fact.subject_team)
    gender = gender_pl(fact.subject_team)
    n = fact.data["streak_length"]
    verb = _past_tense("wygrał", gender)
    noun = _count_form(n, "mecz", "mecze", "meczów")
    venue_clause = _VENUE_CLAUSES.get(fact.data.get("venue"), "")
    return f"{nom} {verb} {n} {noun} z rzędu{venue_clause}."


def unbeaten_streak(fact: Fact) -> str:
    """"[Team] played N matches in a row without a loss [(home/away)]."""
    nom = nominative_pl(fact.subject_team)
    gender = gender_pl(fact.subject_team)
    n = fact.data["streak_length"]
    verb = _past_tense("rozegrał", gender)
    noun = _count_form(n, "mecz", "mecze", "meczów")
    venue_clause = _VENUE_CLAUSES.get(fact.data.get("venue"), "")
    return f"{nom} {verb} {n} {noun} z rzędu bez porażki{venue_clause}."


def scoreless_streak(fact: Fact) -> str:
    """"[Team] hasn't scored in N matches [(home/away)]."""
    nom = nominative_pl(fact.subject_team)
    gender = gender_pl(fact.subject_team)
    n = fact.data["streak_length"]
    verb = _past_tense("strzelił", gender)
    noun = _genitive_count(n, "meczu", "meczów")
    venue_clause = _VENUE_CLAUSES.get(fact.data.get("venue"), "")
    return f"{nom} nie {verb} gola od {n} {noun}{venue_clause}."


def clean_sheet_streak(fact: Fact) -> str:
    """"[Team] hasn't conceded in N matches [(home/away)]."""
    nom = nominative_pl(fact.subject_team)
    gender = gender_pl(fact.subject_team)
    n = fact.data["streak_length"]
    verb = _past_tense("stracił", gender)
    noun = _genitive_count(n, "meczu", "meczów")
    venue_clause = _VENUE_CLAUSES.get(fact.data.get("venue"), "")
    return f"{nom} nie {verb} gola od {n} {noun}{venue_clause}."


def biggest_win(fact: Fact) -> str:
    """"[Team]'s biggest ever win was X-Y against [opponent] (date)."""
    nom = nominative_pl(fact.subject_team)
    gender = gender_pl(fact.subject_team)
    opponent_dative = dative_pl(fact.data["opponent"])
    verb = _past_tense("zanotował", gender)
    goals_for = fact.data["goals_for"]
    goals_against = fact.data["goals_against"]
    date = fact.data["date"]
    return (
        f"{nom} {verb} swoje największe zwycięstwo w historii: "
        f"{goals_for}:{goals_against} przeciwko {opponent_dative} ({date})."
    )


def longest_win_streak_ever(fact: Fact) -> str:
    """"[Team]'s longest ever winning run is N consecutive matches."""
    nom = nominative_pl(fact.subject_team)
    gender = gender_pl(fact.subject_team)
    n = fact.data["streak_length"]
    verb = _past_tense("zanotował", gender)
    noun = _count_form(n, "zwycięstwo", "zwycięstwa", "zwycięstw")
    return f"{nom} {verb} najdłuższą w historii serię {n} {noun} z rzędu."


def longest_unbeaten_streak_ever(fact: Fact) -> str:
    """"[Team]'s longest ever unbeaten run is N consecutive matches."""
    nom = nominative_pl(fact.subject_team)
    gender = gender_pl(fact.subject_team)
    n = fact.data["streak_length"]
    verb = _past_tense("zanotował", gender)
    noun = _count_form(n, "mecz", "mecze", "meczów")
    return f"{nom} {verb} najdłuższą w historii serię {n} {noun} bez porażki."


def match_count_milestone(fact: Fact) -> str:
    """"[Team]'s Nth international match was against [opponent]."""
    nom = nominative_pl(fact.subject_team)
    gender = gender_pl(fact.subject_team)
    n = fact.data["milestone_number"]
    opponent_dative = dative_pl(fact.data["opponent"])
    verb = _past_tense("rozegrał", gender)
    return f"{nom} {verb} swój {n}. mecz w historii reprezentacji, przeciwko {opponent_dative}."


def goal_count_milestone(fact: Fact) -> str:
    """"[Team] passed N career goals in their match against [opponent]."""
    nom = nominative_pl(fact.subject_team)
    gender = gender_pl(fact.subject_team)
    n = fact.data["milestone_number"]
    opponent_dative = dative_pl(fact.data["opponent"])
    verb = _past_tense("przekroczył", gender)
    # Always the many-form ("goli") -- goal_count_milestone's round_numbers
    # start at 100, so the 1/2-4 cases of _count_form never occur here.
    return f"{nom} {verb} granicę {n} goli w historii w meczu przeciwko {opponent_dative}."


def h2h_never_won(fact: Fact) -> str:
    """"[Team] has never beaten [opponent] in N meetings."""
    nom = nominative_pl(fact.subject_team)
    gender = gender_pl(fact.subject_team)
    opponent_dative = dative_pl(fact.data["opponent"])
    n = fact.data["meeting_count"]
    verb = _past_tense("rozegrał", gender)
    noun = _count_form(n, "spotkanie", "spotkania", "spotkań")
    return f"{nom} {verb} {n} {noun} przeciwko {opponent_dative} bez ani jednego zwycięstwa."


def h2h_first_win_since(fact: Fact) -> str:
    """"[Team]'s win against [opponent] in year Y was their first since year Z."""
    nom = nominative_pl(fact.subject_team)
    gender = gender_pl(fact.subject_team)
    opponent_accusative = accusative_pl(fact.data["opponent"])
    win_year = fact.data["win_year"]
    gap_years = fact.data["gap_years"]
    verb = _past_tense("pokonał", gender)
    if gap_years is None:
        return f"{nom} {verb} {opponent_accusative} w {win_year} roku, pierwszy raz w historii."
    years_form = _genitive_count(gap_years, "roku", "lat")
    return f"{nom} {verb} {opponent_accusative} w {win_year} roku, pierwszy raz od {gap_years} {years_form}."
