"""Polish grammatical forms for UEFA national team names, used by narrative sentence templates.

Every template in `templates_pl.py` was deliberately designed around only
three cases plus verb-agreement class, not full 7-case Polish declension:
- nominative (mianownik) -- a Fact's `subject_team`, always the grammatical
  subject of its sentence ("Polska wygrała...", never "Zwycięstwo Polski...").
- dative (celownik) -- an opponent introduced via the fixed "przeciwko
  [opponent, dative]" construction, used by every template that needs one.
- accusative (biernik) -- the one exception, "pokonała [opponent, accusative]"
  ("defeated X") in `h2h_first_win_since`, which reads far more naturally
  than forcing "przeciwko" there too.
- `gender`: which past-tense verb ending a team takes as a grammatical
  subject -- `"f"` (Polska wygrała), `"m"` (Izrael wygrał), `"n"` (Kosowo
  wygrało), or `"pl"` (Niemcy wygrały -- pluralia tantum country names:
  Germany/Italy/Hungary/Czechia/Faroe Islands are all grammatically plural
  nouns in Polish, not "many countries", one country with a plural name).

Same 55-team UEFA member list as `config.fifa_country_codes`, for the same
reason: this project's canonical English team name
(`data.normalization.canonicalize_team_name`'s output) is the only key any
caller has, so this dict is keyed by that, not by FIFA code.

Standard, well-established Polish country-name forms -- not independently
verified against an official style guide team by team (the same standing
this project applies to `fifa_country_codes.py`'s own table). If a specific
form turns out wrong, fix it here; `team_form()`/`dative_pl()`/`accusative_pl()`
degrade to the raw English name rather than crashing for anything not in
this table, so a gap here produces slightly-off Polish, never a failure.
"""

from __future__ import annotations


# canonical_english_name: (nominative, dative, accusative, gender)
_TEAM_FORMS_PL: dict[str, tuple[str, str, str, str]] = {
    "Albania": ("Albania", "Albanii", "Albanię", "f"),
    "Andorra": ("Andora", "Andorze", "Andorę", "f"),
    "Armenia": ("Armenia", "Armenii", "Armenię", "f"),
    "Austria": ("Austria", "Austrii", "Austrię", "f"),
    "Azerbaijan": ("Azerbejdżan", "Azerbejdżanowi", "Azerbejdżan", "m"),
    "Belgium": ("Belgia", "Belgii", "Belgię", "f"),
    "Bosnia and Herzegovina": ("Bośnia i Hercegowina", "Bośni i Hercegowinie", "Bośnię i Hercegowinę", "f"),
    "Belarus": ("Białoruś", "Białorusi", "Białoruś", "f"),
    "Bulgaria": ("Bułgaria", "Bułgarii", "Bułgarię", "f"),
    "Croatia": ("Chorwacja", "Chorwacji", "Chorwację", "f"),
    "Cyprus": ("Cypr", "Cyprowi", "Cypr", "m"),
    "Czechia": ("Czechy", "Czechom", "Czechy", "pl"),
    "Denmark": ("Dania", "Danii", "Danię", "f"),
    "England": ("Anglia", "Anglii", "Anglię", "f"),
    "Spain": ("Hiszpania", "Hiszpanii", "Hiszpanię", "f"),
    "Estonia": ("Estonia", "Estonii", "Estonię", "f"),
    "Faroe Islands": ("Wyspy Owcze", "Wyspom Owczym", "Wyspy Owcze", "pl"),
    "Finland": ("Finlandia", "Finlandii", "Finlandię", "f"),
    "France": ("Francja", "Francji", "Francję", "f"),
    "Georgia": ("Gruzja", "Gruzji", "Gruzję", "f"),
    "Germany": ("Niemcy", "Niemcom", "Niemcy", "pl"),
    "Gibraltar": ("Gibraltar", "Gibraltarowi", "Gibraltar", "m"),
    "Greece": ("Grecja", "Grecji", "Grecję", "f"),
    "Hungary": ("Węgry", "Węgrom", "Węgry", "pl"),
    "Ireland": ("Irlandia", "Irlandii", "Irlandię", "f"),
    "Iceland": ("Islandia", "Islandii", "Islandię", "f"),
    "Israel": ("Izrael", "Izraelowi", "Izrael", "m"),
    "Italy": ("Włochy", "Włochom", "Włochy", "pl"),
    "Kazakhstan": ("Kazachstan", "Kazachstanowi", "Kazachstan", "m"),
    "Kosovo": ("Kosowo", "Kosowu", "Kosowo", "n"),
    "Latvia": ("Łotwa", "Łotwie", "Łotwę", "f"),
    "Liechtenstein": ("Liechtenstein", "Liechtensteinowi", "Liechtenstein", "m"),
    "Lithuania": ("Litwa", "Litwie", "Litwę", "f"),
    "Luxembourg": ("Luksemburg", "Luksemburgowi", "Luksemburg", "m"),
    "Moldova": ("Mołdawia", "Mołdawii", "Mołdawię", "f"),
    "North Macedonia": ("Macedonia Północna", "Macedonii Północnej", "Macedonię Północną", "f"),
    "Malta": ("Malta", "Malcie", "Maltę", "f"),
    "Montenegro": ("Czarnogóra", "Czarnogórze", "Czarnogórę", "f"),
    "Netherlands": ("Holandia", "Holandii", "Holandię", "f"),
    "Northern Ireland": ("Irlandia Północna", "Irlandii Północnej", "Irlandię Północną", "f"),
    "Norway": ("Norwegia", "Norwegii", "Norwegię", "f"),
    "Poland": ("Polska", "Polsce", "Polskę", "f"),
    "Portugal": ("Portugalia", "Portugalii", "Portugalię", "f"),
    "Romania": ("Rumunia", "Rumunii", "Rumunię", "f"),
    "Russia": ("Rosja", "Rosji", "Rosję", "f"),
    "Scotland": ("Szkocja", "Szkocji", "Szkocję", "f"),
    "San Marino": ("San Marino", "San Marino", "San Marino", "n"),
    "Serbia": ("Serbia", "Serbii", "Serbię", "f"),
    "Switzerland": ("Szwajcaria", "Szwajcarii", "Szwajcarię", "f"),
    "Slovakia": ("Słowacja", "Słowacji", "Słowację", "f"),
    "Slovenia": ("Słowenia", "Słowenii", "Słowenię", "f"),
    "Sweden": ("Szwecja", "Szwecji", "Szwecję", "f"),
    "Turkey": ("Turcja", "Turcji", "Turcję", "f"),
    "Ukraine": ("Ukraina", "Ukrainie", "Ukrainę", "f"),
    "Wales": ("Walia", "Walii", "Walię", "f"),
}


def team_form(team_name: str) -> tuple[str, str, str, str]:
    """Return `(nominative, dative, accusative, gender)` for a team, falling back to the raw name.

    A team not in `_TEAM_FORMS_PL` (any non-UEFA opponent a friendly could
    produce a Fact about) gets its raw canonical English name repeated for
    every case and `"n"` (neuter -- an indeclinable foreign name reads most
    naturally that way in Polish, e.g. "San Marino") as the safest
    verb-agreement default -- degrades to slightly-off Polish, never raises.
    """
    return _TEAM_FORMS_PL.get(team_name, (team_name, team_name, team_name, "n"))


def nominative_pl(team_name: str) -> str:
    """Polish nominative form (sentence subject)."""
    return team_form(team_name)[0]


def dative_pl(team_name: str) -> str:
    """Polish dative form (for "przeciwko X" constructions)."""
    return team_form(team_name)[1]


def accusative_pl(team_name: str) -> str:
    """Polish accusative form (for "pokonała X" constructions)."""
    return team_form(team_name)[2]


def gender_pl(team_name: str) -> str:
    """Verb-agreement class: `"f"`, `"m"`, `"n"`, or `"pl"`."""
    return team_form(team_name)[3]
