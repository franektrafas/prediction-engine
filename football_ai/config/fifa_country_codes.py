"""FIFA/UEFA 3-letter national-team codes, as used by Wikipedia's `{{fb|XXX}}` templates.

Needed because `scrapers/sources/uefa_nations_league.py` parses match data
straight out of Wikipedia's raw wikitext (`{{Football box|team1={{fb-rt|POL}}...}}`),
which identifies teams by 3-letter code, not by name -- a different alias
problem than `config.team_aliases` (which resolves *name* spelling
differences between sources, not code-to-name).

UEFA member codes below were verified directly against real, cross-referenced
Wikipedia data (2026-07-30: rendered country names vs. raw wikitext codes for
the same tables, `2026–27 UEFA Nations League B`) for POL/BIH/SWE/ROU/SCO/SUI/
SVN/MKD/HUN/UKR/GEO/NIR/ISR/AUT/IRL/KOS/CZE/ENG/GRE/NOR/TUR/WAL/ALB/FIN/ISL/
KAZ/MNE. The rest of the table is standard, well-established FIFA
country-code knowledge, not independently re-verified page by page --
`resolve_fifa_code` raises on an unmapped code rather than guessing, so a
gap here fails loudly (a KeyError on some future page) instead of silently
misattributing a match.
"""

from __future__ import annotations

FIFA_CODE_TO_CANONICAL_NAME: dict[str, str] = {
    "ALB": "Albania",
    "AND": "Andorra",
    "ARM": "Armenia",
    "AUT": "Austria",
    "AZE": "Azerbaijan",
    "BEL": "Belgium",
    "BIH": "Bosnia and Herzegovina",
    "BLR": "Belarus",
    "BUL": "Bulgaria",
    "CRO": "Croatia",
    "CYP": "Cyprus",
    "CZE": "Czechia",
    "DEN": "Denmark",
    "ENG": "England",
    "ESP": "Spain",
    "EST": "Estonia",
    "FAR": "Faroe Islands",
    "FIN": "Finland",
    "FRA": "France",
    "GEO": "Georgia",
    "GER": "Germany",
    "GIB": "Gibraltar",
    "GRE": "Greece",
    "HUN": "Hungary",
    "IRL": "Ireland",
    "ISL": "Iceland",
    "ISR": "Israel",
    "ITA": "Italy",
    "KAZ": "Kazakhstan",
    "KOS": "Kosovo",
    "LAT": "Latvia",
    "LIE": "Liechtenstein",
    "LTU": "Lithuania",
    "LUX": "Luxembourg",
    "MDA": "Moldova",
    "MKD": "North Macedonia",
    "MLT": "Malta",
    "MNE": "Montenegro",
    "NED": "Netherlands",
    "NIR": "Northern Ireland",
    "NOR": "Norway",
    "POL": "Poland",
    "POR": "Portugal",
    "ROU": "Romania",
    "RUS": "Russia",
    "SCO": "Scotland",
    "SMR": "San Marino",
    "SRB": "Serbia",
    "SUI": "Switzerland",
    "SVK": "Slovakia",
    "SVN": "Slovenia",
    "SWE": "Sweden",
    "TUR": "Turkey",
    "UKR": "Ukraine",
    "WAL": "Wales",
}


def resolve_fifa_code(code: str) -> str:
    """Map a 3-letter FIFA/UEFA code to this project's canonical team name.

    Raises `KeyError` on an unmapped code -- deliberately loud, not a
    silent pass-through -- since an unresolved code here would otherwise
    become a garbage "team name" (the raw 3-letter string) flowing into
    `canonicalize_team_name` and every downstream consumer.
    """
    normalized = code.strip().upper()
    if normalized not in FIFA_CODE_TO_CANONICAL_NAME:
        raise KeyError(
            f"No canonical name mapped for FIFA code {code!r} -- add it to "
            "FIFA_CODE_TO_CANONICAL_NAME rather than guessing."
        )
    return FIFA_CODE_TO_CANONICAL_NAME[normalized]
