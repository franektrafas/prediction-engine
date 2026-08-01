"""Canonical names and aliases for national teams."""

from __future__ import annotations

TEAM_ALIASES: dict[str, str] = {
    "usa": "United States",
    "united states of america": "United States",
    "england national football team": "England",
    "republic of ireland": "Ireland",
    "ir iran": "Iran",
    "korea republic": "South Korea",
    "korea rep": "South Korea",
    "south korea": "South Korea",
    "korea dpr": "North Korea",
    "cote d'ivoire": "Ivory Coast",
    "cote divoire": "Ivory Coast",
    "ivory coast": "Ivory Coast",
    "czech republic": "Czechia",
    "macedonia": "North Macedonia",
    "bosnia-herzegovina": "Bosnia and Herzegovina",
    "uae": "United Arab Emirates",
    "dr congo": "Congo DR",
    "congo dr": "Congo DR",
    "cape verde": "Cabo Verde",
    # StatsBomb's real spelling for AFCON 2023 (confirmed 2026-08-01 diagnosing
    # the merge_advanced_stats gap) -- without this, every Cape Verde match in
    # that tournament silently failed to merge, not because of a date/timezone
    # issue like most of the rest of that gap, just an unmapped spelling.
    "cape verde islands": "Cabo Verde",
    "curacao": "Curacao",
    "syria": "Syria",
    "eswatini": "Swaziland",
    "burma": "Myanmar",
    "china pr": "China",
    "trinidad and tobago": "Trinidad and Tobago",
    "northern ireland": "Northern Ireland",
    "wales": "Wales",
    "scotland": "Scotland",
    # ESPN's real spelling for Turkey (its own official rebrand, confirmed
    # 2026-08-02 diagnosing why 24 of 841 real ESPN matches weren't merging
    # into the base file -- all 24 involved Turkey, none of them a
    # date/timezone issue like the earlier StatsBomb gap, just this
    # unmapped spelling. Base file's own canonical spelling is "Turkey"
    # (314 real rows), so that stays canonical rather than switching over.
    "turkiye": "Turkey",
    # Historical entities with a clean, continuous successor -- sourced from
    # martj42/international_results' former_names.csv (2026-07-24), which
    # documents renames of a still-continuous federation, plus "west germany"
    # (not in that file -- martj42's own results.csv already folds West
    # Germany's matches into "Germany" directly, so it never needed the
    # alias internally, but other sources might still spell it out).
    # Deliberately NOT aliased: "german dr" (East Germany) and "yugoslavia"
    # have no single clean modern successor and stay their own canonical
    # entities -- attributing e.g. Yugoslavia's records to Serbia would be
    # wrong, not just imprecise. See football_ai/scrapers/sources/
    # international_results.py for how this feeds the historical merge.
    "west germany": "Germany",
    "dahomey": "Benin",
    "upper volta": "Burkina Faso",
    "netherlands antilles": "Curacao",
    "bohemia": "Czechoslovakia",
    "bohemia and moravia": "Czechoslovakia",
    "representation of czechs and slovaks": "Czechoslovakia",
    "belgian congo": "Congo DR",
    "congo-leopoldville": "Congo DR",
    "congo-kinshasa": "Congo DR",
    "soviet union": "Russia",
    "fr yugoslavia": "Serbia",
}
