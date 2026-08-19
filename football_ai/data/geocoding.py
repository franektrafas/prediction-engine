"""Real, sourced coordinates + timezone for international-match locations.

Built for the Travel Fatigue Index: both entry points return `None` rather
than a guessed value when a location can't be confidently resolved --
never fall back to a coordinate recalled from model memory.

Data source: `geonamescache` (https://pypi.org/project/geonamescache/,
installed v3.0.2), which bundles GeoNames.org's (https://www.geonames.org/)
open geographic database -- 252 countries, ~34k cities, each with a real
`latitude`/`longitude`/`timezone` (IANA tz name) and a list of real
alternate names (local-language/historical spellings) sourced from the
same database, not guessed here.

Two entry points:
  - `coordinates_for_city_country(city, country)` -- for `records_frame`,
    which carries explicit historical `city`/`country` columns.
  - `coordinates_for_stadium(stadium_name)` -- for upcoming fixtures (e.g.
    UEFA Nations League, scraped from Wikipedia via
    `scrapers.sources.uefa_nations_league.UefaNationsLeagueSource`), which
    only carry a `"StadiumName, City"` (optionally `"(Country)"`-suffixed)
    string, resolved via `_NATIONS_LEAGUE_STADIUMS` below.

Country-name resolution: `records_frame.country` and NL `home_team_name`
strings don't always match `geonamescache`'s own country names verbatim.
Two real reasons: (1) UEFA/FIFA football associations that aren't
sovereign ISO 3166 states -- England, Scotland, Wales, Northern Ireland
each field their own national team under the United Kingdom, and Kosovo,
Palestine, Northern Cyprus, Zanzibar, Tahiti and a handful of others play
under a name that isn't the exact ISO short name of the state that
administers or borders them; (2) genuine spelling/former-name variants
("Cape Verde"/"Cabo Verde", "Swaziland"/"Eswatini", "Czech Republic"/
"Czechia"). Matching tries an accent/case-insensitive match against
`geonamescache`'s own country names first; only the real mismatches above
fall back to `_COUNTRY_ALIASES`, each commented with the concrete reason.
Anything not covered by either path returns `None`.
"""

from __future__ import annotations

import re
from functools import lru_cache

from football_ai.data.normalization import normalize_text


def _normalize_for_matching(value: str) -> str:
    """Strip accents/case (via `normalize_text`) then punctuation, for name matching.

    `normalize_text` alone leaves punctuation in place ("St. Gallen" stays
    "st. gallen"), which would make an otherwise-identical name fail to
    match its `geonamescache` counterpart ("Sankt Gallen" -> "sankt
    gallen"). Collapsing non-alphanumeric runs to a single space (and
    dropping a leading "the ") lets city/country strings that differ only
    in punctuation or a definite article still match.
    """
    text = normalize_text(value)
    text = re.sub(r"^the ", "", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return text.strip()


# Real name mismatches between this project's country strings and
# geonamescache's own country names -- NOT guesses. Each entry is either a
# UEFA/FIFA football association that isn't itself an ISO 3166 sovereign
# state (mapped to the real state whose territory/city data actually
# covers it in geonamescache), or a documented former/alternate country
# name. Keys are pre-normalized (`_normalize_for_matching` output).
_COUNTRY_ALIASES: dict[str, str] = {
    # Channel Islands: Alderney is part of the Bailiwick of Guernsey.
    "alderney": "GG",
    # ISO 3166 short name is "Bonaire, Sint Eustatius and Saba".
    "bonaire": "BQ",
    # Official short name since 2013 is "Cabo Verde".
    "cape verde": "CV",
    "congo": "CG",
    # Official short name since 2016 is "Czechia".
    "czech republic": "CZ",
    "dr congo": "CD",
    # Separate FIFA/UEFA national team, part of the United Kingdom.
    "england": "GB",
    "macau": "MO",
    # Not a UN member state; geographically the northern part of the
    # island of Cyprus -- geonamescache's single CY entry (which includes
    # Nicosia/Famagusta/Kyrenia) covers the whole island.
    "northern cyprus": "CY",
    # Separate FIFA/UEFA national team, part of the United Kingdom.
    "northern ireland": "GB",
    # geonamescache's own name is "Palestinian Territory".
    "palestine": "PS",
    # Common football-context name for Ireland, used to disambiguate from
    # Northern Ireland.
    "republic of ireland": "IE",
    # Separate FIFA/UEFA national team, part of the United Kingdom.
    "scotland": "GB",
    # Official short name since 2018 is "Eswatini".
    "swaziland": "SZ",
    # Tahiti is the main island of French Polynesia; plays as its own team
    # in non-FIFA (OFC-adjacent) competitions.
    "tahiti": "PF",
    "timor leste": "TL",
    "united states virgin islands": "VI",
    # Separate FIFA/UEFA national team, part of the United Kingdom.
    "wales": "GB",
    # Semi-autonomous region of Tanzania with its own historical football
    # association.
    "zanzibar": "TZ",
}


@lru_cache(maxsize=1)
def _country_name_index() -> dict[str, str]:
    import geonamescache

    countries = geonamescache.GeonamesCache().get_countries()
    return {
        _normalize_for_matching(entry["name"]): code
        for code, entry in countries.items()
    }


def _country_code(country: str) -> str | None:
    key = _normalize_for_matching(country)
    if not key:
        return None
    return _country_name_index().get(key) or _COUNTRY_ALIASES.get(key)


@lru_cache(maxsize=1)
def _city_index() -> dict[tuple[str, str], dict]:
    """Build a `(country_code, normalized_name) -> city record` index.

    Indexes both each city's primary name and its real `alternatenames`
    (geonamescache's own local-language/historical spellings), so e.g.
    Vienna's German name "Wien" or Seville's Spanish name "Sevilla" resolve
    without a hand-maintained alias table. On a collision within the same
    country (two cities sharing a normalized name/alias), the more
    populous one wins.
    """
    import geonamescache

    index: dict[tuple[str, str], dict] = {}
    for city in geonamescache.GeonamesCache().get_cities().values():
        candidate_names = [city["name"], *city.get("alternatenames", [])]
        for name in candidate_names:
            key = (city["countrycode"], _normalize_for_matching(name))
            if not key[1]:
                continue
            current = index.get(key)
            if current is None or city["population"] > current["population"]:
                index[key] = city
    return index


def _coordinates_from_city_record(city: dict) -> tuple[float, float, str]:
    return (float(city["latitude"]), float(city["longitude"]), city["timezone"])


def coordinates_for_city_country(
    city: str, country: str
) -> tuple[float, float, str] | None:
    """Real `(latitude, longitude, iana_timezone)` for a `records_frame`-style location.

    Returns `None` if `country` doesn't resolve to a real country (via
    geonamescache's own names or `_COUNTRY_ALIASES`) or `city` isn't a
    known city (or alternate name of one) within it -- never guesses.
    """
    if not city or not country:
        return None
    country_code = _country_code(country)
    if country_code is None:
        return None
    record = _city_index().get((country_code, _normalize_for_matching(city)))
    if record is None:
        return None
    return _coordinates_from_city_record(record)


_STADIUM_STRING_PATTERN = re.compile(
    r"^(?P<venue>.+?),\s*(?P<city>[^(]+?)(?:\s*\((?P<country>[^)]+)\))?$"
)

# Real (stadium, home_team_name) pairs from the 2026-27 UEFA Nations League
# schedule, fetched 2026-08-19 via
# `scrapers.sources.uefa_nations_league.UefaNationsLeagueSource.fetch_group_matches`
# across divisions A-D, groups 1-4 (67 of the 132 fixtures scraped that day
# had a non-empty `stadium` field; the rest are TBD on Wikipedia). Divisions
# C group 2/3 and D group 3/4 raised unrelated `FIFA_CODE_TO_CANONICAL_NAME`
# gaps in the scraper (Latvia/Faroe Islands codes, and two blank stadium
# fields) -- pre-existing scraper gaps, not a geocoding problem, so those
# groups aren't represented here.
#
# `home_team_name` is only used as a fallback country when the stadium
# string itself carries no `"(Country)"` suffix -- Wikipedia adds that
# suffix precisely for the neutral-venue cases (e.g. Ukraine hosting "at
# home" in Slovakia, Israel in Hungary) where the home team's own country
# would otherwise be wrong, so the suffix always wins when present.
_NATIONS_LEAGUE_STADIUMS: tuple[tuple[str, str], ...] = (
    ("Allianz Arena, Munich", "Germany"),
    ("Anton Malatinský Stadium, Trnava (Slovakia)", "Ukraine"),
    ("Arena Kombëtare, Tirana", "Albania"),
    ("Arena Națională, Bucharest", "Romania"),
    ("Aviva Stadium, Dublin", "Ireland"),
    ("Bilino Polje Stadium, Zenica", "Bosnia and Herzegovina"),
    ("Boris Paichadze Dinamo Arena, Tbilisi", "Georgia"),
    ("Cardiff City Stadium, Cardiff", "Wales"),
    ("Carlos Tartiere, Oviedo", "Spain"),
    ("Centennial Atatürk Stadium, Bursa", "Turkey"),
    ("Darius and Girėnas Stadium, Kaunas", "Lithuania"),
    ("Ernst-Happel-Stadion, Wien", "Austria"),
    ("Estadi de la FAF, Encamp", "Andorra"),
    ("Estádio José Alvalade, Lisbon", "Portugal"),
    ("Estádio Municipal, Braga", "Portugal"),
    ("Estádio do Dragão, Porto", "Portugal"),
    ("Europa Sports Park, Gibraltar", "Gibraltar"),
    ("Fadil Vokrri Stadium, Pristina", "Kosovo"),
    ("Fortuna Arena, Prague", "Czechia"),
    ("Ganja City Stadium, Ganja", "Azerbaijan"),
    ("Hampden Park, Glasgow", "Scotland"),
    ("Helsinki Olympic Stadium, Helsinki", "Finland"),
    ("Hristo Botev Stadium, Plovdiv", "Bulgaria"),
    ("Johan Cruyff Arena, Amsterdam", "Netherlands"),
    ("King Baudouin Stadium, Brussels", "Belgium"),
    ("Kocaeli Stadium, İzmit", "Turkey"),
    ("Laugardalsvöllur, Reykjavík", "Iceland"),
    ("Lilleküla Stadium, Tallinn", "Estonia"),
    ("Metropolitano, Madrid", "Spain"),
    ("Nagyerdei Stadion, Debrecen (Hungary)", "Israel"),
    ("National Stadium, Ta' Qali", "Malta"),
    ("Nationalarenan, Solna", "Sweden"),
    ("Olympiastadion, Berlin", "Germany"),
    ("Opus Arena, Osijek", "Croatia"),
    ("Panthessaliko Stadium, Volos", "Greece"),
    ("Parken Stadium, Copenhagen", "Denmark"),
    ("Philips Stadion, Eindhoven", "Netherlands"),
    ("Puskás Aréna, Budapest", "Hungary"),
    ("Raiffeisen Arena, Linz", "Austria"),
    ("Rajko Mitić Stadium, Belgrade", "Serbia"),
    ("Ramón Sánchez–Pizjuán, Seville", "Spain"),
    ("Rheinpark Stadion, Vaduz", "Liechtenstein"),
    ("San Marino Stadium, Serravalle", "San Marino"),
    ("San Siro, Milan", "Italy"),
    ("Sitterstadion, St. Gallen", "Switzerland"),
    ("Stade Atlantique, Bordeaux", "France"),
    ("Stade Maurice Dufrasne, Liège", "Belgium"),
    ("Stade de France, Saint-Denis", "France"),
    ("Stade de Luxembourg, Luxembourg City", "Luxembourg"),
    ("Stadio Olimpico, Rome", "Italy"),
    ("Stadio Renato Dall'Ara, Bologna", "Italy"),
    ("Stadion HNK Rijeka, Rijeka", "Croatia"),
    ("Stadion Letná, Prague", "Czechia"),
    ("Stadion Narodowy, Warsaw", "Poland"),
    ("Stadion Poljud, Split", "Croatia"),
    ("Stožice Stadium, Ljubljana", "Slovenia"),
    ("TSC Arena, Bačka Topola (Serbia)", "Ireland"),
    ("Thermoplan Arena, Lucerne", "Switzerland"),
    ("Tofiq Bahramov Republican Stadium, Baku", "Azerbaijan"),
    ("Toumba Stadium, Thessaloniki", "Greece"),
    ("Toše Proeski Arena, Skopje", "North Macedonia"),
    ("Töölö Football Stadium, Helsinki", "Finland"),
    ("Ullevaal Stadion, Oslo", "Norway"),
    ("WWK Arena, Augsburg", "Germany"),
    ("Wembley Stadium, London", "England"),
    ("Windsor Park, Belfast", "Northern Ireland"),
    ("Wrocław Stadium, Wrocław", "Poland"),
)


@lru_cache(maxsize=1)
def _stadium_index() -> dict[str, tuple[float, float, str]]:
    """Resolve `_NATIONS_LEAGUE_STADIUMS` into a `stadium string -> coordinates` index.

    Three of the 67 entries (San Marino Stadium/Serravalle, Estadi de la
    FAF/Encamp, National Stadium/Ta' Qali) have no real match in
    geonamescache -- each of those three countries is small enough that
    geonamescache only carries one or a handful of settlements, none of
    which is genuinely the same place as the actual venue town, so they're
    skipped here rather than pointed at a nearby-but-different city.
    """
    index: dict[str, tuple[float, float, str]] = {}
    for stadium_string, home_team_name in _NATIONS_LEAGUE_STADIUMS:
        match = _STADIUM_STRING_PATTERN.match(stadium_string)
        if match is None:
            continue
        city = match.group("city").strip()
        country = match.group("country") or home_team_name
        coordinates = coordinates_for_city_country(city, country)
        if coordinates is None:
            continue
        index[_normalize_for_matching(stadium_string)] = coordinates
    return index


def coordinates_for_stadium(stadium_name: str) -> tuple[float, float, str] | None:
    """Real `(latitude, longitude, iana_timezone)` for a `"StadiumName, City"` string.

    Expects the same format as `UefaNationsLeagueSource.fetch_group_matches`'s
    `stadium` column (optionally `"(Country)"`-suffixed). Returns `None` for
    an unrecognized or not-yet-announced (empty) stadium string -- never
    guesses a venue's location.
    """
    if not stadium_name:
        return None
    return _stadium_index().get(_normalize_for_matching(stadium_name))
