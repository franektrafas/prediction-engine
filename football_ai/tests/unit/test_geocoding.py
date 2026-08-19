"""Tests for real, sourced coordinate/timezone lookups for match locations."""

from __future__ import annotations

import math
import unittest

from football_ai.data.geocoding import (
    coordinates_for_city_country,
    coordinates_for_stadium,
)


def _haversine_km(point_a: tuple[float, float], point_b: tuple[float, float]) -> float:
    """Great-circle distance in km, for sanity-checking real coordinates in tests."""
    earth_radius_km = 6371.0
    lat1, lon1 = math.radians(point_a[0]), math.radians(point_a[1])
    lat2, lon2 = math.radians(point_b[0]), math.radians(point_b[1])
    delta_lat, delta_lon = lat2 - lat1, lon2 - lon1
    a = (
        math.sin(delta_lat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(delta_lon / 2) ** 2
    )
    return 2 * earth_radius_km * math.asin(math.sqrt(a))


class CoordinatesForCityCountryTestCase(unittest.TestCase):
    """Validate city/country resolution against real geonamescache data."""

    def test_known_capitals_resolve_to_real_coordinates(self) -> None:
        """Warsaw and London should resolve to their real coordinates/timezones."""
        warsaw = coordinates_for_city_country("Warsaw", "Poland")
        london = coordinates_for_city_country("London", "England")
        self.assertIsNotNone(warsaw)
        self.assertIsNotNone(london)
        lat_w, lon_w, tz_w = warsaw
        lat_l, lon_l, _tz_l = london
        self.assertAlmostEqual(lat_w, 52.23, delta=0.5)
        self.assertAlmostEqual(lon_w, 21.01, delta=0.5)
        self.assertEqual(tz_w, "Europe/Warsaw")

        distance_km = _haversine_km((lat_w, lon_w), (lat_l, lon_l))
        self.assertAlmostEqual(distance_km, 1450, delta=100)

    def test_uk_football_associations_resolve_via_country_alias(self) -> None:
        """England/Scotland/Wales/N. Ireland aren't ISO states but are UEFA teams."""
        cardiff = coordinates_for_city_country("Cardiff", "Wales")
        glasgow = coordinates_for_city_country("Glasgow", "Scotland")
        belfast = coordinates_for_city_country("Belfast", "Northern Ireland")
        self.assertIsNotNone(cardiff)
        self.assertIsNotNone(glasgow)
        self.assertIsNotNone(belfast)

    def test_former_or_alternate_country_names_resolve(self) -> None:
        """Cape Verde/Czechia/Eswatini etc. renamed since some old rows were written."""
        self.assertIsNotNone(coordinates_for_city_country("Praia", "Cape Verde"))
        self.assertIsNotNone(coordinates_for_city_country("Prague", "Czech Republic"))
        self.assertIsNotNone(coordinates_for_city_country("Mbabane", "Swaziland"))

    def test_unknown_city_returns_none_rather_than_guessing(self) -> None:
        """An unrecognized city must return None, not a fabricated coordinate."""
        self.assertIsNone(coordinates_for_city_country("Not A Real City", "Poland"))

    def test_unknown_country_returns_none_rather_than_guessing(self) -> None:
        """An unrecognized country must return None, not a fabricated coordinate."""
        self.assertIsNone(coordinates_for_city_country("Warsaw", "Not A Real Country"))

    def test_empty_inputs_return_none(self) -> None:
        """Empty city/country strings must return None."""
        self.assertIsNone(coordinates_for_city_country("", "Poland"))
        self.assertIsNone(coordinates_for_city_country("Warsaw", ""))


class CoordinatesForStadiumTestCase(unittest.TestCase):
    """Validate stadium-string resolution against the real 2026-27 NL fixture table."""

    def test_known_stadium_resolves(self) -> None:
        """A known stadium string resolves to its city's real coordinates."""
        result = coordinates_for_stadium("Stadion Narodowy, Warsaw")
        self.assertIsNotNone(result)
        lat, _lon, tz = result
        self.assertAlmostEqual(lat, 52.23, delta=0.5)
        self.assertEqual(tz, "Europe/Warsaw")

    def test_country_suffixed_neutral_venue_uses_suffix_not_home_team(self) -> None:
        """Israel's home_team_name is Israel, but this venue is really in Hungary."""
        result = coordinates_for_stadium("Nagyerdei Stadion, Debrecen (Hungary)")
        self.assertIsNotNone(result)
        _lat, _lon, tz = result
        self.assertEqual(tz, "Europe/Budapest")

    def test_local_language_city_name_resolves_via_alternate_names(self) -> None:
        """"Wien" (German for Vienna) resolves via geonamescache's alternate names."""
        result = coordinates_for_stadium("Ernst-Happel-Stadion, Wien")
        self.assertIsNotNone(result)

    def test_microstate_venue_without_a_real_city_match_returns_none(self) -> None:
        """Serravalle has no distinct entry in geonamescache's tiny San Marino list."""
        self.assertIsNone(coordinates_for_stadium("San Marino Stadium, Serravalle"))

    def test_unrecognized_stadium_returns_none(self) -> None:
        """A stadium string outside the known table must return None."""
        self.assertIsNone(coordinates_for_stadium("Not A Real Stadium, Nowhere"))

    def test_empty_stadium_string_returns_none(self) -> None:
        """An empty (not-yet-announced) stadium string must return None."""
        self.assertIsNone(coordinates_for_stadium(""))


if __name__ == "__main__":
    unittest.main()
