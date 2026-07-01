"""Open-Meteo weather enrichment."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd
from sqlalchemy.orm import Session

from football_ai.database.models import Match, Weather
from football_ai.scrapers.base import BaseDataSource


class OpenMeteoSource(BaseDataSource):
    """Fetcher for historical weather observations."""

    BASE_URL = "https://archive-api.open-meteo.com/v1/archive"

    def __init__(self) -> None:
        """Initialize the source."""
        super().__init__(source_name="open_meteo")

    def fetch_hourly_weather(
        self,
        latitude: float,
        longitude: float,
        match_datetime: datetime,
    ) -> dict[str, Any]:
        """Fetch the nearest hourly weather snapshot for a match time."""
        match_hour = match_datetime.replace(minute=0, second=0, microsecond=0)
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "start_date": match_hour.date().isoformat(),
            "end_date": match_hour.date().isoformat(),
            "hourly": ",".join(
                [
                    "temperature_2m",
                    "apparent_temperature",
                    "precipitation",
                    "windspeed_10m",
                    "relativehumidity_2m",
                    "surface_pressure",
                    "cloudcover",
                    "weathercode",
                ]
            ),
            "timezone": "UTC",
        }
        payload = self.fetch_json(self.BASE_URL, params=params)
        hourly_frame = pd.DataFrame(payload.get("hourly", {}))
        if hourly_frame.empty:
            return {}
        hourly_frame["time"] = pd.to_datetime(hourly_frame["time"], utc=True)
        row = hourly_frame.loc[hourly_frame["time"] == pd.Timestamp(match_hour, tz="UTC")]
        if row.empty:
            row = hourly_frame.iloc[[0]]
        values = row.iloc[0].to_dict()
        return {
            "temperature_c": values.get("temperature_2m"),
            "apparent_temperature_c": values.get("apparent_temperature"),
            "precipitation_mm": values.get("precipitation"),
            "wind_speed_kph": values.get("windspeed_10m"),
            "humidity_pct": values.get("relativehumidity_2m"),
            "pressure_hpa": values.get("surface_pressure"),
            "cloud_cover_pct": values.get("cloudcover"),
            "weather_code": values.get("weathercode"),
        }

    def enrich_matches(self, session: Session, matches: list[Match]) -> None:
        """Fetch and persist weather snapshots for matches with coordinates."""
        for match in matches:
            if match.venue_latitude is None or match.venue_longitude is None:
                continue
            weather_payload = self.fetch_hourly_weather(
                latitude=match.venue_latitude,
                longitude=match.venue_longitude,
                match_datetime=match.match_date,
            )
            if not weather_payload:
                continue
            session.add(Weather(match_id=match.id, **weather_payload))
        session.commit()


if __name__ == "__main__":
    print("Use OpenMeteoSource.fetch_hourly_weather(latitude, longitude, match_datetime).")
