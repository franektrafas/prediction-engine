"""Base utilities for external data connectors."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import json
import logging

import pandas as pd
import requests

from football_ai.config.settings import AppSettings
from football_ai.config.settings import get_settings
from football_ai.data.io import ensure_parent_directory, write_table


LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class BaseDataSource:
    """Base data source abstraction."""

    source_name: str
    timeout_seconds: int = 60
    settings: AppSettings = field(default_factory=get_settings)

    def fetch_json(self, url: str, params: dict[str, Any] | None = None) -> Any:
        """Fetch JSON content from a remote URL."""
        response = requests.get(url, params=params, timeout=self.timeout_seconds)
        response.raise_for_status()
        return response.json()

    def fetch_json_with_headers(
        self,
        url: str,
        headers: dict[str, str],
        params: dict[str, Any] | None = None,
    ) -> Any:
        """Fetch JSON content using custom HTTP headers."""
        response = requests.get(
            url,
            params=params,
            headers=headers,
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        return response.json()

    def fetch_text_with_headers(
        self,
        url: str,
        headers: dict[str, str],
        params: dict[str, Any] | None = None,
    ) -> str:
        """Fetch text content using custom HTTP headers."""
        response = requests.get(
            url,
            params=params,
            headers=headers,
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        return response.text

    def fetch_text(self, url: str, params: dict[str, Any] | None = None) -> str:
        """Fetch text content from a remote URL."""
        response = requests.get(url, params=params, timeout=self.timeout_seconds)
        response.raise_for_status()
        return response.text

    def fetch_table(self, url: str) -> pd.DataFrame:
        """Fetch the first HTML table from a page."""
        tables = pd.read_html(url)
        if not tables:
            raise ValueError(f"No tables found at {url}")
        return tables[0]

    def raw_output_directory(self) -> Path:
        """Return the raw storage directory for the source."""
        return self.settings.project_root / "data" / "raw" / self.source_name

    def persist_raw_json(self, payload: Any, file_name: str) -> Path:
        """Persist raw JSON for traceability and offline reuse."""
        output_path = self.raw_output_directory() / file_name
        ensure_parent_directory(output_path)
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return output_path

    def persist_frame(self, frame: pd.DataFrame, file_name: str) -> Path:
        """Persist a normalized table."""
        output_path = self.raw_output_directory() / file_name
        write_table(frame, output_path)
        return output_path
