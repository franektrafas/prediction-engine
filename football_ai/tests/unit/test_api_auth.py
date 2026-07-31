"""Tests for football_ai/api/routes/auth.py."""

from __future__ import annotations

import os
import unittest
from unittest import mock

from fastapi import HTTPException

from football_ai.api.routes.auth import require_api_key


class RequireApiKeyTestCase(unittest.TestCase):
    """Validate the fail-closed API-key dependency."""

    def test_unconfigured_server_key_returns_503(self) -> None:
        """A server with no FOOTBALL_AI_API_KEY set must reject every request, not silently allow it."""
        with mock.patch.dict(os.environ, {"FOOTBALL_AI_API_KEY": ""}):
            with self.assertRaises(HTTPException) as context:
                require_api_key(x_api_key="anything")
            self.assertEqual(context.exception.status_code, 503)

    def test_wrong_key_returns_401(self) -> None:
        with mock.patch.dict(os.environ, {"FOOTBALL_AI_API_KEY": "real-secret"}):
            with self.assertRaises(HTTPException) as context:
                require_api_key(x_api_key="wrong-guess")
            self.assertEqual(context.exception.status_code, 401)

    def test_missing_header_returns_401(self) -> None:
        with mock.patch.dict(os.environ, {"FOOTBALL_AI_API_KEY": "real-secret"}):
            with self.assertRaises(HTTPException) as context:
                require_api_key(x_api_key=None)
            self.assertEqual(context.exception.status_code, 401)

    def test_correct_key_passes_through(self) -> None:
        with mock.patch.dict(os.environ, {"FOOTBALL_AI_API_KEY": "real-secret"}):
            require_api_key(x_api_key="real-secret")  # must not raise


if __name__ == "__main__":
    unittest.main()
