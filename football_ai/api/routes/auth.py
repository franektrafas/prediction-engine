"""API-key authentication for the /v1 route surface.

Only the new Krok 5 routes (`matches.py`, `competitions.py`) require this --
the pre-existing `/predict`/`/health` endpoints are left exactly as they
were, unauthenticated, to avoid a breaking change for whatever already
calls them.
"""

from __future__ import annotations

from fastapi import Header, HTTPException

from football_ai.config.settings import get_settings


def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    """Reject a request unless it carries the configured `X-API-Key` header.

    Fails closed, not open: a server with no `FOOTBALL_AI_API_KEY`
    configured rejects every request with 503 (a real misconfiguration to
    fix) rather than silently treating "no key set" as "auth disabled".
    """
    settings = get_settings()
    if not settings.api_key:
        raise HTTPException(status_code=503, detail="API key not configured on server.")
    if x_api_key != settings.api_key:
        raise HTTPException(status_code=401, detail="Missing or invalid X-API-Key header.")
