"""FastAPI application entrypoint."""

from __future__ import annotations

from fastapi import FastAPI

from football_ai.api.routes.predict import router as prediction_router


app = FastAPI(
    title="Football AI",
    version="0.1.0",
    description="REST API for national team football match predictions.",
)
app.include_router(prediction_router)


@app.get("/health")
def healthcheck() -> dict[str, str]:
    """Return a basic healthcheck payload."""
    return {"status": "ok"}
