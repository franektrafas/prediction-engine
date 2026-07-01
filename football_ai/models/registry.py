"""Filesystem-backed model registry."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json

from football_ai.config.settings import get_settings


@dataclass(slots=True)
class RegisteredModel:
    """Persisted registry entry."""

    model_name: str
    target_name: str
    task_type: str
    version: str
    artifact_path: str
    metrics: dict[str, float]


class ModelRegistry:
    """Registry for serialized model artifacts."""

    def __init__(self, registry_path: str | Path | None = None) -> None:
        """Initialize the registry."""
        settings = get_settings()
        self.registry_path = Path(registry_path or settings.model_artifact_dir / "registry.json")
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> dict[str, list[RegisteredModel]]:
        """Load registry entries from disk."""
        if not self.registry_path.exists():
            return {}
        raw = json.loads(self.registry_path.read_text(encoding="utf-8"))
        return {
            key: [RegisteredModel(**item) for item in values]
            for key, values in raw.items()
        }

    def save(self, registry: dict[str, list[RegisteredModel]]) -> None:
        """Persist registry entries to disk."""
        payload = {
            key: [asdict(item) for item in values]
            for key, values in registry.items()
        }
        self.registry_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def register(self, model: RegisteredModel) -> None:
        """Add a model artifact to the registry."""
        registry = self.load()
        registry.setdefault(model.target_name, []).append(model)
        self.save(registry)

    def get_latest(self, target_name: str) -> RegisteredModel | None:
        """Fetch the latest registered model for a target."""
        registered = self.load().get(target_name, [])
        if not registered:
            return None
        return registered[-1]
