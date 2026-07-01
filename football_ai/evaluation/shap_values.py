"""SHAP utilities."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from football_ai.data.io import ensure_parent_directory
from football_ai.models.base import _require_dependency


def save_shap_summary_plot(
    estimator: object,
    features: pd.DataFrame,
    output_path: str | Path,
) -> Path:
    """Generate and save a SHAP summary plot."""
    shap = _require_dependency("shap")
    matplotlib = _require_dependency("matplotlib.pyplot")
    explainer = shap.Explainer(estimator)
    shap_values = explainer(features)
    output = Path(output_path)
    ensure_parent_directory(output)
    shap.summary_plot(shap_values, features, show=False)
    matplotlib.tight_layout()
    matplotlib.savefig(output, dpi=200)
    matplotlib.close()
    return output
