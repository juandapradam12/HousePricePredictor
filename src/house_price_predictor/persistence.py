"""Persist fitted models and feature schemas with joblib."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union

import joblib

from .features import FeatureSchema

ARTIFACT_DEFAULT = Path(__file__).resolve().parents[2] / "artifacts" / "models"


def save_model_bundle(
    path: Union[str, Path],
    model: Any,
    schema: FeatureSchema,
    metadata: Optional[Dict[str, Any]] = None,
) -> Path:
    """Save ``{"model", "schema", "metadata"}`` via joblib."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    bundle = {"model": model, "schema": schema, "metadata": metadata or {}}
    joblib.dump(bundle, path)
    return path


def load_model_bundle(path: Union[str, Path]) -> Dict[str, Any]:
    """Load a bundle previously written by ``save_model_bundle``."""
    bundle = joblib.load(Path(path))
    if not isinstance(bundle, dict) or "model" not in bundle or "schema" not in bundle:
        raise ValueError(f"Invalid model bundle at {path}")
    return bundle
