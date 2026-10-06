"""Build Kaggle-style submission CSVs for the Ames Housing test set."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import pandas as pd

from .features import FeatureSchema, transform_with_schema
from .persistence import load_model_bundle


def predict_saleprice(
    model: Any,
    schema: FeatureSchema,
    frame: pd.DataFrame,
) -> np.ndarray:
    """Predict USD sale prices (applies ``expm1`` when schema used a log target)."""
    X = transform_with_schema(frame, schema)
    pred = np.asarray(model.predict(X), dtype=float).reshape(-1)
    if schema.log_target:
        pred = np.expm1(pred)
    return pred


def make_submission(
    model: Any,
    schema: FeatureSchema,
    test_frame: pd.DataFrame,
    out_path: Union[str, Path],
) -> Path:
    """Write ``Id,SalePrice`` CSV ready for Kaggle upload."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    preds = predict_saleprice(model, schema, test_frame)
    ids = test_frame["Id"] if "Id" in test_frame.columns else np.arange(1, len(test_frame) + 1)
    sub = pd.DataFrame({"Id": ids, "SalePrice": preds})
    sub.to_csv(out_path, index=False)
    return out_path


def make_submission_from_bundle(
    bundle_path: Union[str, Path],
    test_frame: pd.DataFrame,
    out_path: Union[str, Path],
) -> Path:
    """Load a saved bundle and write a submission file."""
    bundle = load_model_bundle(bundle_path)
    return make_submission(bundle["model"], bundle["schema"], test_frame, out_path)
