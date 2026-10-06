"""SHAP helpers for tree model interpretability."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Sequence, Union

import numpy as np


def explain_tree_model(
    model: Any,
    X: np.ndarray,
    feature_names: Optional[Sequence[str]] = None,
    max_samples: int = 400,
    random_state: int = 42,
):
    """
    Compute SHAP values for a LightGBM / tree model.

    ``model`` may be a ``LightGBMRegressor`` or a raw booster with ``predict``.
    """
    import shap

    booster = getattr(model, "booster_", model)
    X = np.asarray(X, dtype=float)
    rng = np.random.default_rng(random_state)
    if X.shape[0] > max_samples:
        idx = rng.choice(X.shape[0], size=max_samples, replace=False)
        X_s = X[idx]
    else:
        X_s = X

    explainer = shap.TreeExplainer(booster)
    shap_values = explainer.shap_values(X_s)
    return explainer, shap_values, X_s, feature_names


def save_shap_summary(
    shap_values: np.ndarray,
    X: np.ndarray,
    feature_names: Optional[Sequence[str]],
    out_path: Union[str, Path],
    max_display: int = 15,
) -> Path:
    """Write a SHAP beeswarm-style summary plot to ``out_path``."""
    import matplotlib.pyplot as plt
    import shap

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    shap.summary_plot(
        shap_values,
        X,
        feature_names=list(feature_names) if feature_names is not None else None,
        max_display=max_display,
        show=False,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close()
    return out_path


def save_shap_bar(
    shap_values: np.ndarray,
    feature_names: Optional[Sequence[str]],
    out_path: Union[str, Path],
    max_display: int = 15,
) -> Path:
    """Mean-|SHAP| bar chart."""
    import matplotlib.pyplot as plt
    import shap

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    shap.summary_plot(
        shap_values,
        feature_names=list(feature_names) if feature_names is not None else None,
        plot_type="bar",
        max_display=max_display,
        show=False,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close()
    return out_path
