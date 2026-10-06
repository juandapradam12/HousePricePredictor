"""Gradient boosting baseline (LightGBM) for a nonlinear performance ceiling."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

try:
    import lightgbm as lgb

    HAS_LIGHTGBM = True
except ImportError:  # pragma: no cover
    lgb = None  # type: ignore
    HAS_LIGHTGBM = False


DEFAULT_LGBM_PARAMS: Dict[str, Any] = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.03,
    "num_leaves": 63,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "min_data_in_leaf": 15,
    "reg_lambda": 1.0,
    "verbosity": -1,
}


class LightGBMRegressor:
    """Thin wrapper so LightGBM matches the project's ``fit`` / ``predict`` API."""

    def __init__(
        self,
        params: Optional[Dict[str, Any]] = None,
        n_estimators: int = 500,
        early_stopping_rounds: int = 50,
        random_state: int = 42,
    ) -> None:
        if not HAS_LIGHTGBM:
            raise ImportError("lightgbm is required for LightGBMRegressor. pip install lightgbm")
        self.params = {**DEFAULT_LGBM_PARAMS, **(params or {}), "seed": random_state}
        self.n_estimators = n_estimators
        self.early_stopping_rounds = early_stopping_rounds
        self.random_state = random_state
        self.model_: Any = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> "LightGBMRegressor":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        train_set = lgb.Dataset(X, label=y)
        valid_sets = [train_set]
        callbacks = []
        if X_val is not None and y_val is not None:
            valid_sets.append(lgb.Dataset(np.asarray(X_val, dtype=float), label=np.asarray(y_val, dtype=float)))
            callbacks.append(lgb.early_stopping(self.early_stopping_rounds, verbose=False))
        self.model_ = lgb.train(
            self.params,
            train_set,
            num_boost_round=self.n_estimators,
            valid_sets=valid_sets,
            callbacks=callbacks or None,
        )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("Model is not fitted.")
        return self.model_.predict(np.asarray(X, dtype=float))

    @property
    def feature_importances_(self) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("Model is not fitted.")
        return self.model_.feature_importance(importance_type="gain")
