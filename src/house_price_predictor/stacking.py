"""Stacking ensemble: Ridge + LightGBM → linear meta-learner on OOF predictions."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.linear_model import Ridge

from .boosting import HAS_LIGHTGBM, LightGBMRegressor
from .metrics import rmse
from .ridge import RidgeRegressor


def _kfold_indices(n: int, n_splits: int, rng: np.random.Generator) -> list:
    idx = rng.permutation(n)
    folds = np.array_split(idx, n_splits)
    return [
        (np.concatenate([folds[j] for j in range(n_splits) if j != i]), folds[i])
        for i in range(n_splits)
    ]


class StackingRegressor:
    """
    Two-level stack.

    Base models (default): RidgeRegressor + LightGBMRegressor.
    Meta model: sklearn Ridge on out-of-fold base predictions.
    """

    def __init__(
        self,
        ridge_alpha: float = 10.0,
        lgbm_params: Optional[Dict[str, Any]] = None,
        meta_alpha: float = 1.0,
        n_splits: int = 5,
        random_state: int = 42,
        n_estimators: int = 800,
    ) -> None:
        if not HAS_LIGHTGBM:
            raise ImportError("lightgbm is required for StackingRegressor")
        self.ridge_alpha = ridge_alpha
        self.lgbm_params = lgbm_params
        self.meta_alpha = meta_alpha
        self.n_splits = n_splits
        self.random_state = random_state
        self.n_estimators = n_estimators
        self.base_ridge_: Optional[RidgeRegressor] = None
        self.base_lgbm_: Optional[LightGBMRegressor] = None
        self.meta_: Optional[Ridge] = None
        self.oof_rmse_: Optional[float] = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> "StackingRegressor":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        rng = np.random.default_rng(self.random_state)
        splits = _kfold_indices(X.shape[0], self.n_splits, rng)

        oof = np.zeros((X.shape[0], 2), dtype=float)
        for tr_idx, va_idx in splits:
            ridge = RidgeRegressor(alpha=self.ridge_alpha).fit(X[tr_idx], y[tr_idx])
            lgbm = LightGBMRegressor(
                params=self.lgbm_params,
                n_estimators=self.n_estimators,
                early_stopping_rounds=50,
                random_state=self.random_state,
            )
            # tiny ES split inside fold train
            n_tr = len(tr_idx)
            cut = max(1, int(0.15 * n_tr))
            inner = rng.permutation(n_tr)
            es_va = tr_idx[inner[:cut]]
            es_tr = tr_idx[inner[cut:]]
            lgbm.fit(X[es_tr], y[es_tr], X_val=X[es_va], y_val=y[es_va])
            oof[va_idx, 0] = ridge.predict(X[va_idx])
            oof[va_idx, 1] = lgbm.predict(X[va_idx])

        self.meta_ = Ridge(alpha=self.meta_alpha)
        self.meta_.fit(oof, y)
        self.oof_rmse_ = rmse(y, self.meta_.predict(oof))

        # Refit bases on all data
        self.base_ridge_ = RidgeRegressor(alpha=self.ridge_alpha).fit(X, y)
        self.base_lgbm_ = LightGBMRegressor(
            params=self.lgbm_params,
            n_estimators=self.n_estimators,
            early_stopping_rounds=60,
            random_state=self.random_state,
        )
        if X_val is not None and y_val is not None:
            self.base_lgbm_.fit(X, y, X_val=X_val, y_val=y_val)
        else:
            n = X.shape[0]
            cut = max(1, int(0.15 * n))
            perm = rng.permutation(n)
            self.base_lgbm_.fit(X[perm[cut:]], y[perm[cut:]], X_val=X[perm[:cut]], y_val=y[perm[:cut]])
        return self

    def _base_matrix(self, X: np.ndarray) -> np.ndarray:
        assert self.base_ridge_ is not None and self.base_lgbm_ is not None
        return np.column_stack(
            [self.base_ridge_.predict(X), self.base_lgbm_.predict(X)]
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.meta_ is None:
            raise RuntimeError("Model is not fitted.")
        return self.meta_.predict(self._base_matrix(np.asarray(X, dtype=float)))

    @property
    def meta_weights_(self) -> np.ndarray:
        if self.meta_ is None:
            raise RuntimeError("Model is not fitted.")
        return self.meta_.coef_
