"""Elastic Net / Lasso via coordinate descent (sklearn-backed with project API)."""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np
from sklearn.linear_model import ElasticNet as SkElasticNet
from sklearn.linear_model import ElasticNetCV, Lasso, LassoCV
from sklearn.preprocessing import StandardScaler


class ElasticNetRegressor:
    """
    Elastic Net: ``(1/2n)||y - Xw||^2 + α * l1_ratio * ||w||1 + α * (1-l1_ratio)/2 * ||w||^2``.

    Features are standardized internally. Use ``l1_ratio=1`` for Lasso.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        l1_ratio: float = 0.5,
        max_iter: int = 10000,
        random_state: int = 42,
    ) -> None:
        self.alpha = float(alpha)
        self.l1_ratio = float(l1_ratio)
        self.max_iter = max_iter
        self.random_state = random_state
        self.scaler_: Optional[StandardScaler] = None
        self.model_: Optional[SkElasticNet] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ElasticNetRegressor":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        self.scaler_ = StandardScaler()
        Xs = self.scaler_.fit_transform(X)
        self.model_ = SkElasticNet(
            alpha=self.alpha,
            l1_ratio=self.l1_ratio,
            max_iter=self.max_iter,
            random_state=self.random_state,
        )
        self.model_.fit(Xs, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model_ is None or self.scaler_ is None:
            raise RuntimeError("Model is not fitted.")
        Xs = self.scaler_.transform(np.asarray(X, dtype=float))
        return self.model_.predict(Xs)

    @property
    def coef_(self) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("Model is not fitted.")
        return self.model_.coef_

    @property
    def intercept_(self) -> float:
        if self.model_ is None:
            raise RuntimeError("Model is not fitted.")
        return float(self.model_.intercept_)

    @property
    def n_nonzero_(self) -> int:
        return int(np.sum(np.abs(self.coef_) > 1e-12))


class LassoRegressor(ElasticNetRegressor):
    """Lasso = Elastic Net with ``l1_ratio=1``."""

    def __init__(self, alpha: float = 1.0, max_iter: int = 10000, random_state: int = 42) -> None:
        super().__init__(alpha=alpha, l1_ratio=1.0, max_iter=max_iter, random_state=random_state)


def select_elastic_net_cv(
    X: np.ndarray,
    y: np.ndarray,
    l1_ratio: float | Sequence[float] = (0.1, 0.5, 0.7, 0.9, 0.95, 1.0),
    n_alphas: int = 40,
    cv: int = 5,
    random_state: int = 42,
) -> Tuple[ElasticNetRegressor, float, float]:
    """
    Cross-validate Elastic Net hyperparameters.

    Returns fitted ``ElasticNetRegressor``, best ``alpha``, best ``l1_ratio``.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    search = ElasticNetCV(
        l1_ratio=l1_ratio,
        alphas=n_alphas,
        cv=cv,
        max_iter=20000,
        random_state=random_state,
        n_jobs=None,
    )
    search.fit(Xs, y)
    model = ElasticNetRegressor(alpha=float(search.alpha_), l1_ratio=float(search.l1_ratio_))
    model.scaler_ = scaler
    model.model_ = SkElasticNet(
        alpha=float(search.alpha_),
        l1_ratio=float(search.l1_ratio_),
        max_iter=20000,
        random_state=random_state,
    )
    model.model_.coef_ = search.coef_
    model.model_.intercept_ = search.intercept_
    model.model_.n_features_in_ = search.n_features_in_
    return model, float(search.alpha_), float(search.l1_ratio_)


def select_lasso_cv(
    X: np.ndarray,
    y: np.ndarray,
    n_alphas: int = 40,
    cv: int = 5,
    random_state: int = 42,
) -> Tuple[LassoRegressor, float]:
    """Cross-validate Lasso ``alpha``; returns fitted model and best alpha."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    search = LassoCV(alphas=n_alphas, cv=cv, max_iter=20000, random_state=random_state)
    search.fit(Xs, y)
    model = LassoRegressor(alpha=float(search.alpha_))
    model.scaler_ = scaler
    wrapped = Lasso(alpha=float(search.alpha_), max_iter=20000, random_state=random_state)
    wrapped.coef_ = search.coef_
    wrapped.intercept_ = search.intercept_
    wrapped.n_features_in_ = search.n_features_in_
    # store as ElasticNet-compatible object for predict path
    model.model_ = SkElasticNet(alpha=float(search.alpha_), l1_ratio=1.0, max_iter=20000)
    model.model_.coef_ = search.coef_
    model.model_.intercept_ = search.intercept_
    model.model_.n_features_in_ = search.n_features_in_
    return model, float(search.alpha_)
