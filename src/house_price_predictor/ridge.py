"""Ridge (L2-regularized) linear regression with cross-validated lambda."""

from __future__ import annotations

from typing import Iterable, Optional, Sequence, Tuple

import numpy as np

from .least_squares import add_intercept
from .metrics import rmse


def ridge_regression_weights(
    X: np.ndarray,
    y: np.ndarray,
    lambda_param: float,
    penalize_intercept: bool = False,
) -> np.ndarray:
    """
    Closed-form ridge solution.

    With standardized features and a mean-centered target, the usual estimator is
    ``w = (X^T X + λ I)^{-1} X^T y``.  When an intercept column is present we
    leave it unpenalized by default (``penalize_intercept=False``), which matches
    scikit-learn's ``Ridge``.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    aug = add_intercept(X)
    n_features = aug.shape[1]
    penalty = np.eye(n_features, dtype=float) * float(lambda_param)
    if not penalize_intercept:
        penalty[0, 0] = 0.0

    # Solve (X^T X + λ I) w = X^T y  — more stable than explicit inverse
    xtx = aug.T @ aug
    xty = aug.T @ y
    return np.linalg.solve(xtx + penalty, xty)


def _kfold_indices(n: int, n_splits: int, rng: np.random.Generator) -> list:
    idx = rng.permutation(n)
    folds = np.array_split(idx, n_splits)
    splits = []
    for i in range(n_splits):
        val = folds[i]
        train = np.concatenate([folds[j] for j in range(n_splits) if j != i])
        splits.append((train, val))
    return splits


def select_lambda_cv(
    X: np.ndarray,
    y: np.ndarray,
    lambdas: Optional[Sequence[float]] = None,
    n_splits: int = 5,
    random_state: int = 42,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Choose λ by k-fold CV minimizing RMSE on the validation folds.

    Features are re-standardized within each fold using training-fold stats only.
    Target is mean-centered within each fold.

    Returns
    -------
    best_lambda, lambdas_grid, mean_cv_rmse
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    if lambdas is None:
        lambdas = np.logspace(-3, 3, 40)

    lambdas = np.asarray(list(lambdas), dtype=float)
    rng = np.random.default_rng(random_state)
    splits = _kfold_indices(X.shape[0], n_splits, rng)
    mean_scores = np.zeros(len(lambdas), dtype=float)

    for i, lam in enumerate(lambdas):
        fold_errors = []
        for train_idx, val_idx in splits:
            X_tr, X_va = X[train_idx], X[val_idx]
            y_tr, y_va = y[train_idx], y[val_idx]

            mean = X_tr.mean(axis=0)
            std = X_tr.std(axis=0)
            std = np.where(std == 0, 1.0, std)
            X_tr_s = (X_tr - mean) / std
            X_va_s = (X_va - mean) / std

            y_mean = y_tr.mean()
            y_tr_c = y_tr - y_mean

            w = ridge_regression_weights(X_tr_s, y_tr_c, lam)
            pred = add_intercept(X_va_s) @ w + y_mean
            fold_errors.append(rmse(y_va, pred))
        mean_scores[i] = float(np.mean(fold_errors))

    best = float(lambdas[int(np.argmin(mean_scores))])
    return best, lambdas, mean_scores


class RidgeRegressor:
    """
    Ridge regressor that standardizes features and mean-centers the target
    internally, then maps predictions back to the original target scale.
    """

    def __init__(self, alpha: float = 1.0, penalize_intercept: bool = False) -> None:
        self.alpha = float(alpha)
        self.penalize_intercept = penalize_intercept
        self.weights_: Optional[np.ndarray] = None
        self.feature_mean_: Optional[np.ndarray] = None
        self.feature_std_: Optional[np.ndarray] = None
        self.target_mean_: Optional[float] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RidgeRegressor":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        self.feature_mean_ = X.mean(axis=0)
        self.feature_std_ = X.std(axis=0)
        self.feature_std_ = np.where(self.feature_std_ == 0, 1.0, self.feature_std_)
        self.target_mean_ = float(y.mean())

        X_s = (X - self.feature_mean_) / self.feature_std_
        y_c = y - self.target_mean_
        self.weights_ = ridge_regression_weights(
            X_s, y_c, self.alpha, penalize_intercept=self.penalize_intercept
        )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.weights_ is None:
            raise RuntimeError("Model is not fitted.")
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        X_s = (X - self.feature_mean_) / self.feature_std_
        return add_intercept(X_s) @ self.weights_ + self.target_mean_

    @property
    def intercept_(self) -> float:
        if self.weights_ is None or self.target_mean_ is None:
            raise RuntimeError("Model is not fitted.")
        return float(self.weights_[0] + self.target_mean_)

    @property
    def coef_(self) -> np.ndarray:
        if self.weights_ is None:
            raise RuntimeError("Model is not fitted.")
        # Coefficients on the standardized feature scale
        return self.weights_[1:]
