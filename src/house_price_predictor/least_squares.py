"""Ordinary least squares — closed form and estimator class."""

from __future__ import annotations

from typing import Optional

import numpy as np


def add_intercept(X: np.ndarray) -> np.ndarray:
    """Prepend a column of ones."""
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    ones = np.ones((X.shape[0], 1), dtype=float)
    return np.concatenate([ones, X], axis=1)


def least_squares_weights(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Solve ``w_LS = argmin ||Xw - y||^2`` via a numerically stable QR/LSTSQ solve.

    Equivalent to the textbook formula ``(X^T X)^{-1} X^T y`` when ``X`` has
    full column rank, but avoids an explicit matrix inverse.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if X.shape[0] != y.shape[0]:
        raise ValueError(
            f"X and y length mismatch: X has {X.shape[0]} rows, y has {y.shape[0]}."
        )

    aug = add_intercept(X)
    weights, *_ = np.linalg.lstsq(aug, y, rcond=None)
    return weights


class LeastSquaresRegressor:
    """Simple OLS regressor with optional intercept (always included)."""

    def __init__(self) -> None:
        self.weights_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LeastSquaresRegressor":
        self.weights_ = least_squares_weights(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.weights_ is None:
            raise RuntimeError("Model is not fitted.")
        aug = add_intercept(X)
        return aug @ self.weights_

    @property
    def intercept_(self) -> float:
        if self.weights_ is None:
            raise RuntimeError("Model is not fitted.")
        return float(self.weights_[0])

    @property
    def coef_(self) -> np.ndarray:
        if self.weights_ is None:
            raise RuntimeError("Model is not fitted.")
        return self.weights_[1:]
