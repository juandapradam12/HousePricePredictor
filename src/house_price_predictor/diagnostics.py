"""Residual diagnostics helpers for linear models."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


def residuals(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return np.asarray(y_true, dtype=float).reshape(-1) - np.asarray(y_pred, dtype=float).reshape(-1)


def leverage_approx(X: np.ndarray) -> np.ndarray:
    """
    Diagonal of the hat matrix ``H = X(X'X)^{-1}X'`` using a QR-based approach
    on an intercept-augmented design (numerically stabler than forming H).
    """
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    ones = np.ones((X.shape[0], 1))
    aug = np.concatenate([ones, X], axis=1)
    q, _ = np.linalg.qr(aug, mode="reduced")
    return np.sum(q * q, axis=1)


def standardized_residuals(y_true: np.ndarray, y_pred: np.ndarray, X: np.ndarray) -> np.ndarray:
    r = residuals(y_true, y_pred)
    h = leverage_approx(X)
    n, p = X.shape[0], X.shape[1] + 1
    mse = float(np.sum(r**2) / max(n - p, 1))
    denom = np.sqrt(np.maximum(mse * (1.0 - h), 1e-12))
    return r / denom


def cook_distance(y_true: np.ndarray, y_pred: np.ndarray, X: np.ndarray) -> np.ndarray:
    r_std = standardized_residuals(y_true, y_pred, X)
    h = leverage_approx(X)
    p = X.shape[1] + 1
    return (r_std**2) * h / (p * np.maximum(1.0 - h, 1e-12))


def learning_curve_rmse(
    model_factory,
    X: np.ndarray,
    y: np.ndarray,
    train_sizes: np.ndarray | None = None,
    val_fraction: float = 0.2,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fit on increasing subsets; return ``sizes, train_rmse, val_rmse``.

    ``model_factory`` must be a zero-arg callable returning an object with fit/predict.
    """
    from .metrics import rmse
    from .data import train_val_split

    X_tr, X_va, y_tr, y_va = train_val_split(X, y, val_size=val_fraction, random_state=random_state)
    n = X_tr.shape[0]
    if train_sizes is None:
        train_sizes = np.linspace(0.2, 1.0, 5)
    sizes = []
    train_scores = []
    val_scores = []
    for frac in train_sizes:
        m = max(10, int(frac * n))
        model = model_factory()
        model.fit(X_tr[:m], y_tr[:m])
        sizes.append(m)
        train_scores.append(rmse(y_tr[:m], model.predict(X_tr[:m])))
        val_scores.append(rmse(y_va, model.predict(X_va)))
    return np.asarray(sizes), np.asarray(train_scores), np.asarray(val_scores)


def diagnostics_summary(y_true: np.ndarray, y_pred: np.ndarray, X: np.ndarray) -> Dict[str, float]:
    r = residuals(y_true, y_pred)
    cooks = cook_distance(y_true, y_pred, X)
    return {
        "residual_mean": float(np.mean(r)),
        "residual_std": float(np.std(r)),
        "max_abs_residual": float(np.max(np.abs(r))),
        "max_cooks_distance": float(np.max(cooks)),
        "n_high_leverage": int(np.sum(leverage_approx(X) > 2 * (X.shape[1] + 1) / X.shape[0])),
    }
