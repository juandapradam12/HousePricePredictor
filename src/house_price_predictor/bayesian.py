"""Conjugate Gaussian Bayesian linear regression (analytical posterior)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .least_squares import add_intercept, least_squares_weights


@dataclass
class BayesianPosterior:
    """Posterior ``N(mu, Sigma)`` for the weight vector (incl. intercept)."""

    mu: np.ndarray
    Sigma: np.ndarray
    sigma2: float
    lambda_param: float


def estimate_noise_variance(X_aug: np.ndarray, y: np.ndarray, weights: np.ndarray) -> float:
    """
    Unbiased residual variance estimate:

    ``σ² ≈ 1/(n - d) Σ (y_i - x_i^T w)^2``
    """
    n, d = X_aug.shape
    resid = y - X_aug @ weights
    denom = max(n - d, 1)
    return float(np.sum(resid**2) / denom)


def map_coefficients(
    X_aug: np.ndarray,
    y: np.ndarray,
    lambda_param: float,
    sigma2: float,
) -> np.ndarray:
    """
    MAP mean under prior ``w ~ N(0, λ^{-1} I)`` and likelihood noise ``σ²``:

    ``μ = (λ σ² I + X^T X)^{-1} X^T y``

    When ``lambda_param == 0`` this reduces to ordinary least squares.
    """
    X_aug = np.asarray(X_aug, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    d = X_aug.shape[1]

    if lambda_param == 0:
        weights, *_ = np.linalg.lstsq(X_aug, y, rcond=None)
        return weights

    penalty = float(lambda_param) * float(sigma2) * np.eye(d)
    # Leave intercept lightly constrained: still regularized for conjugacy,
    # which matches the classic textbook derivation used in this project.
    xtx = X_aug.T @ X_aug
    xty = X_aug.T @ y
    return np.linalg.solve(xtx + penalty, xty)


def posterior_covariance(
    X_aug: np.ndarray,
    sigma2: float,
    lambda_param: float,
) -> np.ndarray:
    """
    Posterior covariance:

    ``Σ = (λ I + σ^{-2} X^T X)^{-1}``
    """
    d = X_aug.shape[1]
    precision = float(lambda_param) * np.eye(d) + (1.0 / float(sigma2)) * (X_aug.T @ X_aug)
    return np.linalg.inv(precision)


def predictive_moments(
    x_aug: np.ndarray,
    mu: np.ndarray,
    Sigma: np.ndarray,
    sigma2: float,
) -> Tuple[float, float]:
    """
    Predictive distribution for a single observation:

    ``p(y0 | x0) = N(μ0, σ0²)`` with
    ``μ0 = x0^T μ`` and ``σ0² = σ² + x0^T Σ x0``.
    """
    x_aug = np.asarray(x_aug, dtype=float).reshape(-1)
    mu0 = float(x_aug @ mu)
    sigma0 = float(sigma2 + x_aug @ Sigma @ x_aug)
    return mu0, sigma0


class BayesianLinearRegression:
    """
    Analytical conjugate Bayesian linear regression.

    Prior: ``w ~ N(0, λ^{-1} I)``
    Likelihood: ``y | X, w ~ N(X w, σ² I)`` with ``σ²`` estimated from OLS residuals.
    """

    def __init__(self, lambda_param: float = 0.1) -> None:
        self.lambda_param = float(lambda_param)
        self.posterior_: Optional[BayesianPosterior] = None
        self.feature_mean_: Optional[np.ndarray] = None
        self.feature_std_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "BayesianLinearRegression":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        # Standardize features for a sensible isotropic Gaussian prior
        self.feature_mean_ = X.mean(axis=0)
        self.feature_std_ = X.std(axis=0)
        self.feature_std_ = np.where(self.feature_std_ == 0, 1.0, self.feature_std_)
        X_s = (X - self.feature_mean_) / self.feature_std_
        X_aug = add_intercept(X_s)

        ols = least_squares_weights(X_s, y)
        sigma2 = estimate_noise_variance(X_aug, y, ols)
        mu = map_coefficients(X_aug, y, self.lambda_param, sigma2)
        Sigma = posterior_covariance(X_aug, sigma2, self.lambda_param)

        self.posterior_ = BayesianPosterior(
            mu=mu, Sigma=Sigma, sigma2=sigma2, lambda_param=self.lambda_param
        )
        return self

    def predict(self, X: np.ndarray, return_std: bool = False):
        if self.posterior_ is None:
            raise RuntimeError("Model is not fitted.")
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        X_s = (X - self.feature_mean_) / self.feature_std_
        X_aug = add_intercept(X_s)

        means = X_aug @ self.posterior_.mu
        if not return_std:
            return means

        # predictive variance for each row
        # σ0² = σ² + x^T Σ x  = σ² + sum((x @ Σ) * x)
        quad = np.einsum("ij,jk,ik->i", X_aug, self.posterior_.Sigma, X_aug)
        std = np.sqrt(self.posterior_.sigma2 + quad)
        return means, std

    @property
    def intercept_(self) -> float:
        if self.posterior_ is None:
            raise RuntimeError("Model is not fitted.")
        return float(self.posterior_.mu[0])

    @property
    def coef_(self) -> np.ndarray:
        if self.posterior_ is None:
            raise RuntimeError("Model is not fitted.")
        return self.posterior_.mu[1:]
