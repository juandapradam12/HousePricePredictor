"""Optional PyMC MCMC helpers for Bayesian linear regression."""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np


def fit_bayesian_mcmc(
    X: np.ndarray,
    y: np.ndarray,
    draws: int = 1000,
    tune: int = 1000,
    chains: int = 2,
    random_seed: int = 42,
    target_accept: float = 0.9,
) -> Tuple[Any, Any]:
    """
    Fit a Bayesian linear model with PyMC (NUTS).

    Features are standardized inside the model context for stable sampling.
    Returns ``(idata, model)`` where ``idata`` is an ArviZ InferenceData object.
    """
    import pymc as pm

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X_std = np.where(X_std == 0, 1.0, X_std)
    X_s = (X - X_mean) / X_std
    n_features = X_s.shape[1]

    with pm.Model() as model:
        intercept = pm.Normal("intercept", mu=float(np.mean(y)), sigma=10.0)
        beta = pm.Normal("beta", mu=0.0, sigma=1.0, shape=n_features)
        sigma = pm.HalfNormal("sigma", sigma=1.0)

        mu = intercept + pm.math.dot(X_s, beta)
        pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y)

        idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            target_accept=target_accept,
            random_seed=random_seed,
            progressbar=False,
        )

    # stash preprocessing for prediction helpers
    model.X_mean_ = X_mean  # type: ignore[attr-defined]
    model.X_std_ = X_std  # type: ignore[attr-defined]
    return idata, model


def posterior_predictive_mean(idata: Any) -> Dict[str, float]:
    """Summarize posterior means for intercept, betas, and sigma."""
    posterior = idata.posterior
    summary = {
        "intercept": float(posterior["intercept"].mean()),
        "sigma": float(posterior["sigma"].mean()),
    }
    beta_mean = posterior["beta"].mean(dim=("chain", "draw")).values
    for i, value in enumerate(np.atleast_1d(beta_mean)):
        summary[f"beta[{i}]"] = float(value)
    return summary


def generate_synthetic_regression(
    n_obs: int = 100,
    intercept: float = 20.0,
    betas: Optional[Sequence[float]] = None,
    sigma: float = 0.5,
    seed: int = 1738,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """Create a small linear synthetic dataset for MCMC recovery checks."""
    betas = list(betas) if betas is not None else [0.2, 0.4]
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_obs, len(betas)))
    y = intercept + X @ np.asarray(betas) + rng.standard_normal(n_obs) * sigma
    truth = {"intercept": intercept, "sigma": sigma}
    for i, b in enumerate(betas):
        truth[f"beta[{i}]"] = float(b)
    return X, y, truth
