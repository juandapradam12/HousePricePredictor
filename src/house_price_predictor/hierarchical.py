"""Hierarchical Bayesian linear regression with neighborhood partial pooling (PyMC)."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd


def fit_hierarchical_neighborhood(
    X: np.ndarray,
    y: np.ndarray,
    neighborhood: np.ndarray,
    draws: int = 800,
    tune: int = 800,
    chains: int = 2,
    random_seed: int = 42,
    target_accept: float = 0.9,
) -> Tuple[Any, Any, Dict[str, np.ndarray]]:
    """
    Varying-intercept model:

    ``y_i ~ N(α_{neigh[i]} + x_i^T β, σ)``
    ``α_j ~ N(μ_α, τ_α)``  (partial pooling across neighborhoods)

    Features in ``X`` should already be numeric (engineered). They are standardized
    inside the model for stable sampling.
    """
    import pymc as pm

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    neigh = pd.Series(neighborhood).astype(str)
    levels, codes = np.unique(neigh, return_inverse=True)
    n_groups = len(levels)
    n_features = X.shape[1]

    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X_std = np.where(X_std == 0, 1.0, X_std)
    X_s = (X - X_mean) / X_std

    with pm.Model() as model:
        mu_alpha = pm.Normal("mu_alpha", mu=float(np.mean(y)), sigma=2.0)
        tau_alpha = pm.HalfNormal("tau_alpha", sigma=1.0)
        alpha = pm.Normal("alpha", mu=mu_alpha, sigma=tau_alpha, shape=n_groups)

        beta = pm.Normal("beta", mu=0.0, sigma=1.0, shape=n_features)
        sigma = pm.HalfNormal("sigma", sigma=1.0)

        mu = alpha[codes] + pm.math.dot(X_s, beta)
        pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y)

        idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            target_accept=target_accept,
            random_seed=random_seed,
            progressbar=False,
        )

    meta = {
        "levels": levels,
        "X_mean": X_mean,
        "X_std": X_std,
        "codes": codes,
    }
    model.meta_ = meta  # type: ignore[attr-defined]
    return idata, model, meta


def hierarchical_predict_mean(
    idata: Any,
    X: np.ndarray,
    neighborhood: np.ndarray,
    meta: Dict[str, np.ndarray],
) -> np.ndarray:
    """Posterior-mean predictions for new rows (unseen neighborhoods → μ_α)."""
    X = np.asarray(X, dtype=float)
    X_s = (X - meta["X_mean"]) / meta["X_std"]
    post = idata.posterior
    beta = post["beta"].mean(dim=("chain", "draw")).values
    mu_alpha = float(post["mu_alpha"].mean())
    alpha = post["alpha"].mean(dim=("chain", "draw")).values
    level_to_idx = {str(lvl): i for i, lvl in enumerate(meta["levels"])}

    intercepts = np.array(
        [alpha[level_to_idx[n]] if str(n) in level_to_idx else mu_alpha for n in np.asarray(neighborhood).astype(str)]
    )
    return intercepts + X_s @ beta
