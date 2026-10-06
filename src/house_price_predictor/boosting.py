"""Gradient boosting baseline (LightGBM) with Optuna tuning and quantile models."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import lightgbm as lgb

    HAS_LIGHTGBM = True
except ImportError:  # pragma: no cover
    lgb = None  # type: ignore
    HAS_LIGHTGBM = False

try:
    import optuna

    HAS_OPTUNA = True
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ImportError:  # pragma: no cover
    optuna = None  # type: ignore
    HAS_OPTUNA = False


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
        self.best_iteration_: Optional[int] = None

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
        callbacks: List[Any] = []
        if X_val is not None and y_val is not None:
            valid_sets.append(
                lgb.Dataset(np.asarray(X_val, dtype=float), label=np.asarray(y_val, dtype=float))
            )
            callbacks.append(lgb.early_stopping(self.early_stopping_rounds, verbose=False))
        self.model_ = lgb.train(
            self.params,
            train_set,
            num_boost_round=self.n_estimators,
            valid_sets=valid_sets,
            callbacks=callbacks or None,
        )
        self.best_iteration_ = getattr(self.model_, "best_iteration", None) or self.n_estimators
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("Model is not fitted.")
        kwargs = {}
        if self.best_iteration_:
            kwargs["num_iteration"] = self.best_iteration_
        return self.model_.predict(np.asarray(X, dtype=float), **kwargs)

    @property
    def feature_importances_(self) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("Model is not fitted.")
        return self.model_.feature_importance(importance_type="gain")

    @property
    def booster_(self) -> Any:
        if self.model_ is None:
            raise RuntimeError("Model is not fitted.")
        return self.model_


def _kfold_indices(n: int, n_splits: int, rng: np.random.Generator) -> list:
    idx = rng.permutation(n)
    folds = np.array_split(idx, n_splits)
    return [
        (np.concatenate([folds[j] for j in range(n_splits) if j != i]), folds[i])
        for i in range(n_splits)
    ]


def tune_lightgbm_optuna(
    X: np.ndarray,
    y: np.ndarray,
    n_trials: int = 30,
    n_splits: int = 3,
    timeout: Optional[float] = 120.0,
    random_state: int = 42,
    n_estimators: int = 1500,
) -> Tuple[LightGBMRegressor, Dict[str, Any], float]:
    """
    Optuna search minimizing k-fold RMSE on the log target.

    Returns a model fitted on the full ``(X, y)`` with the best params,
    the best param dict, and the best CV RMSE.
    """
    if not HAS_LIGHTGBM:
        raise ImportError("lightgbm is required")
    if not HAS_OPTUNA:
        raise ImportError("optuna is required for tune_lightgbm_optuna. pip install optuna")

    from .metrics import rmse

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    rng = np.random.default_rng(random_state)
    splits = _kfold_indices(X.shape[0], n_splits, rng)

    def objective(trial: "optuna.Trial") -> float:
        params = {
            **DEFAULT_LGBM_PARAMS,
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 16, 96),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 5, 40),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 10.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 5.0, log=True),
            "seed": random_state,
        }
        fold_scores = []
        for tr_idx, va_idx in splits:
            model = LightGBMRegressor(params=params, n_estimators=n_estimators, early_stopping_rounds=60)
            model.fit(X[tr_idx], y[tr_idx], X_val=X[va_idx], y_val=y[va_idx])
            fold_scores.append(rmse(y[va_idx], model.predict(X[va_idx])))
        return float(np.mean(fold_scores))

    sampler = optuna.samplers.TPESampler(seed=random_state)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=False)

    best_params = {**DEFAULT_LGBM_PARAMS, **study.best_params, "seed": random_state}
    # Refit on all data with a small holdout for early stopping
    n = X.shape[0]
    cut = max(1, int(0.15 * n))
    perm = rng.permutation(n)
    va_idx, tr_idx = perm[:cut], perm[cut:]
    model = LightGBMRegressor(params=best_params, n_estimators=n_estimators, early_stopping_rounds=80)
    model.fit(X[tr_idx], y[tr_idx], X_val=X[va_idx], y_val=y[va_idx])
    return model, best_params, float(study.best_value)


class QuantileLightGBM:
    """Fit lower / median / upper LightGBM quantile models for prediction intervals."""

    def __init__(
        self,
        quantiles: Tuple[float, float, float] = (0.05, 0.5, 0.95),
        n_estimators: int = 800,
        random_state: int = 42,
        params: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not HAS_LIGHTGBM:
            raise ImportError("lightgbm is required")
        self.quantiles = quantiles
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.base_params = params or {}
        self.models_: Dict[float, LightGBMRegressor] = {}

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> "QuantileLightGBM":
        for q in self.quantiles:
            params = {
                **DEFAULT_LGBM_PARAMS,
                **self.base_params,
                "objective": "quantile",
                "alpha": q,
                "metric": "quantile",
            }
            model = LightGBMRegressor(
                params=params,
                n_estimators=self.n_estimators,
                early_stopping_rounds=60,
                random_state=self.random_state,
            )
            model.fit(X, y, X_val=X_val, y_val=y_val)
            self.models_[q] = model
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Point prediction = median quantile."""
        mid = self.quantiles[1]
        return self.models_[mid].predict(X)

    def predict_intervals(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        lo, mid, hi = self.quantiles
        lower = self.models_[lo].predict(X)
        median = self.models_[mid].predict(X)
        upper = self.models_[hi].predict(X)
        # enforce ordering
        lower = np.minimum(lower, median)
        upper = np.maximum(upper, median)
        return lower, median, upper
