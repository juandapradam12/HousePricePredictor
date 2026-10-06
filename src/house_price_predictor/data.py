"""Data loading, feature selection, and preprocessing helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# Repo root is two levels above this file: src/house_price_predictor/data.py
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"

# Strong linear correlates of SalePrice on the Ames train set (Pearson |r| >= 0.5)
DEFAULT_FEATURES: List[str] = [
    "OverallQual",
    "GrLivArea",
    "GarageCars",
    "GarageArea",
    "TotalBsmtSF",
    "1stFlrSF",
    "FullBath",
    "TotRmsAbvGrd",
    "YearBuilt",
    "YearRemodAdd",
]


def load_housing_data(
    train_path: Optional[Path | str] = None,
    test_path: Optional[Path | str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load Ames Housing train/test CSVs from ``data/`` by default."""
    train_path = Path(train_path) if train_path else DATA_DIR / "train.csv"
    test_path = Path(test_path) if test_path else DATA_DIR / "test.csv"
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    return train, test


def correlation_with_target(
    frame: pd.DataFrame,
    target: str = "SalePrice",
) -> pd.Series:
    """Return Pearson correlations of numeric columns with ``target``."""
    numeric = frame.select_dtypes(include=[np.number])
    if target not in numeric.columns:
        raise KeyError(f"Target '{target}' not found among numeric columns.")
    return numeric.corr()[target].drop(target).sort_values(ascending=False)


def select_correlated_features(
    frame: pd.DataFrame,
    target: str = "SalePrice",
    threshold: float = 0.5,
) -> List[str]:
    """Select features whose |corr| with ``target`` is at least ``threshold``."""
    corr = correlation_with_target(frame, target=target)
    return corr[corr.abs() >= threshold].index.tolist()


def apply_cutoffs(
    frame: pd.DataFrame,
    cutoffs: Sequence[Tuple[str, float, float]],
) -> pd.DataFrame:
    """Keep rows where each listed column stays within ``[low, high]``."""
    out = frame.copy()
    for column, low, high in cutoffs:
        out = out.loc[(out[column] >= low) & (out[column] <= high)]
    return out.reset_index(drop=True)


def prepare_xy(
    frame: pd.DataFrame,
    features: Optional[Sequence[str]] = None,
    target: str = "SalePrice",
    log_target: bool = True,
    dropna: bool = True,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Build design matrix ``X`` and target ``y``.

    Parameters
    ----------
    log_target:
        If True, model ``log1p(SalePrice)`` — standard for skewed house prices.
    """
    features = list(features) if features is not None else list(DEFAULT_FEATURES)
    cols = list(features) + [target]
    subset = frame[cols].copy()
    if dropna:
        subset = subset.dropna()

    X = subset[features].to_numpy(dtype=float)
    y_raw = subset[target].to_numpy(dtype=float)
    y = np.log1p(y_raw) if log_target else y_raw
    return X, y, features


def train_val_split(
    X: np.ndarray,
    y: np.ndarray,
    val_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Simple reproducible hold-out split without sklearn dependency in callers."""
    rng = np.random.default_rng(random_state)
    n = X.shape[0]
    idx = rng.permutation(n)
    n_val = max(1, int(round(n * val_size)))
    val_idx, train_idx = idx[:n_val], idx[n_val:]
    return X[train_idx], X[val_idx], y[train_idx], y[val_idx]


def standardize(
    X_train: np.ndarray,
    X_other: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray, np.ndarray]:
    """
    Z-score features using training statistics only.

    Returns
    -------
    X_train_std, X_other_std (or None), mean, std
    """
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std = np.where(std == 0, 1.0, std)
    X_train_std = (X_train - mean) / std
    X_other_std = None if X_other is None else (X_other - mean) / std
    return X_train_std, X_other_std, mean, std


def center_target(
    y_train: np.ndarray,
    y_other: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray], float]:
    """Mean-center the target using the training mean."""
    y_mean = float(y_train.mean())
    y_train_c = y_train - y_mean
    y_other_c = None if y_other is None else y_other - y_mean
    return y_train_c, y_other_c, y_mean
