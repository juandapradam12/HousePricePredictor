"""Prediction-interval calibration utilities."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np


def empirical_coverage(
    y_true: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    z: float = 1.96,
) -> float:
    """Fraction of targets falling inside ``mean ± z * std``."""
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    mean = np.asarray(mean, dtype=float).reshape(-1)
    std = np.asarray(std, dtype=float).reshape(-1)
    lower = mean - z * std
    upper = mean + z * std
    return float(np.mean((y_true >= lower) & (y_true <= upper)))


def calibration_curve(
    y_true: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    levels: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compare nominal coverage levels to empirical coverage.

    Parameters
    ----------
    levels:
        Nominal probabilities in (0, 1), default ``0.1 … 0.9``.

    Returns
    -------
    nominal, empirical
    """
    from scipy import stats  # scipy comes with pymc/sklearn stack

    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    mean = np.asarray(mean, dtype=float).reshape(-1)
    std = np.asarray(std, dtype=float).reshape(-1)
    if levels is None:
        levels = np.linspace(0.1, 0.9, 9)
    levels = np.asarray(levels, dtype=float)
    empirical = []
    for p in levels:
        z = float(stats.norm.ppf(0.5 + p / 2.0))
        empirical.append(empirical_coverage(y_true, mean, std, z=z))
    return levels, np.asarray(empirical, dtype=float)


def pit_values(y_true: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Probability integral transform values for Gaussian predictive distributions."""
    from scipy import stats

    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    mean = np.asarray(mean, dtype=float).reshape(-1)
    std = np.asarray(std, dtype=float).reshape(-1)
    std = np.where(std <= 0, 1e-8, std)
    return stats.norm.cdf(y_true, loc=mean, scale=std)


def calibration_report(
    y_true: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> Dict[str, float]:
    """Summary metrics for interval quality."""
    cov_50 = empirical_coverage(y_true, mean, std, z=0.674)
    cov_90 = empirical_coverage(y_true, mean, std, z=1.645)
    cov_95 = empirical_coverage(y_true, mean, std, z=1.96)
    pit = pit_values(y_true, mean, std)
    return {
        "coverage_50": cov_50,
        "coverage_90": cov_90,
        "coverage_95": cov_95,
        "pit_mean": float(np.mean(pit)),
        "pit_std": float(np.std(pit)),
        "mean_interval_width_95": float(np.mean(2 * 1.96 * np.asarray(std))),
    }
