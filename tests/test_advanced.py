"""Tests for OOF encoding, Optuna/stacking (smoke), quantiles, and one-hots."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from house_price_predictor import (
    HAS_LIGHTGBM,
    HAS_OPTUNA,
    build_feature_frame,
    load_housing_data,
    target_encode_oof,
)
from house_price_predictor.features import apply_target_means, fit_target_means


@pytest.fixture(scope="module")
def housing():
    return load_housing_data()


def test_oof_encoding_differs_from_full_fit(housing):
    train, _ = housing
    y = np.log1p(train["SalePrice"].to_numpy(dtype=float))
    oof, means, gmean = target_encode_oof(train["Neighborhood"], y, n_splits=5, random_state=0)
    full = apply_target_means(train["Neighborhood"], *fit_target_means(train["Neighborhood"], y))
    # OOF should not be identical to in-sample means for all rows
    assert oof.shape == full.shape
    assert not np.allclose(oof, full)
    assert means and gmean > 0


def test_onehot_and_oof_in_feature_frame(housing):
    train, test = housing
    eng_tr, eng_te = build_feature_frame(
        train.head(250), test.head(40), oof_target_encoding=True, include_onehot=True
    )
    assert eng_tr.schema.oof_target_encoding is True
    assert "MSZoning" in eng_tr.schema.onehot_cols
    assert any(n.startswith("MSZoning__") for n in eng_tr.schema.feature_names)
    assert eng_tr.X.shape[1] == eng_te.X.shape[1] == len(eng_tr.schema.feature_names)
    from house_price_predictor.features import transform_with_schema

    X2 = transform_with_schema(eng_te.frame, eng_tr.schema)
    np.testing.assert_allclose(X2, eng_te.X, rtol=1e-5, atol=1e-5)


def test_optuna_and_stack_smoke(housing):
    if not (HAS_LIGHTGBM and HAS_OPTUNA):
        pytest.skip("lightgbm/optuna not installed")
    from house_price_predictor import StackingRegressor, tune_lightgbm_optuna

    train, _ = housing
    eng, _ = build_feature_frame(train.head(350), oof_target_encoding=True)
    X, y = eng.X, eng.y
    model, params, cv_rmse = tune_lightgbm_optuna(
        X, y, n_trials=3, n_splits=2, timeout=30.0, n_estimators=80
    )
    assert cv_rmse > 0
    assert "num_leaves" in params or "learning_rate" in params
    assert model.predict(X[:10]).shape == (10,)

    stack = StackingRegressor(
        ridge_alpha=10.0, lgbm_params=params, n_splits=2, n_estimators=60
    ).fit(X, y)
    pred = stack.predict(X[:15])
    assert pred.shape == (15,)
    assert stack.oof_rmse_ is not None


def test_quantile_intervals(housing):
    if not HAS_LIGHTGBM:
        pytest.skip("lightgbm not installed")
    from house_price_predictor import QuantileLightGBM

    train, _ = housing
    eng, _ = build_feature_frame(train.head(300), oof_target_encoding=True)
    X, y = eng.X, eng.y
    q = QuantileLightGBM(n_estimators=60).fit(X[:240], y[:240], X_val=X[240:], y_val=y[240:])
    lo, mid, hi = q.predict_intervals(X[240:])
    assert lo.shape == mid.shape == hi.shape
    assert np.all(lo <= mid + 1e-6)
    assert np.all(mid <= hi + 1e-6)
