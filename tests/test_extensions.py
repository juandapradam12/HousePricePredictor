"""Extended tests for features, elastic net, calibration, persistence, submission."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from house_price_predictor import (
    BayesianLinearRegression,
    ElasticNetRegressor,
    HAS_LIGHTGBM,
    LeastSquaresRegressor,
    build_feature_frame,
    load_housing_data,
    select_elastic_net_cv,
    select_lasso_cv,
)
from house_price_predictor.calibration import calibration_report, empirical_coverage
from house_price_predictor.diagnostics import cook_distance, diagnostics_summary
from house_price_predictor.features import transform_with_schema
from house_price_predictor.persistence import load_model_bundle, save_model_bundle
from house_price_predictor.submission import make_submission


@pytest.fixture(scope="module")
def housing():
    train, test = load_housing_data()
    return train, test


def test_feature_engineering_shapes(housing):
    train, test = housing
    eng_tr, eng_te = build_feature_frame(train, test, log_target=True)
    assert eng_tr.X.shape[0] == len(eng_tr.ids)
    assert eng_tr.y is not None
    assert eng_te.y is None
    assert eng_tr.X.shape[1] == len(eng_tr.schema.feature_names)
    assert eng_te.X.shape[1] == eng_tr.X.shape[1]
    assert "NeighborhoodEnc" in eng_tr.schema.feature_names
    assert "OverallQual_x_GrLivArea" in eng_tr.schema.feature_names


def test_transform_with_schema_roundtrip(housing):
    train, test = housing
    eng_tr, eng_te = build_feature_frame(train.head(200), test.head(50), log_target=True)
    X2 = transform_with_schema(eng_te.frame, eng_tr.schema)
    np.testing.assert_allclose(X2, eng_te.X, rtol=1e-5, atol=1e-5)


def test_elastic_net_and_lasso_cv(housing):
    train, _ = housing
    eng_tr, _ = build_feature_frame(train.head(300), log_target=True)
    enet, alpha, l1 = select_elastic_net_cv(eng_tr.X, eng_tr.y, cv=3, n_alphas=10)
    pred = enet.predict(eng_tr.X)
    assert pred.shape == eng_tr.y.shape
    assert alpha > 0
    assert 0 < l1 <= 1
    lasso, la = select_lasso_cv(eng_tr.X, eng_tr.y, cv=3, n_alphas=10)
    assert lasso.predict(eng_tr.X).shape == eng_tr.y.shape
    assert la > 0


def test_lightgbm_optional(housing):
    if not HAS_LIGHTGBM:
        pytest.skip("lightgbm not installed")
    from house_price_predictor import LightGBMRegressor

    train, _ = housing
    eng_tr, _ = build_feature_frame(train.head(400), log_target=True)
    X, y = eng_tr.X, eng_tr.y
    model = LightGBMRegressor(n_estimators=50, early_stopping_rounds=10)
    model.fit(X[:300], y[:300], X_val=X[300:], y_val=y[300:])
    pred = model.predict(X[300:])
    assert pred.shape == y[300:].shape


def test_calibration_and_diagnostics(housing):
    train, _ = housing
    eng_tr, _ = build_feature_frame(train.head(400), log_target=True)
    X, y = eng_tr.X, eng_tr.y
    model = BayesianLinearRegression(0.1).fit(X[:320], y[:320])
    mean, std = model.predict(X[320:], return_std=True)
    cov = empirical_coverage(y[320:], mean, std, z=1.96)
    assert 0.0 <= cov <= 1.0
    report = calibration_report(y[320:], mean, std)
    assert "coverage_95" in report
    ols = LeastSquaresRegressor().fit(X[:320], y[:320])
    pred = ols.predict(X[:320])
    summary = diagnostics_summary(y[:320], pred, X[:320])
    assert summary["residual_std"] > 0
    assert cook_distance(y[:320], pred, X[:320]).shape == (320,)


def test_persistence_and_submission(housing, tmp_path):
    train, test = housing
    eng_tr, _ = build_feature_frame(train.head(250), log_target=True)
    model = ElasticNetRegressor(alpha=0.01, l1_ratio=0.5).fit(eng_tr.X, eng_tr.y)
    path = save_model_bundle(tmp_path / "bundle.joblib", model, eng_tr.schema, {"model_name": "enet"})
    loaded = load_model_bundle(path)
    assert loaded["metadata"]["model_name"] == "enet"
    sub = make_submission(loaded["model"], loaded["schema"], test.head(20), tmp_path / "submission.csv")
    assert sub.exists()
    text = sub.read_text().strip().splitlines()
    assert text[0] == "Id,SalePrice"
    assert len(text) == 21
