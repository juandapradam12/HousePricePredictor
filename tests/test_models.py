"""Unit tests for core algorithms and metrics."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression, Ridge

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from house_price_predictor import (
    BayesianLinearRegression,
    LeastSquaresRegressor,
    RidgeRegressor,
    least_squares_weights,
    load_housing_data,
    prepare_xy,
    ridge_regression_weights,
    select_correlated_features,
    select_lambda_cv,
)
from house_price_predictor.bayesian import (
    estimate_noise_variance,
    map_coefficients,
    posterior_covariance,
    predictive_moments,
)
from house_price_predictor.least_squares import add_intercept
from house_price_predictor.metrics import mae, r2_score, rmse


@pytest.fixture(scope="module")
def toy_data():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 3))
    true_w = np.array([1.5, -2.0, 0.5, 3.0])  # intercept + 3 coefs
    y = add_intercept(X) @ true_w + rng.normal(scale=0.1, size=200)
    return X, y, true_w


def test_least_squares_matches_sklearn(toy_data):
    X, y, _ = toy_data
    ours = LeastSquaresRegressor().fit(X, y)
    sk = LinearRegression().fit(X, y)
    np.testing.assert_allclose(ours.intercept_, sk.intercept_, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(ours.coef_, sk.coef_, rtol=1e-6, atol=1e-6)


def test_least_squares_weights_shape(toy_data):
    X, y, _ = toy_data
    w = least_squares_weights(X, y)
    assert w.shape == (X.shape[1] + 1,)


def test_ridge_matches_sklearn_on_standardized(toy_data):
    X, y, _ = toy_data
    # sklearn Ridge does not center y; compare closed forms on mean-centered y
    # and standardized X with intercept unpenalized.
    X_mean, X_std = X.mean(0), X.std(0)
    X_s = (X - X_mean) / X_std
    y_c = y - y.mean()
    lam = 2.5
    w = ridge_regression_weights(X_s, y_c, lam, penalize_intercept=False)

    sk = Ridge(alpha=lam, fit_intercept=True)
    sk.fit(X_s, y_c)
    # sk intercept should be ~0 because y is centered; coefs comparable
    np.testing.assert_allclose(w[1:], sk.coef_, rtol=1e-5, atol=1e-5)


def test_ridge_regressor_predict_roundtrip(toy_data):
    X, y, _ = toy_data
    model = RidgeRegressor(alpha=1.0).fit(X, y)
    pred = model.predict(X)
    assert pred.shape == y.shape
    assert r2_score(y, pred) > 0.9


def test_select_lambda_cv_returns_positive(toy_data):
    X, y, _ = toy_data
    best, grid, scores = select_lambda_cv(X, y, lambdas=[0.01, 0.1, 1.0, 10.0], n_splits=3)
    assert best in set(grid)
    assert scores.shape == grid.shape
    assert np.all(scores > 0)


def test_bayesian_map_close_to_ols_when_lambda_small(toy_data):
    X, y, _ = toy_data
    ols = LeastSquaresRegressor().fit(X, y)
    bayes = BayesianLinearRegression(lambda_param=1e-8).fit(X, y)
    # Bayesian standardizes features internally; compare predictions, not raw coefs
    np.testing.assert_allclose(bayes.predict(X), ols.predict(X), rtol=1e-3, atol=1e-3)


def test_bayesian_predictive_moments_positive_variance(toy_data):
    X, y, _ = toy_data
    model = BayesianLinearRegression(lambda_param=0.1).fit(X, y)
    mean, std = model.predict(X[:5], return_std=True)
    assert mean.shape == (5,)
    assert std.shape == (5,)
    assert np.all(std > 0)


def test_posterior_math_helpers(toy_data):
    X, y, _ = toy_data
    X_aug = add_intercept(X)
    w_ols, *_ = np.linalg.lstsq(X_aug, y, rcond=None)
    sigma2 = estimate_noise_variance(X_aug, y, w_ols)
    mu = map_coefficients(X_aug, y, 0.1, sigma2)
    Sigma = posterior_covariance(X_aug, sigma2, 0.1)
    assert sigma2 > 0
    assert mu.shape == (X.shape[1] + 1,)
    assert Sigma.shape == (X.shape[1] + 1, X.shape[1] + 1)
    mu0, s0 = predictive_moments(X_aug[0], mu, Sigma, sigma2)
    assert s0 > sigma2


def test_metrics():
    y = np.array([1.0, 2.0, 3.0])
    pred = np.array([1.0, 2.0, 4.0])
    assert rmse(y, pred) == pytest.approx(np.sqrt(1 / 3))
    assert mae(y, pred) == pytest.approx(1 / 3)
    assert r2_score(y, y) == pytest.approx(1.0)


def test_load_and_prepare_housing():
    train, test = load_housing_data()
    assert "SalePrice" in train.columns
    assert "SalePrice" not in test.columns
    feats = select_correlated_features(train, threshold=0.5)
    assert "GrLivArea" in feats
    X, y, features = prepare_xy(train, features=feats[:5], log_target=True)
    assert X.shape[0] == y.shape[0]
    assert X.shape[1] == 5
    assert y.max() < 15  # log1p scale sanity
