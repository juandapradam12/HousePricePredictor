"""
House Price Predictor
=====================

From-scratch implementations of classical and Bayesian linear models for
Ames Housing sale-price prediction, with Elastic Net, LightGBM, and PyMC baselines.
"""

from .data import (
    DEFAULT_FEATURES,
    apply_cutoffs,
    correlation_with_target,
    load_housing_data,
    prepare_xy,
    select_correlated_features,
    train_val_split,
)
from .metrics import mae, r2_score, rmse
from .least_squares import LeastSquaresRegressor, least_squares_weights
from .ridge import RidgeRegressor, ridge_regression_weights, select_lambda_cv
from .bayesian import BayesianLinearRegression
from .elastic_net import ElasticNetRegressor, LassoRegressor, select_elastic_net_cv, select_lasso_cv
from .boosting import LightGBMRegressor, HAS_LIGHTGBM
from .features import FeatureSchema, build_feature_frame, transform_with_schema

__all__ = [
    "DEFAULT_FEATURES",
    "HAS_LIGHTGBM",
    "BayesianLinearRegression",
    "ElasticNetRegressor",
    "FeatureSchema",
    "LassoRegressor",
    "LeastSquaresRegressor",
    "LightGBMRegressor",
    "RidgeRegressor",
    "apply_cutoffs",
    "build_feature_frame",
    "correlation_with_target",
    "least_squares_weights",
    "load_housing_data",
    "mae",
    "prepare_xy",
    "r2_score",
    "ridge_regression_weights",
    "rmse",
    "select_correlated_features",
    "select_elastic_net_cv",
    "select_lambda_cv",
    "select_lasso_cv",
    "train_val_split",
    "transform_with_schema",
]

__version__ = "1.1.0"
