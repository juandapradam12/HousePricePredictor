#!/usr/bin/env python3
"""
Ablation + model bake-off.

Compares feature sets and estimators on a shared hold-out split, writes
``artifacts/model_comparison.json`` and ``artifacts/ablation_table.json``,
trains a production bundle, and emits a Kaggle submission CSV.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LinearRegression, Ridge

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from house_price_predictor import (  # noqa: E402
    BayesianLinearRegression,
    ElasticNetRegressor,
    HAS_LIGHTGBM,
    LeastSquaresRegressor,
    LightGBMRegressor,
    RidgeRegressor,
    build_feature_frame,
    load_housing_data,
    prepare_xy,
    select_elastic_net_cv,
    select_lambda_cv,
    select_lasso_cv,
    train_val_split,
)
from house_price_predictor.calibration import calibration_report  # noqa: E402
from house_price_predictor.metrics import regression_report  # noqa: E402
from house_price_predictor.persistence import save_model_bundle  # noqa: E402
from house_price_predictor.submission import make_submission  # noqa: E402


def _usd(y_log: np.ndarray, pred_log: np.ndarray):
    return np.expm1(y_log), np.expm1(pred_log)


def _eval(name: str, model, X_va, y_va) -> dict:
    pred_log = model.predict(X_va)
    y_true, y_pred = _usd(y_va, pred_log)
    report = regression_report(y_true, y_pred)
    report_log = regression_report(y_va, pred_log)
    row = {
        "model": name,
        "rmse_usd": report["rmse"],
        "mae_usd": report["mae"],
        "r2_usd": report["r2"],
        "rmse_log": report_log["rmse"],
        "r2_log": report_log["r2"],
    }
    print(
        f"{name:42s}  RMSE=${report['rmse']:,.0f}  "
        f"MAE=${report['mae']:,.0f}  R²={report['r2']:.4f}"
    )
    return row


def main() -> None:
    train, test = load_housing_data()
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(train))
    n_val = int(round(0.2 * len(train)))
    val_idx, tr_idx = idx[:n_val], idx[n_val:]
    train_part = train.iloc[tr_idx].reset_index(drop=True)
    val_part = train.iloc[val_idx].reset_index(drop=True)

    ablation = []

    # --- Ablation A: raw 2 features ---
    X2, y2, f2 = prepare_xy(train_part, features=["GrLivArea", "YearBuilt"], log_target=True)
    X2v, y2v, _ = prepare_xy(val_part, features=["GrLivArea", "YearBuilt"], log_target=True)
    m = LeastSquaresRegressor().fit(X2, y2)
    row = _eval("Ablation: OLS (GrLivArea+YearBuilt)", m, X2v, y2v)
    row["feature_set"] = "2_raw"
    ablation.append(row)

    # --- Ablation B: corr >= 0.5 defaults ---
    Xb, yb, fb = prepare_xy(train_part, log_target=True)
    Xbv, ybv, _ = prepare_xy(val_part, features=fb, log_target=True)
    m = LeastSquaresRegressor().fit(Xb, yb)
    row = _eval("Ablation: OLS (corr>=0.5)", m, Xbv, ybv)
    row["feature_set"] = "corr_0.5"
    row["n_features"] = len(fb)
    ablation.append(row)

    # --- Ablation C: engineered features ---
    eng_tr, eng_va = build_feature_frame(train_part, val_part, log_target=True)
    Xe, ye = eng_tr.X, eng_tr.y
    Xev, yev = eng_va.X, eng_va.y
    schema = eng_tr.schema
    print(f"\nEngineered features ({len(schema.feature_names)}): {schema.feature_names}")

    m = LeastSquaresRegressor().fit(Xe, ye)
    row = _eval("Ablation: OLS (engineered)", m, Xev, yev)
    row["feature_set"] = "engineered"
    row["n_features"] = len(schema.feature_names)
    ablation.append(row)

    # --- Full model bake-off on engineered features ---
    best_lambda, _, cv_scores = select_lambda_cv(Xe, ye)
    enet, enet_alpha, enet_l1 = select_elastic_net_cv(Xe, ye)
    lasso, lasso_alpha = select_lasso_cv(Xe, ye)

    results = []
    models = {
        "OLS (from scratch)": LeastSquaresRegressor().fit(Xe, ye),
        f"Ridge CV λ={best_lambda:.3g}": RidgeRegressor(alpha=best_lambda).fit(Xe, ye),
        "Bayesian MAP": BayesianLinearRegression(lambda_param=0.1).fit(Xe, ye),
        f"Elastic Net α={enet_alpha:.3g} l1={enet_l1:.2f}": enet,
        f"Lasso α={lasso_alpha:.3g}": lasso,
        "sklearn LinearRegression": LinearRegression().fit(Xe, ye),
        f"sklearn Ridge λ={best_lambda:.3g}": Ridge(alpha=best_lambda).fit(Xe, ye),
    }

    if HAS_LIGHTGBM:
        lgbm = LightGBMRegressor(n_estimators=1200, early_stopping_rounds=80)
        lgbm.fit(Xe, ye, X_val=Xev, y_val=yev)
        models["LightGBM"] = lgbm

    print("\n--- Engineered-feature bake-off ---")
    for name, model in models.items():
        row = _eval(name, model, Xev, yev)
        row["feature_set"] = "engineered"
        row["n_features"] = len(schema.feature_names)
        results.append(row)

    # Bayesian calibration on hold-out
    bayes = models["Bayesian MAP"]
    mean, std = bayes.predict(Xev, return_std=True)
    cal = calibration_report(yev, mean, std)
    print("\nBayesian predictive calibration (log target):")
    for k, v in cal.items():
        print(f"  {k}: {v:.4f}")

    out_dir = ROOT / "artifacts"
    out_dir.mkdir(exist_ok=True)
    (out_dir / "model_comparison.json").write_text(
        json.dumps(
            {
                "feature_names": schema.feature_names,
                "ridge_lambda": best_lambda,
                "elastic_net": {"alpha": enet_alpha, "l1_ratio": enet_l1},
                "lasso_alpha": lasso_alpha,
                "n_train": int(Xe.shape[0]),
                "n_val": int(Xev.shape[0]),
                "calibration": cal,
                "results": results,
            },
            indent=2,
        )
    )
    (out_dir / "ablation_table.json").write_text(json.dumps({"ablation": ablation}, indent=2))

    # Persist the best hold-out model by USD R²
    best_row = max(results, key=lambda r: r["r2_usd"])
    prod_name = best_row["model"]
    prod_model = models[prod_name]
    bundle_path = save_model_bundle(
        out_dir / "models" / "best_model.joblib",
        prod_model,
        schema,
        metadata={"model_name": prod_name, "metrics": best_row, "all_results": results},
    )
    sub_path = make_submission(prod_model, schema, test, out_dir / "submission.csv")
    print(f"\nSaved bundle → {bundle_path} ({prod_name})")
    print(f"Wrote Kaggle submission → {sub_path}")
    print(f"Wrote {out_dir / 'model_comparison.json'}")
    print(f"Wrote {out_dir / 'ablation_table.json'}")


if __name__ == "__main__":
    main()
