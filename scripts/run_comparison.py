#!/usr/bin/env python3
"""
Ablation + model bake-off with Optuna LightGBM, stacking, SHAP, and quantiles.

Writes comparison JSON, ablation table, model bundle, Kaggle submission,
SHAP plots, and quantile coverage stats.
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
    HAS_LIGHTGBM,
    HAS_OPTUNA,
    LeastSquaresRegressor,
    LightGBMRegressor,
    QuantileLightGBM,
    RidgeRegressor,
    StackingRegressor,
    build_feature_frame,
    load_housing_data,
    prepare_xy,
    select_elastic_net_cv,
    select_lambda_cv,
    select_lasso_cv,
    tune_lightgbm_optuna,
)
from house_price_predictor.calibration import calibration_report, empirical_coverage  # noqa: E402
from house_price_predictor.metrics import regression_report  # noqa: E402
from house_price_predictor.persistence import save_model_bundle  # noqa: E402
from house_price_predictor.shap_explain import explain_tree_model, save_shap_bar, save_shap_summary  # noqa: E402
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
    import os

    fast = os.environ.get("HPP_FAST", "").lower() in {"1", "true", "yes"}
    optuna_trials = 8 if fast else 25
    optuna_timeout = 45.0 if fast else 90.0
    optuna_estimators = 400 if fast else 1200

    train, test = load_housing_data()
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(train))
    n_val = int(round(0.2 * len(train)))
    val_idx, tr_idx = idx[:n_val], idx[n_val:]
    train_part = train.iloc[tr_idx].reset_index(drop=True)
    val_part = train.iloc[val_idx].reset_index(drop=True)

    ablation = []

    X2, y2, _ = prepare_xy(train_part, features=["GrLivArea", "YearBuilt"], log_target=True)
    X2v, y2v, _ = prepare_xy(val_part, features=["GrLivArea", "YearBuilt"], log_target=True)
    row = _eval("Ablation: OLS (GrLivArea+YearBuilt)", LeastSquaresRegressor().fit(X2, y2), X2v, y2v)
    row["feature_set"] = "2_raw"
    ablation.append(row)

    Xb, yb, fb = prepare_xy(train_part, log_target=True)
    Xbv, ybv, _ = prepare_xy(val_part, features=fb, log_target=True)
    row = _eval("Ablation: OLS (corr>=0.5)", LeastSquaresRegressor().fit(Xb, yb), Xbv, ybv)
    row["feature_set"] = "corr_0.5"
    row["n_features"] = len(fb)
    ablation.append(row)

    eng_tr, eng_va = build_feature_frame(
        train_part, val_part, log_target=True, oof_target_encoding=True, include_onehot=True
    )
    Xe, ye = eng_tr.X, eng_tr.y
    Xev, yev = eng_va.X, eng_va.y
    schema = eng_tr.schema
    print(f"\nEngineered features ({len(schema.feature_names)}): {schema.feature_names}")
    print(f"OOF neighborhood encoding: {schema.oof_target_encoding}")

    row = _eval("Ablation: OLS (engineered+OOF+onehot)", LeastSquaresRegressor().fit(Xe, ye), Xev, yev)
    row["feature_set"] = "engineered_oof_onehot"
    row["n_features"] = len(schema.feature_names)
    ablation.append(row)

    best_lambda, _, _ = select_lambda_cv(Xe, ye)
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

    lgbm_params = None
    optuna_rmse = None
    if HAS_LIGHTGBM and HAS_OPTUNA:
        print("\nTuning LightGBM with Optuna…")
        lgbm, lgbm_params, optuna_rmse = tune_lightgbm_optuna(
            Xe,
            ye,
            n_trials=optuna_trials,
            n_splits=3,
            timeout=optuna_timeout,
            n_estimators=optuna_estimators,
        )
        # Refit with val early stopping for fair hold-out eval
        lgbm = LightGBMRegressor(
            params=lgbm_params,
            n_estimators=max(600, optuna_estimators),
            early_stopping_rounds=80,
        )
        lgbm.fit(Xe, ye, X_val=Xev, y_val=yev)
        models["LightGBM (Optuna)"] = lgbm
        print(f"Optuna best CV RMSE (log)={optuna_rmse:.4f}")
        print(f"Best params: {lgbm_params}")

        stack = StackingRegressor(
            ridge_alpha=best_lambda,
            lgbm_params=lgbm_params,
            n_splits=5,
            n_estimators=1000,
        ).fit(Xe, ye, X_val=Xev, y_val=yev)
        models["Stack (Ridge+LightGBM)"] = stack
        print(f"Stack OOF RMSE (log)={stack.oof_rmse_:.4f}  meta coefs={stack.meta_weights_}")

        qmodel = QuantileLightGBM(params=lgbm_params, n_estimators=800).fit(
            Xe, ye, X_val=Xev, y_val=yev
        )
        models["LightGBM quantile (median)"] = qmodel
    elif HAS_LIGHTGBM:
        lgbm = LightGBMRegressor(n_estimators=1200, early_stopping_rounds=80)
        lgbm.fit(Xe, ye, X_val=Xev, y_val=yev)
        models["LightGBM"] = lgbm

    print("\n--- Engineered-feature bake-off ---")
    for name, model in models.items():
        row = _eval(name, model, Xev, yev)
        row["feature_set"] = "engineered_oof_onehot"
        row["n_features"] = len(schema.feature_names)
        results.append(row)

    bayes = models["Bayesian MAP"]
    mean, std = bayes.predict(Xev, return_std=True)
    cal = calibration_report(yev, mean, std)
    print("\nBayesian predictive calibration (log target):")
    for k, v in cal.items():
        print(f"  {k}: {v:.4f}")

    quantile_stats = None
    if "LightGBM quantile (median)" in models:
        qmodel = models["LightGBM quantile (median)"]
        lo, mid, hi = qmodel.predict_intervals(Xev)
        # empirical coverage of [lo, hi] on log target
        cov = float(np.mean((yev >= lo) & (yev <= hi)))
        width = float(np.mean(hi - lo))
        quantile_stats = {"nominal": 0.90, "empirical_coverage": cov, "mean_width_log": width}
        print(f"\nQuantile 5–95% coverage={cov:.4f}  mean width(log)={width:.4f}")

    out_dir = ROOT / "artifacts"
    out_dir.mkdir(exist_ok=True)

    # SHAP for Optuna LightGBM when available
    shap_paths = {}
    tree_key = "LightGBM (Optuna)" if "LightGBM (Optuna)" in models else ("LightGBM" if "LightGBM" in models else None)
    if tree_key:
        try:
            _, shap_values, X_s, _ = explain_tree_model(
                models[tree_key], Xe, feature_names=schema.feature_names
            )
            shap_paths["summary"] = str(
                save_shap_summary(shap_values, X_s, schema.feature_names, out_dir / "shap_summary.png")
            )
            shap_paths["bar"] = str(
                save_shap_bar(shap_values, schema.feature_names, out_dir / "shap_bar.png")
            )
            print(f"Wrote SHAP plots → {shap_paths}")
        except Exception as exc:  # pragma: no cover
            print(f"SHAP skipped: {exc}")

    payload = {
        "feature_names": schema.feature_names,
        "oof_target_encoding": schema.oof_target_encoding,
        "onehot_cols": schema.onehot_cols,
        "ridge_lambda": best_lambda,
        "elastic_net": {"alpha": enet_alpha, "l1_ratio": enet_l1},
        "lasso_alpha": lasso_alpha,
        "lightgbm_optuna": {"best_params": lgbm_params, "cv_rmse_log": optuna_rmse},
        "n_train": int(Xe.shape[0]),
        "n_val": int(Xev.shape[0]),
        "calibration": cal,
        "quantile_intervals": quantile_stats,
        "shap_artifacts": shap_paths,
        "results": results,
    }
    (out_dir / "model_comparison.json").write_text(json.dumps(payload, indent=2, default=str))
    (out_dir / "ablation_table.json").write_text(json.dumps({"ablation": ablation}, indent=2))

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
