# Model Card — House Price Predictor

**Version:** 1.2.0  
**Task:** Regression — predict residential `SalePrice` (USD) in Ames, Iowa  
**Primary metrics:** Hold-out RMSE / MAE / R² on the USD scale (models train on `log1p(SalePrice)`)  
**Reference docs:** [DOCUMENTATION.md](DOCUMENTATION.md) · [README](../README.md)

---

## Intended use

**In scope**

- Educational / portfolio demonstration of classical linear algebra, Bayesian inference, leakage-safe encoding, boosting, stacking, interpretability, and a minimal shipping path (bundle + demo + submission CSV).
- Offline experimentation on the public Ames Housing competition data.

**Out of scope**

- Automated appraisals, lending, insurance underwriting, or any high-stakes housing decision without licensed human review and local market validation.
- Deployment as a production pricing API for live transactions.

## Training data

| Item | Detail |
|---|---|
| Source | [Ames Housing](https://www.kaggle.com/c/house-prices-advanced-regression-techniques) (Dean De Cock) |
| Train | 1,460 labeled homes (`data/train.csv`) |
| Test | 1,459 unlabeled rows (`data/test.csv`) |
| Target | `SalePrice` (USD), modeled as `log1p` |
| Filters | Drop train rows with `GrLivArea > 4000` |

## Features

- Numeric quality / size / garage / basement / year fields
- Ordinal quality maps (`Ex…Po`)
- **OOF target-encoded** `Neighborhood` (fold-safe on train; full means at serve-time)
- One-hot: `MSZoning`, `SaleCondition`, `GarageType`
- Interaction: `OverallQual × GrLivArea`

## Models considered

OLS, Ridge (CV), Bayesian MAP, Lasso / Elastic Net, Optuna-tuned LightGBM, **Ridge + LightGBM stack**, quantile LightGBM, hierarchical Bayes (notebook), MCMC (notebook).

`artifacts/models/best_model.joblib` stores the **best hold-out R²** model from `scripts/run_comparison.py` (typically the stack on the current split).

## Evaluation protocol

1. 80/20 hold-out on train (seed 42)  
2. Feature schema fit on the train split only (OOF neighborhood encoding)  
3. USD metrics via `expm1`  
4. Bayesian calibration + LightGBM quantile coverage recorded in `artifacts/model_comparison.json`  

**Indicative hold-out (rebuild to refresh):** stack R² ≈ 0.91, RMSE ≈ $21.4k; engineered OLS R² ≈ 0.91; Bayesian 95% coverage ≈ 0.94.

## Uncertainty

| Method | Notes |
|---|---|
| Bayesian MAP | Analytic predictive σ; coverage/PIT checked in notebook `08` |
| Quantile LightGBM | 5% / 50% / 95% bands; empirical coverage may be **below** nominal — inspect JSON |

Intervals are approximate and not a substitute for appraisal confidence statements.

## Ethical considerations

- Sale prices encode location, amenity, and historical housing patterns. Neighborhood / zoning encodings can amplify disparities if misused for offers or credit.
- No explicit protected-class features are used; proxies (neighborhood, zoning) can still correlate with them.
- Do not use outputs to discriminate in housing, credit, or insurance.

## Limitations

- Ames-only; not transferable without retraining.
- ~1.5k rows — strong linear models remain competitive with trees.
- Missing inputs imputed from training medians.
- Streamlit demo fills unspecified fields with medians (illustrative UI).

## Maintenance

| Artifact | Path |
|---|---|
| Bundle | `artifacts/models/best_model.joblib` |
| Submission | `artifacts/submission.csv` |
| Metrics | `artifacts/model_comparison.json` |
| SHAP | `artifacts/shap_bar.png`, `artifacts/shap_summary.png` |
| Retrain | `python scripts/run_comparison.py` |
| Demo | `streamlit run app/streamlit_app.py` or `docker compose up --build` |

## Contact

Repository: [juandapradam12/HousePricePredictor](https://github.com/juandapradam12/HousePricePredictor)
