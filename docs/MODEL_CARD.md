# Model Card — House Price Predictor

**Version:** 1.2  
**Task:** Regression — predict residential `SalePrice` (USD) in Ames, Iowa  
**Primary metrics:** Hold-out RMSE / MAE / R² on USD scale (models trained on `log1p(SalePrice)`)

---

## Intended use

- **Educational / portfolio:** demonstrate classical linear algebra, Bayesian inference, regularization, leakage-safe encoding, boosting, stacking, and interpretability on a public housing dataset.
- **Not for:** automated appraisals, lending decisions, insurance underwriting, or any high-stakes housing valuation without human expert review and local market validation.

## Training data

| Item | Detail |
|---|---|
| Source | [Ames Housing](https://www.kaggle.com/c/house-prices-advanced-regression-techniques) (Dean De Cock) |
| Train size | 1,460 labeled homes (`data/train.csv`) |
| Test size | 1,459 unlabeled rows for Kaggle-style submission |
| Target | `SalePrice` (USD); modeled as `log1p` |
| Filters | Drop train rows with `GrLivArea > 4000` |

## Features

- Numeric correlates (quality, area, garage, basement, year, …)
- Ordinal quality maps (`Ex…Po`)
- **OOF target-encoded** `Neighborhood` (fold-safe on train; full means for serve-time)
- One-hot: `MSZoning`, `SaleCondition`, `GarageType`
- Interaction: `OverallQual × GrLivArea`
- Optuna-tuned LightGBM + Ridge stack; SHAP explanations; quantile intervals

## Models considered

OLS, Ridge (CV), Bayesian MAP, Lasso / Elastic Net, Optuna-tuned LightGBM, Ridge+LightGBM stack, quantile LightGBM, hierarchical Bayes (notebook).

The production `artifacts/models/best_model.joblib` stores the **best hold-out R²** model from `scripts/run_comparison.py`.

## Evaluation protocol

1. 80/20 hold-out on train (seed 42)  
2. Feature schema fit on the train split only (OOF neighborhood encoding)  
3. Metrics reported in USD via `expm1`  
4. Bayesian interval calibration + LightGBM quantile coverage reported in artifacts JSON  

See `artifacts/model_comparison.json` and `artifacts/ablation_table.json` for the latest numbers.

## Uncertainty

- **Bayesian MAP:** analytic predictive σ; calibration checked via coverage / PIT  
- **Quantile LightGBM:** 5% / 50% / 95% quantiles for non-parametric intervals  

Intervals are approximate and not a substitute for appraisal confidence bands.

## Ethical & fairness considerations

- Housing prices encode historical **location, segregation, and amenity** effects. Neighborhood encodings can amplify disparities if used to set offers or loans.
- The model has **no protected-class features** by design, but proxies (neighborhood, zoning) can correlate with them.
- Do not use outputs to discriminate in housing, credit, or insurance contexts.

## Limitations

- Trained only on Ames, Iowa listings from the Kaggle competition era — not transferable to other cities without retraining.
- ~1.5k rows: boosting gains are limited; strong linear models remain competitive.
- Missing fields on new homes are median/mode imputed from training statistics.
- Streamlit demo fills unspecified columns with training medians — illustrative only.

## Maintenance

| Artifact | Path |
|---|---|
| Bundle | `artifacts/models/best_model.joblib` |
| Submission | `artifacts/submission.csv` |
| Metrics | `artifacts/model_comparison.json` |
| Retrain | `python scripts/run_comparison.py` |
| Demo | `streamlit run app/streamlit_app.py` or `docker compose up --build` |

## Contact

Repository: [juandapradam12/HousePricePredictor](https://github.com/juandapradam12/HousePricePredictor)
