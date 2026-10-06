# House Price Predictor

### Learn the math. Ship the pipeline. Explain the prediction.

A portfolio-grade take on the [Ames Housing](https://www.kaggle.com/c/house-prices-advanced-regression-techniques) price problem: **from-scratch linear & Bayesian estimators**, leakage-safe feature engineering, **Optuna-tuned LightGBM**, a **Ridge + LightGBM stack**, SHAP explanations, calibrated uncertainty, and a Dockerized Streamlit demo — all tested and CI-wired.

<p align="center">
  <img src="figures/saleprice_relationships.png" alt="Sale price vs living area and year built" width="900"/>
</p>

---

## The use case

**Question:** given a house’s attributes in Ames, Iowa, what should it sell for — and how sure are we?

This repo answers that end-to-end:

| Layer | What you get |
|---|---|
| **Understanding** | OLS, Ridge, conjugate Bayes, and MCMC implemented so you can see the algebra |
| **Accuracy** | Engineered features + stack → hold-out **R² ≈ 0.91**, RMSE ≈ **$21.4k** |
| **Trust** | Bayesian predictive coverage ≈ **94%** at nominal 95%; quantile bands for trees |
| **Explainability** | SHAP beeswarm & bar plots for the boosted model |
| **Delivery** | `joblib` bundle, Kaggle `submission.csv`, Streamlit UI, Docker one-liner |

If you need a black-box Kaggle blitz only, use LightGBM alone. If you need a **story you can defend in an interview** — math → leakage → tuning → stack → explain → ship — this is the repo.

## Visual walkthrough

### Data & relationships

<p align="center">
  <img src="figures/eda_corr_heatmap.png" alt="Correlation heatmap of SalePrice and top features" width="720"/>
</p>

<p align="center"><em>Strong linear signal from quality, living area, garage, and year fields — the base of the feature ladder.</em></p>

### Classical fits

<p align="center">
  <img src="figures/ols_fit.png" alt="OLS fit of SalePrice vs GrLivArea" width="720"/>
</p>

<p align="center">
  <img src="figures/ridge_cv_curve.png" alt="Ridge lambda selection via cross-validation" width="640"/>
</p>

<p align="center"><em>From-scratch OLS on living area, and CV-tuned Ridge λ on the log target.</em></p>

### Uncertainty you can check

<p align="center">
  <img src="figures/bayesian_uncertainty.png" alt="Bayesian predictive mean with uncertainty band" width="720"/>
</p>

<p align="center">
  <img src="figures/calibration_curve.png" alt="Bayesian predictive calibration curve" width="480"/>
</p>

<p align="center"><em>Conjugate Bayesian predictive bands, with empirical coverage close to the diagonal (≈94% at nominal 95%).</em></p>

### Explain & intervalize the tree model

<p align="center">
  <img src="figures/shap_bar.png" alt="SHAP mean absolute feature importance" width="700"/>
</p>

<p align="center">
  <img src="figures/shap_summary.png" alt="SHAP beeswarm summary plot" width="700"/>
</p>

<p align="center">
  <img src="figures/quantile_intervals.png" alt="LightGBM quantile prediction intervals vs living area" width="720"/>
</p>

<p align="center"><em>SHAP shows what drives LightGBM; quantile bands give non-parametric 5–95% intervals.</em></p>

### Residual health

<p align="center">
  <img src="figures/diagnostics_residuals.png" alt="Residual diagnostics for engineered OLS" width="900"/>
</p>

<p align="center"><em>Residuals, histogram, and influence for the engineered linear model.</em></p>

## Results that sell the story

Hold-out **20%** of train (seed 42). Rebuild anytime with `python scripts/run_comparison.py`.

### 1. Features beat algorithms (ablation on OLS)

| Feature set | RMSE | MAE | R² |
|---|---:|---:|---:|
| Living area + year built only | $44.2k | $28.1k | 0.616 |
| Top numeric correlates (\|r\| ≥ 0.5) | $28.2k | $19.3k | 0.844 |
| **Engineered + OOF encoding + one-hots (46 feats)** | **$21.9k** | **$15.3k** | **0.905** |

Most of the jump is **feature design**, not model choice.

### 2. Stack edges the field

| Model | RMSE | MAE | R² |
|---|---:|---:|---:|
| **Stack (Ridge + LightGBM)** | **$21.4k** | **$14.7k** | **0.910** |
| Ridge (CV-tuned λ) | $21.7k | $15.3k | 0.908 |
| Elastic Net / Lasso | $21.9k | $15.3k | 0.906 |
| OLS / Bayesian MAP | $21.9k | $15.3k | 0.905 |
| LightGBM (Optuna) | $23.6k | $15.1k | 0.891 |

Production artifact `artifacts/models/best_model.joblib` stores the **best hold-out R²** model (currently the stack).

## Quick start

```bash
git clone https://github.com/juandapradam12/HousePricePredictor.git
cd HousePricePredictor
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,demo]"

# Once on Linux (PyMC / PyTensor C extensions):
# sudo apt-get install -y python3-dev

pytest -q
python scripts/run_comparison.py    # metrics, SHAP, bundle, submission.csv
streamlit run app/streamlit_app.py  # interactive demo

# Or one command:
docker compose up --build
```

Tip: `HPP_FAST=1 python scripts/run_comparison.py` shortens Optuna (used in CI).

## Repository map

```text
HousePricePredictor/
├── data/                      # Ames train.csv / test.csv
├── src/house_price_predictor/ # installable modeling package
├── notebooks/                 # 01–11 guided tour (+ archive/)
├── scripts/run_comparison.py  # ablation + bake-off + artifacts
├── app/streamlit_app.py       # demo UI
├── figures/                   # README gallery plots
├── artifacts/                 # metrics, SHAP, submission, joblib
├── docs/
│   ├── DOCUMENTATION.md       # math, API, design, changelog
│   └── MODEL_CARD.md          # intended use, ethics, limits
├── Dockerfile / docker-compose.yml
└── .github/workflows/ci.yml
```

| Path | Role |
|---|---|
| `src/house_price_predictor/` | Features, OLS, Ridge, Bayes, Elastic Net, LightGBM/Optuna, stack, SHAP, quantiles, MCMC, hierarchical Bayes, calibration, diagnostics, persistence, submission |
| `notebooks/01`–`11` | EDA → models → diagnostics → calibration → hierarchical Bayes → SHAP → quantiles |
| `notebooks/archive/` | Original educational notebooks (preserved) |
| `figures/` | Committed plots used in this README |
| `docs/DOCUMENTATION.md` | Full technical write-up |
| `docs/MODEL_CARD.md` | Use / misuse / fairness notes |

## Feature engineering (leakage-aware)

```text
target             log1p(SalePrice)
outliers           drop train rows with GrLivArea > 4000
ordinals           Ex/Gd/TA/Fa/Po → 5..1 (missing → 0)
Neighborhood       out-of-fold target encoding on train;
                   full-data means stored for serve-time / test
one-hot            MSZoning, SaleCondition, GarageType
interaction        OverallQual × GrLivArea
numerics           quality, area, garage, basement, baths, years, LotArea, …
imputation         train medians only
```

## Models

| Model | Why it’s here |
|---|---|
| **OLS / Ridge / Bayesian MAP** | Closed-form estimators you can derive and debug |
| **MCMC (PyMC 5)** | Posterior sampling when conjugacy is dropped |
| **Hierarchical Bayes** | Neighborhood partial pooling |
| **Lasso / Elastic Net** | Sparse shrinkage with CV |
| **LightGBM + Optuna** | Tuned nonlinear baseline |
| **Stack** | OOF Ridge + LightGBM → linear meta-learner |
| **Quantile LightGBM** | 5% / 50% / 95% prediction intervals |

## Minimal API example

```python
from house_price_predictor import (
    load_housing_data,
    build_feature_frame,
    tune_lightgbm_optuna,
    StackingRegressor,
)
from house_price_predictor.submission import make_submission

train, test = load_housing_data()
eng_tr, _ = build_feature_frame(train, test, oof_target_encoding=True)

_, params, cv_rmse = tune_lightgbm_optuna(eng_tr.X, eng_tr.y, n_trials=20)
stack = StackingRegressor(lgbm_params=params).fit(eng_tr.X, eng_tr.y)

make_submission(stack, eng_tr.schema, test, "artifacts/submission.csv")
print("CV RMSE (log):", cv_rmse)
```

## Notebook tour

| # | Notebook | Focus |
|---|---|---|
| 01 | Exploratory analysis | Correlations, target skew, visuals |
| 02 | Least squares | From-scratch OLS vs sklearn |
| 03 | Ridge regression | Standardization + CV for λ |
| 04 | Bayesian regression | MAP, Σ, predictive uncertainty |
| 05 | MCMC (PyMC) | NUTS on synthetic + housing |
| 06 | Model comparison | Package bake-off entrypoint |
| 07 | Diagnostics | Residuals, Cook’s D, learning curves |
| 08 | Calibration | Coverage curve + PIT |
| 09 | Hierarchical Bayes | Varying intercepts by neighborhood |
| 10 | SHAP | Tree explanations |
| 11 | Quantile intervals | 5–95% LightGBM bands |

## Documentation

- **[docs/DOCUMENTATION.md](docs/DOCUMENTATION.md)** — problem setup, math, API reference, design decisions, changelog  
- **[docs/MODEL_CARD.md](docs/MODEL_CARD.md)** — intended use, evaluation protocol, ethics, limitations  

## Data

[Ames Housing dataset](https://www.kaggle.com/c/house-prices-advanced-regression-techniques) (Dean De Cock). Labels are USD sale prices; `data/test.csv` has no `SalePrice` (Kaggle submission format).

## License

Educational / portfolio project. Dataset subject to Kaggle and original authors’ terms.
