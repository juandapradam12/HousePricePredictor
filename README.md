# House Price Predictor

### From-scratch linear & Bayesian models → engineered features → Optuna LightGBM → stacking, SHAP & a shippable demo

Predict Ames sale prices while *seeing the math*, then push accuracy with leakage-safe encoding, tuned boosting, and a Ridge+LightGBM stack. Includes calibration, quantile intervals, SHAP, Streamlit/Docker demo, and a Kaggle submission CSV.

<p align="center">
  <img src="figures/saleprice_relationships.png" alt="Sale price vs living area and year built" width="900"/>
</p>

---

## Why this project

Most housing tutorials call `model.fit()` and move on. This repo does the opposite:

- **Implements classical estimators by hand** (stable linear algebra, not toy inverses)
- **Leakage-safe features** — OOF target-encoded `Neighborhood`, ordinals, one-hots, interactions
- **Tunes and stacks** — Optuna LightGBM + Ridge→linear meta-learner
- **Explains predictions** — SHAP beeswarm / bar plots
- **Quantifies uncertainty** — Bayesian σ and LightGBM quantile bands
- **Ships the loop** — tests, CI, `joblib` bundle, `submission.csv`, Streamlit + Docker

## Results at a glance

Hold-out 20% (seed 42). Re-run: `python scripts/run_comparison.py`.

### Feature ablation (OLS)

| Feature set | RMSE | MAE | R² |
|---|---:|---:|---:|
| `GrLivArea` + `YearBuilt` | $44.2k | $28.1k | 0.616 |
| Corr ≥ 0.5 numerics | $28.2k | $19.3k | 0.844 |
| **Engineered + OOF + one-hot (46)** | **$21.9k** | **$15.3k** | **0.905** |

### Model bake-off (engineered features)

| Model | RMSE | MAE | R² |
|---|---:|---:|---:|
| **Stack (Ridge + LightGBM)** | **$21.6k** | **$14.6k** | **0.908** |
| Ridge CV | $21.7k | $15.3k | 0.908 |
| Elastic Net / Lasso | $21.9k | $15.3k | 0.906 |
| OLS / Bayesian MAP | $21.9k | $15.3k | 0.905 |
| LightGBM (Optuna) | $23.9k | $15.0k | 0.888 |

Production bundle = best hold-out R² (currently the **stack**). Bayesian 95% coverage ≈ 0.94; see `08_calibration.ipynb` and `11_quantile_intervals.ipynb`.

<p align="center">
  <img src="artifacts/shap_bar.png" alt="SHAP feature importance" width="720"/>
</p>

## Quick start

```bash
git clone https://github.com/juandapradam12/HousePricePredictor.git
cd HousePricePredictor
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,demo]"

# Linux (once) for PyMC / PyTensor:
# sudo apt-get install -y python3-dev

pytest -q
python scripts/run_comparison.py          # metrics + SHAP + bundle + submission.csv
streamlit run app/streamlit_app.py

# or
docker compose up --build
```

`HPP_FAST=1` shortens Optuna for CI smoke tests.

## What's inside

| Path | Purpose |
|---|---|
| `src/house_price_predictor/` | Features, models, stacking, SHAP, quantiles, calibration, persistence |
| `notebooks/` | `01`–`11` guided tour (incl. SHAP & quantiles) |
| `scripts/run_comparison.py` | Ablation + bake-off + artifacts |
| `app/streamlit_app.py` | Interactive demo |
| `Dockerfile` / `docker-compose.yml` | One-command demo |
| `docs/DOCUMENTATION.md` | Math & API |
| `docs/MODEL_CARD.md` | Intended use, limits, ethics |
| `artifacts/` | Metrics, SHAP plots, `submission.csv`, `best_model.joblib` |

## Feature engineering

```text
ordinals           Ex/Gd/TA/Fa/Po → 5..1
Neighborhood       OOF target encoding on train (full means for serve-time)
one-hot            MSZoning, SaleCondition, GarageType
interaction        OverallQual × GrLivArea
numeric base       quality, area, garage, basement, year, LotArea, …
outliers           GrLivArea > 4000 dropped on train
target             log1p(SalePrice)
```

## Models

| Model | Idea |
|---|---|
| **OLS / Ridge / Bayes / MCMC** | From-scratch educational core |
| **Lasso / Elastic Net** | Sparse shrinkage with CV |
| **LightGBM + Optuna** | Tuned nonlinear baseline |
| **Stack** | OOF Ridge + LightGBM → Ridge meta |
| **Quantile LightGBM** | 5% / 50% / 95% prediction bands |
| **Hierarchical Bayes** | Neighborhood partial pooling |

## Example

```python
from house_price_predictor import (
    load_housing_data, build_feature_frame, tune_lightgbm_optuna, StackingRegressor
)
from house_price_predictor.submission import make_submission

train, test = load_housing_data()
eng_tr, eng_te = build_feature_frame(train, test, oof_target_encoding=True)
lgbm, params, cv = tune_lightgbm_optuna(eng_tr.X, eng_tr.y, n_trials=20)
stack = StackingRegressor(lgbm_params=params).fit(eng_tr.X, eng_tr.y)
make_submission(stack, eng_tr.schema, test, "artifacts/submission.csv")
```

## Notebooks

1–6 core models · 7 diagnostics · 8 calibration · 9 hierarchical Bayes · **10 SHAP** · **11 quantile intervals**

## Documentation

- **[docs/DOCUMENTATION.md](docs/DOCUMENTATION.md)** — math, API, changelog  
- **[docs/MODEL_CARD.md](docs/MODEL_CARD.md)** — intended use & limitations  

## Data

[Ames Housing](https://www.kaggle.com/c/house-prices-advanced-regression-techniques) (Dean De Cock).

## License

Educational / portfolio project. Dataset subject to Kaggle / original authors' terms.
