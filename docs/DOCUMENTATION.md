# Documentation — House Price Predictor

Technical companion to the README. Covers the problem setup, mathematics, feature pipeline, full API surface, design decisions, how to reproduce artifacts, and the project changelog.

**Package version:** 1.2.0

---

## 1. Problem statement

Given tabular attributes of residential properties in **Ames, Iowa**, estimate sale price in USD.

| Item | Value |
|---|---|
| Train | 1,460 labeled homes (`data/train.csv`) |
| Test | 1,459 unlabeled rows (`data/test.csv`) for Kaggle-style submission |
| Raw attributes | ~80 columns (numeric + categorical) |
| Target transform | $\log(1 + \text{SalePrice})$ during training; metrics also reported via `expm1` |

**Portfolio framing:** derive classical estimators *and* run a modern pipeline (leakage-safe features → tuned boosting → stack → SHAP → ship).

Default numeric correlates ($|r| \ge 0.5$ with `SalePrice`):

`OverallQual`, `GrLivArea`, `GarageCars`, `GarageArea`, `TotalBsmtSF`, `1stFlrSF`, `FullBath`, `TotRmsAbvGrd`, `YearBuilt`, `YearRemodAdd`.

The production feature set expands this with ordinals, OOF neighborhood encoding, one-hots, and an interaction (≈ **46** columns after engineering).

---

## 2. Classical & Bayesian algorithms

### 2.1 Ordinary least squares

$$
w_{\mathrm{LS}} = (X^\top X)^{-1} X^\top y
$$

**Implementation:** `np.linalg.lstsq` on an intercept-augmented design (equivalent when $X$ has full column rank; better conditioned than an explicit inverse).

**API:** `LeastSquaresRegressor`, `least_squares_weights`.

### 2.2 Ridge regression

$$
w_{\mathrm{RR}} = (X^\top X + \lambda I)^{-1} X^\top y
$$

- Features z-scored; target mean-centered inside `RidgeRegressor`
- Intercept **unpenalized** (matches scikit-learn)
- $\lambda$ via **5-fold CV** minimizing log-target RMSE (`select_lambda_cv`)

**API:** `RidgeRegressor`, `ridge_regression_weights`, `select_lambda_cv`.

### 2.3 Bayesian linear regression (conjugate Gaussian)

$$
p(w) = \mathcal{N}(0, \lambda^{-1} I), \qquad
p(y \mid w, X) = \mathcal{N}(Xw, \sigma^2 I)
$$

$$
\Sigma = \big(\lambda I + \sigma^{-2} X^\top X\big)^{-1}, \qquad
\mu = \big(\lambda \sigma^2 I + X^\top X\big)^{-1} X^\top y
$$

$$
\hat\sigma^2 = \frac{1}{n-d}\sum_{i=1}^{n}(y_i - x_i^\top w_{\mathrm{LS}})^2
$$

Predictive:

$$
\mu_0 = x_0^\top \mu, \qquad
\sigma_0^2 = \sigma^2 + x_0^\top \Sigma x_0
$$

**Bug fixes vs. original notebooks**

| Issue | Fix |
|---|---|
| MAP used $\lambda\sigma$ instead of $\lambda\sigma^2$ | `map_coefficients` |
| $\Sigma$ used \$1/\sigma$ vs \$1/\sigma^2$ | `posterior_covariance` |
| Predict helper missing $\sigma$ | Class API stores $\sigma^2$ on the posterior |
| Raw-scale isotropic prior | Features standardized in `fit` |

**API:** `BayesianLinearRegression.predict(..., return_std=True)`.

### 2.4 MCMC (PyMC ≥ 5)

NUTS sampling for illustration / non-conjugate workflows:

- `intercept ~ Normal(mean(y), ·)`
- `beta ~ Normal(0, 1)` on standardized features
- `sigma ~ HalfNormal` (replaces an invalid Normal-on-scale prior from the PyMC3 era)

**API:** `house_price_predictor.mcmc.fit_bayesian_mcmc`, `generate_synthetic_regression`.

### 2.5 Hierarchical Bayes by neighborhood

Varying intercept with partial pooling:

$$
y_i \sim \mathcal{N}(\alpha_{\mathrm{neigh}[i]} + x_i^\top \beta,\ \sigma), \qquad
\alpha_j \sim \mathcal{N}(\mu_\alpha,\ \tau_\alpha)
$$

**API:** `fit_hierarchical_neighborhood`, `hierarchical_predict_mean` (notebook `09`).

---

## 3. Feature engineering (`features.py`)

`build_feature_frame(train, other, **opts)` → `(EngineeredData_train, EngineeredData_other)`.

### Pipeline steps

1. Drop train rows with `GrLivArea > 4000`
2. Map ordinal quality strings (`Ex…Po`) → integers (`NA` → 0)
3. Median-impute numerics using **train** statistics only
4. **Out-of-fold target-encode** `Neighborhood` on the training matrix (`oof_target_encoding=True` by default); store full-data means on `FeatureSchema` for `other` / production
5. Add interaction `OverallQual × GrLivArea`
6. One-hot `MSZoning`, `SaleCondition`, `GarageType` (levels fit on train)
7. Build `y = log1p(SalePrice)` when the target is present

`transform_with_schema(frame, schema)` applies a fitted schema to Kaggle test rows or Streamlit inputs.

### Why OOF encoding matters

Fitting neighborhood means on all of `train_part` and then running CV *inside* that matrix leaks target info into validation folds. OOF encoding assigns each training row a mean computed **without** that row’s fold, while serve-time still uses stable full-data means.

### Key helpers

| Symbol | Role |
|---|---|
| `FeatureSchema` | Serializable feature contract (names, medians, encodings, one-hot levels) |
| `EngineeredData` | `X`, `y`, `ids`, `schema`, `frame` |
| `target_encode_oof` | Standalone OOF encoder |
| `fit_target_means` / `apply_target_means` | Category → mean maps |

---

## 4. Modern estimators & tooling

| Module | API | Role |
|---|---|---|
| `elastic_net.py` | `ElasticNetRegressor`, `LassoRegressor`, `select_elastic_net_cv`, `select_lasso_cv` | Sparse / mixed-norm linear models |
| `boosting.py` | `LightGBMRegressor`, `tune_lightgbm_optuna`, `QuantileLightGBM` | Tuned trees + quantile bands |
| `stacking.py` | `StackingRegressor` | OOF Ridge + LightGBM → Ridge meta |
| `shap_explain.py` | `explain_tree_model`, `save_shap_summary`, `save_shap_bar` | TreeSHAP plots |
| `calibration.py` | `empirical_coverage`, `calibration_curve`, `pit_values`, `calibration_report` | Interval quality |
| `diagnostics.py` | residuals, leverage, Cook’s D, learning curves | Linear-model health |
| `persistence.py` | `save_model_bundle` / `load_model_bundle` | joblib `{model, schema, metadata}` |
| `submission.py` | `make_submission`, `predict_saleprice` | Kaggle `Id,SalePrice` CSV |

### Optuna LightGBM

`tune_lightgbm_optuna(X, y, n_trials=…, n_splits=3, …)` minimizes k-fold RMSE over learning rate, leaves, regularization, bagging, etc., then refits with early stopping.

### Stacking

1. Build OOF predictions from `RidgeRegressor` and `LightGBMRegressor`
2. Fit a Ridge meta-model on the OOF matrix
3. Refit both bases on all training rows for inference

### Quantile intervals

`QuantileLightGBM` fits objectives at 5%, 50%, 95%. Empirical coverage on Ames hold-out is typically **below** the nominal 90% (bands are a bit narrow) — report honest coverage from `artifacts/model_comparison.json` rather than assuming calibration.

---

## 5. Data utilities & metrics

| Function | Role |
|---|---|
| `load_housing_data()` | Load `data/train.csv` + `data/test.csv` |
| `correlation_with_target` | Pearson corrs (numeric only) |
| `select_correlated_features` | Thresholded feature list |
| `prepare_xy` | Simple numeric design matrix (+ optional `log1p`) |
| `train_val_split` | Reproducible hold-out |
| `apply_cutoffs` | Row filters |

**Metrics** (`metrics.py`): `rmse`, `mae`, `r2_score`, `regression_report`.  
The comparison script always reports **USD** metrics after `expm1`, plus log-space companions.

---

## 6. Reproducing results

```bash
pip install -e ".[dev,demo]"
# sudo apt-get install -y python3-dev   # Linux, for PyMC/PyTensor

pytest -q
python scripts/run_comparison.py
# HPP_FAST=1 python scripts/run_comparison.py   # shorter Optuna (CI)

streamlit run app/streamlit_app.py
docker compose up --build
```

### Artifacts written by `run_comparison.py`

| Path | Contents |
|---|---|
| `artifacts/model_comparison.json` | Metrics, calibration, Optuna params, quantile stats |
| `artifacts/ablation_table.json` | Feature-set ladder |
| `artifacts/models/best_model.joblib` | Best hold-out R² model + schema |
| `artifacts/submission.csv` | Kaggle upload file |
| `artifacts/shap_summary.png` / `shap_bar.png` | Interpretability plots |

### Notebooks

| # | File | Focus |
|---|---|---|
| 01 | `01_exploratory_analysis.ipynb` | EDA |
| 02 | `02_least_squares.ipynb` | OLS |
| 03 | `03_ridge_regression.ipynb` | Ridge + CV |
| 04 | `04_bayesian_regression.ipynb` | Conjugate Bayes |
| 05 | `05_mcmc_pymc.ipynb` | PyMC NUTS |
| 06 | `06_model_comparison.ipynb` | Bake-off entrypoint |
| 07 | `07_diagnostics.ipynb` | Residuals / influence |
| 08 | `08_calibration.ipynb` | Coverage + PIT |
| 09 | `09_hierarchical_bayes.ipynb` | Partial pooling |
| 10 | `10_shap_interpretability.ipynb` | SHAP |
| 11 | `11_quantile_intervals.ipynb` | Quantile bands |

Original notebooks live under `notebooks/archive/` (legacy paths / `pymc3`). Prefer the package + numbered notebooks.

---

## 7. Design decisions

1. **From-scratch + library baselines** — interview-ready math with sklearn / LightGBM parity checks.
2. **`src/` layout** — importable, tested logic; thin notebooks.
3. **`log1p` target by default** — standard for skewed prices.
4. **OOF neighborhood encoding** — no target leakage into CV folds on the train matrix.
5. **Production bundle = best hold-out R²** — currently the Ridge+LightGBM stack on this split.
6. **Honest uncertainty** — publish empirical coverage (Bayesian ≈ well-calibrated; quantiles often undercover).
7. **CI** — pytest + `HPP_FAST=1` comparison smoke on PRs.
8. **Docker** — reproducible Streamlit demo without local dependency friction.

---

## 8. App, Docker, submission

```bash
python scripts/run_comparison.py
streamlit run app/streamlit_app.py
docker compose up --build   # http://localhost:8501
```

The Streamlit app loads `artifacts/models/best_model.joblib` when present; otherwise it falls back to a quick Bayesian MAP fit. Unspecified columns are filled with training medians — **demo only**, not a complete listing form.

---

## 9. Limitations

- Ames-only; not transferable to other markets without retraining.
- ~1.5k rows: linear models remain highly competitive; boosting alone may not dominate R².
- Quantile 5–95% bands can undercover vs nominal 90% — check JSON before claiming calibration.
- Hierarchical Bayes / MCMC notebooks subsample for interactive runtime.
- Neighborhood / zoning features can proxy sensitive structure; see the model card.
- No formal hyperparameter nested CV around the full stack (Optuna is inner-CV on LightGBM only).

---

## 10. Related: model card

Operational intended-use, ethics, and maintenance notes: **[MODEL_CARD.md](MODEL_CARD.md)**.

---

## 11. Changelog

### v1.2 — high-value polish
- Leakage-safe **OOF target encoding** for `Neighborhood`
- One-hot `MSZoning`, `SaleCondition`, `GarageType`
- **Optuna** LightGBM + **Ridge+LightGBM stack**
- SHAP plots; quantile LightGBM intervals
- Model card; Docker / Compose; notebooks `10`–`11`
- `HPP_FAST` CI mode

### v1.1 — enhancement pass
- Feature engineering (ordinals, neighborhood encoding, interaction)
- Lasso / Elastic Net + LightGBM baseline
- Calibration + diagnostics; hierarchical Bayes
- joblib persistence, Kaggle submission, Streamlit demo
- GitHub Actions CI; ablation table; notebooks `07`–`09`

### v1.0 — revival
- Installable package, tests, requirements, comparison CLI
- Fixed Bayesian σ² algebra and ridge λ selection
- Broader default features; PyMC 3 → 5
- Selling README + documentation; archived originals
