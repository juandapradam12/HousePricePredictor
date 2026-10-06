# Documentation — House Price Predictor

This document explains the modeling stack, the mathematical corrections made during the revival of this project, the Python API, and how to reproduce results.

---

## 1. Problem statement

Given tabular attributes of residential properties in Ames, Iowa, estimate sale price. The training file contains 1,460 labeled homes and ~80 raw attributes. This project focuses on **linear models you can derive**, plus Elastic Net / LightGBM baselines and hierarchical Bayes so the portfolio also shows a modern end-to-end pipeline.

We model

\[
\log(1 + \text{SalePrice}) \approx w_0 + \sum_{j=1}^{p} w_j x_j
\]

using features with strong Pearson correlation to the target (\(|r| \ge 0.5\) by default). The log transform reduces right-skew and stabilizes variance; metrics are also reported in USD via `expm1`.

### Default feature set

`OverallQual`, `GrLivArea`, `GarageCars`, `GarageArea`, `TotalBsmtSF`, `1stFlrSF`, `FullBath`, `TotRmsAbvGrd`, `YearBuilt`, `YearRemodAdd`.

---

## 2. Algorithms

### 2.1 Ordinary least squares

Textbook estimator:

\[
w_{\mathrm{LS}} = (X^\top X)^{-1} X^\top y
\]

**Implementation:** `np.linalg.lstsq` on an intercept-augmented design matrix. Equivalent when \(X\) has full column rank, but better conditioned than forming \(X^\top X\) and inverting it.

**API:** `LeastSquaresRegressor`, `least_squares_weights`.

### 2.2 Ridge regression

\[
w_{\mathrm{RR}} = (X^\top X + \lambda I)^{-1} X^\top y
\]

**Details:**

- Features are z-scored; the target is mean-centered during fitting.
- The intercept is **not** penalized (`penalty[0,0] = 0`), matching scikit-learn.
- \(\lambda\) is chosen by **5-fold cross-validation** minimizing RMSE on the log target (`select_lambda_cv`).

**Why this changed:** the original notebook searched \(\lambda\) by minimizing mean absolute residual on the first 50 rows of training data — not a validation procedure, and easily overfit.

**API:** `RidgeRegressor`, `ridge_regression_weights`, `select_lambda_cv`.

### 2.3 Bayesian linear regression (conjugate Gaussian)

Prior and likelihood:

\[
p(w) = \mathcal{N}(0, \lambda^{-1} I), \qquad
p(y \mid w, X) = \mathcal{N}(Xw, \sigma^2 I)
\]

Posterior:

\[
\Sigma = \big(\lambda I + \sigma^{-2} X^\top X\big)^{-1}, \qquad
\mu = \big(\lambda \sigma^2 I + X^\top X\big)^{-1} X^\top y
\]

Noise level \(\sigma^2\) is estimated from OLS residuals:

\[
\hat\sigma^2 = \frac{1}{n-d}\sum_{i=1}^{n}(y_i - x_i^\top w_{\mathrm{LS}})^2
\]

Predictive distribution for a new point \(x_0\):

\[
\mu_0 = x_0^\top \mu, \qquad
\sigma_0^2 = \sigma^2 + x_0^\top \Sigma x_0
\]

**Bug fixes vs. original notebook:**

| Location | Original issue | Fix |
|---|---|---|
| MAP mean | Used \(\lambda \sigma\) instead of \(\lambda \sigma^2\) on the diagonal | `map_coefficients` uses \(\lambda \sigma^2 I\) |
| Posterior Σ | Divided by `sigma` while the docstring said \(\sigma^2\) | `posterior_covariance` uses \(1/\sigma^2\) |
| `predict_bayes_reg` | Called `predict` without the required `sigma` argument | Class API bundles σ² in the posterior object |
| Feature scale | Raw `GrLivArea` / `YearBuilt` with isotropic prior | Features standardized inside `BayesianLinearRegression.fit` |

**API:** `BayesianLinearRegression` (`.predict(..., return_std=True)` for intervals).

### 2.4 MCMC with PyMC 5

When conjugacy is dropped (or for illustration), we sample with NUTS:

- `intercept ~ Normal(mean(y), 10)`
- `beta ~ Normal(0, 1)` (on standardized features)
- `sigma ~ HalfNormal(1)` — fixes the original `Normal` prior on a scale parameter that could go negative
- Likelihood `y ~ Normal(μ, σ)`

**Migration:** `pymc3` → `pymc>=5` + ArviZ for traces/posterior plots. Helper: `house_price_predictor.mcmc.fit_bayesian_mcmc`.

---

## 3. Data utilities

| Function | Role |
|---|---|
| `load_housing_data()` | Reads `data/train.csv` and `data/test.csv` |
| `correlation_with_target` | Pearson correlations (numeric columns only) |
| `select_correlated_features` | Thresholded feature list (no global state) |
| `prepare_xy` | Design matrix + optional `log1p` target |
| `train_val_split` | Reproducible hold-out split |
| `apply_cutoffs` | Row filters e.g. outlier caps on `GrLivArea` |

**Original bug:** `select_corr_columns(data_frame)` iterated over the global `data` variable. The new helper always uses the frame you pass in.

---

## 4. Metrics

Implemented in `house_price_predictor.metrics`:

- `rmse`, `mae`, `r2_score`, `regression_report`

The comparison script reports both log-space and USD-space errors.

---

## 5. Reproducing results

```bash
pip install -e ".[dev]"
# Linux: install headers once so PyMC/PyTensor can compile extensions
# sudo apt-get install -y python3-dev
pytest -q
python scripts/run_comparison.py
```

Notebooks (run from repo root or `notebooks/` — paths auto-detect):

1. `01_exploratory_analysis.ipynb`
2. `02_least_squares.ipynb`
3. `03_ridge_regression.ipynb`
4. `04_bayesian_regression.ipynb`
5. `05_mcmc_pymc.ipynb` (slower; sampling)
6. `06_model_comparison.ipynb`
7. `07_diagnostics.ipynb`
8. `08_calibration.ipynb`
9. `09_hierarchical_bayes.ipynb`

Original notebooks are preserved under `notebooks/archive/` for reference. They expect CSVs in the working directory and `pymc3`; prefer the new package for anything you extend.

---

## 6. Feature engineering (`features.py`)

`build_feature_frame(train, other)` returns aligned `EngineeredData` objects:

1. Drop `GrLivArea > 4000` on train
2. Map ordinal quality columns (`Ex…Po`) to integers
3. Median-impute numerics from train statistics
4. Target-encode `Neighborhood` (rare / unseen → global mean)
5. Add `OverallQual × GrLivArea`
6. Optional `log1p` target

`transform_with_schema` applies a fitted `FeatureSchema` to Kaggle test rows.

---

## 7. Additional estimators

| Module | API |
|---|---|
| `elastic_net.py` | `ElasticNetRegressor`, `LassoRegressor`, `select_elastic_net_cv`, `select_lasso_cv` |
| `boosting.py` | `LightGBMRegressor` (optional if LightGBM installed) |
| `hierarchical.py` | `fit_hierarchical_neighborhood`, `hierarchical_predict_mean` |
| `calibration.py` | coverage, calibration curve, PIT, `calibration_report` |
| `diagnostics.py` | residuals, leverage, Cook’s distance, learning curves |
| `persistence.py` | `save_model_bundle` / `load_model_bundle` (joblib) |
| `submission.py` | `make_submission` → `Id,SalePrice` CSV |

---

## 8. Design decisions

1. **From-scratch + sklearn / LightGBM checks** — educational core with a clear performance ceiling.
2. **Package under `src/`** — notebooks stay thin; logic is importable and tested.
3. **Log target by default** — standard for this dataset; disable with `log_target=False`.
4. **Train-only encoding stats** — neighborhood means and medians never leak from validation/test.
5. **Production bundle** picks the best hold-out model by USD R² (often Ridge or Elastic Net on this feature set).
6. **CI** runs pytest + `scripts/run_comparison.py` on pushes/PRs.

---

## 9. App & submission

```bash
python scripts/run_comparison.py
# → artifacts/model_comparison.json
# → artifacts/ablation_table.json
# → artifacts/models/best_model.joblib
# → artifacts/submission.csv

streamlit run app/streamlit_app.py
```

---

## 10. Limitations

- LightGBM hyperparameters are sensible defaults, not a full Optuna sweep.
- Hierarchical Bayes notebook subsamples for interactive runtime.
- Streamlit demo uses medians for unspecified columns — illustrative, not a full form for every Ames field.
- Target encoding can overfit small neighborhoods; CV-aware encoding is a natural follow-up.

---

## 11. Changelog

### v1.1 — full enhancement pass
- Feature engineering (ordinals, neighborhood encoding, interaction)
- Lasso / Elastic Net + LightGBM baseline
- Bayesian calibration tools + diagnostics
- Hierarchical Bayes by neighborhood
- joblib persistence, Kaggle submission, Streamlit demo
- GitHub Actions CI; ablation table in comparison script
- Notebooks 07–09

### v1.0 — revival
- Installable package, tests, requirements, comparison CLI
- Fixed Bayesian σ² algebra and ridge λ selection
- Expanded features beyond `GrLivArea` + `YearBuilt`
- Migrated MCMC stack to PyMC 5
- Selling README + documentation; archived original notebooks
