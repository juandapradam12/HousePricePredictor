"""Streamlit demo: predict Ames sale price with uncertainty when available."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from house_price_predictor import (  # noqa: E402
    BayesianLinearRegression,
    build_feature_frame,
    load_housing_data,
)
from house_price_predictor.persistence import load_model_bundle  # noqa: E402
from house_price_predictor.features import QUALITY_MAP  # noqa: E402

BUNDLE = ROOT / "artifacts" / "models" / "best_model.joblib"


@st.cache_resource
def load_bundle_or_fallback():
    if BUNDLE.exists():
        return load_model_bundle(BUNDLE), "saved"
    train, _ = load_housing_data()
    eng_tr, _ = build_feature_frame(train, log_target=True)
    model = BayesianLinearRegression(lambda_param=0.1).fit(eng_tr.X, eng_tr.y)
    return {"model": model, "schema": eng_tr.schema, "metadata": {"model_name": "Bayesian MAP"}}, "fallback"


def main() -> None:
    st.set_page_config(page_title="House Price Predictor", page_icon="🏠", layout="centered")
    st.title("Ames House Price Predictor")
    st.caption("Interactive demo backed by the trained model bundle (or Bayesian MAP fallback).")

    bundle, source = load_bundle_or_fallback()
    model = bundle["model"]
    schema = bundle["schema"]
    meta = bundle.get("metadata") or {}
    st.write(f"Model: **{meta.get('model_name', type(model).__name__)}** ({source})")

    train, _ = load_housing_data()
    neighborhoods = sorted(train["Neighborhood"].dropna().astype(str).unique().tolist())

    c1, c2 = st.columns(2)
    with c1:
        overall_qual = st.slider("Overall quality", 1, 10, 7)
        gr_liv = st.number_input("Living area (GrLivArea)", 500, 4000, 1500, step=50)
        year_built = st.slider("Year built", 1870, 2010, 2000)
        garage_cars = st.slider("Garage cars", 0, 4, 2)
    with c2:
        total_bsmt = st.number_input("Total basement SF", 0, 3000, 900, step=50)
        full_bath = st.slider("Full baths", 0, 4, 2)
        neighborhood = st.selectbox("Neighborhood", neighborhoods, index=min(5, len(neighborhoods) - 1))
        kitchen_qual = st.selectbox("Kitchen quality", list(QUALITY_MAP.keys()), index=1)

    # Build a one-row frame with sensible defaults from train medians
    row = train.median(numeric_only=True).to_dict()
    row.update(
        {
            "OverallQual": overall_qual,
            "GrLivArea": gr_liv,
            "YearBuilt": year_built,
            "YearRemodAdd": max(year_built, 1950),
            "GarageCars": garage_cars,
            "GarageArea": max(200, garage_cars * 280),
            "TotalBsmtSF": total_bsmt,
            "1stFlrSF": max(total_bsmt, gr_liv // 2),
            "FullBath": full_bath,
            "TotRmsAbvGrd": max(5, full_bath + 3),
            "Neighborhood": neighborhood,
            "KitchenQual": kitchen_qual,
            "ExterQual": kitchen_qual,
            "BsmtQual": kitchen_qual,
            "HeatingQC": "Ex",
            "Id": 0,
        }
    )
    frame = pd.DataFrame([row])

    from house_price_predictor.features import transform_with_schema

    X = transform_with_schema(frame, schema)
    pred = model.predict(X)
    try:
        mean, std = model.predict(X, return_std=True)
        mean, std = float(np.asarray(mean).ravel()[0]), float(np.asarray(std).ravel()[0])
        if schema.log_target:
            lo, hi = np.expm1(mean - 1.96 * std), np.expm1(mean + 1.96 * std)
            point = float(np.expm1(mean))
            st.metric("Predicted sale price", f"${point:,.0f}")
            st.write(f"Approx. 95% interval: **${lo:,.0f} – ${hi:,.0f}**")
        else:
            st.metric("Predicted sale price", f"${mean:,.0f}")
            st.write(f"Approx. 95% interval: **${mean - 1.96 * std:,.0f} – ${mean + 1.96 * std:,.0f}**")
    except TypeError:
        pred = float(np.asarray(pred).ravel()[0])
        if schema.log_target:
            pred = float(np.expm1(pred))
        st.metric("Predicted sale price", f"${pred:,.0f}")

    with st.expander("Feature vector"):
        st.write(dict(zip(schema.feature_names, X.ravel().round(3))))


if __name__ == "__main__":
    main()
