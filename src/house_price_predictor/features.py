"""Feature engineering for Ames Housing: ordinals, target encoding, interactions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .data import DEFAULT_FEATURES

# Standard Ames quality scale (missing / NA treated as 0 = none)
QUALITY_MAP: Dict[str, int] = {
    "Ex": 5,
    "Gd": 4,
    "TA": 3,
    "Fa": 2,
    "Po": 1,
}

ORDINAL_QUALITY_COLS: List[str] = [
    "ExterQual",
    "ExterCond",
    "BsmtQual",
    "BsmtCond",
    "HeatingQC",
    "KitchenQual",
    "FireplaceQu",
    "GarageQual",
    "GarageCond",
]

NUMERIC_BASE: List[str] = list(DEFAULT_FEATURES) + [
    "LotArea",
    "MasVnrArea",
    "BsmtFinSF1",
    "WoodDeckSF",
    "OpenPorchSF",
    "Fireplaces",
    "LotFrontage",
]


@dataclass
class FeatureSchema:
    """Serializable description of the engineered feature matrix."""

    feature_names: List[str]
    numeric_cols: List[str]
    ordinal_cols: List[str]
    neighborhood_levels: List[str]
    neighborhood_means: Dict[str, float]
    global_target_mean: float
    log_target: bool = True
    include_interactions: bool = True
    include_neighborhood: bool = True


@dataclass
class EngineeredData:
    """Train/test matrices plus schema and aligned IDs."""

    X: np.ndarray
    y: Optional[np.ndarray]
    ids: np.ndarray
    schema: FeatureSchema
    frame: pd.DataFrame = field(repr=False)


def _map_ordinals(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for col in ORDINAL_QUALITY_COLS:
        if col not in out.columns:
            continue
        if pd.api.types.is_numeric_dtype(out[col]):
            out[col] = out[col].fillna(0).astype(float)
            continue
        out[col] = out[col].map(QUALITY_MAP).fillna(0).astype(float)
    return out


def _impute_numeric(train: pd.DataFrame, other: pd.DataFrame, cols: Sequence[str]) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    medians: Dict[str, float] = {}
    train_out = train.copy()
    other_out = other.copy()
    for col in cols:
        if col not in train_out.columns:
            train_out[col] = 0.0
            other_out[col] = 0.0
        median = float(train_out[col].median()) if train_out[col].notna().any() else 0.0
        medians[col] = median
        train_out[col] = train_out[col].fillna(median)
        if col not in other_out.columns:
            other_out[col] = median
        else:
            other_out[col] = other_out[col].fillna(median)
    return train_out, other_out, medians


def _target_encode_neighborhood(
    train: pd.DataFrame,
    other: pd.DataFrame,
    target: str = "SalePrice",
    log_target: bool = True,
    min_count: int = 5,
) -> Tuple[pd.Series, pd.Series, Dict[str, float], float, List[str]]:
    """Mean target encoding with global-mean fallback for rare / unseen levels."""
    y = np.log1p(train[target].to_numpy(dtype=float)) if log_target else train[target].to_numpy(dtype=float)
    global_mean = float(np.mean(y))
    stats = (
        pd.DataFrame({"Neighborhood": train["Neighborhood"].astype(str), "y": y})
        .groupby("Neighborhood")["y"]
        .agg(["mean", "count"])
    )
    means: Dict[str, float] = {}
    for neigh, row in stats.iterrows():
        if row["count"] >= min_count:
            means[str(neigh)] = float(row["mean"])
        else:
            means[str(neigh)] = global_mean

    def encode(series: pd.Series) -> pd.Series:
        return series.astype(str).map(lambda v: means.get(v, global_mean)).astype(float)

    levels = sorted(means.keys())
    return encode(train["Neighborhood"]), encode(other["Neighborhood"]), means, global_mean, levels


def build_feature_frame(
    train: pd.DataFrame,
    other: Optional[pd.DataFrame] = None,
    *,
    target: str = "SalePrice",
    log_target: bool = True,
    include_interactions: bool = True,
    include_neighborhood: bool = True,
    drop_outliers: bool = True,
) -> Tuple[EngineeredData, EngineeredData]:
    """
    Build aligned engineered matrices for train and ``other`` (val/test).

    Engineering steps
    -----------------
    1. Drop extreme ``GrLivArea`` outliers on train only (Ames cookbook).
    2. Map ordinal quality strings to integers.
    3. Median-impute numeric columns from train statistics.
    4. Target-encode ``Neighborhood`` (fit on train only).
    5. Optional interaction: ``OverallQual * GrLivArea``.
    """
    other = train.iloc[0:0].copy() if other is None else other.copy()
    train = train.copy()

    if drop_outliers and "GrLivArea" in train.columns and target in train.columns:
        # Partial-sale mansions commonly removed in Ames tutorials
        train = train.loc[train["GrLivArea"] <= 4000].copy()

    train = _map_ordinals(train)
    other = _map_ordinals(other)

    numeric_cols = [c for c in NUMERIC_BASE if c in train.columns or c in other.columns]
    # ensure ordinals present as numeric features
    for col in ORDINAL_QUALITY_COLS:
        if col in train.columns:
            numeric_cols.append(col)
    # de-dupe preserving order
    seen = set()
    numeric_cols = [c for c in numeric_cols if not (c in seen or seen.add(c))]

    train, other, _ = _impute_numeric(train, other, numeric_cols)

    feature_parts_train: List[pd.Series] = []
    feature_parts_other: List[pd.Series] = []
    names: List[str] = []

    for col in numeric_cols:
        feature_parts_train.append(train[col].astype(float))
        feature_parts_other.append(other[col].astype(float) if len(other) else pd.Series(dtype=float))
        names.append(col)

    neigh_means: Dict[str, float] = {}
    global_mean = 0.0
    neigh_levels: List[str] = []
    if include_neighborhood and "Neighborhood" in train.columns:
        tr_enc, ot_enc, neigh_means, global_mean, neigh_levels = _target_encode_neighborhood(
            train, other if len(other) else train, target=target, log_target=log_target
        )
        feature_parts_train.append(tr_enc)
        feature_parts_other.append(ot_enc if len(other) else tr_enc.iloc[0:0])
        names.append("NeighborhoodEnc")

    if include_interactions and "OverallQual" in train.columns and "GrLivArea" in train.columns:
        feature_parts_train.append((train["OverallQual"] * train["GrLivArea"]).astype(float))
        if len(other):
            feature_parts_other.append((other["OverallQual"] * other["GrLivArea"]).astype(float))
        else:
            feature_parts_other.append(pd.Series(dtype=float))
        names.append("OverallQual_x_GrLivArea")

    X_train = pd.concat(feature_parts_train, axis=1).to_numpy(dtype=float)
    X_other = (
        pd.concat(feature_parts_other, axis=1).to_numpy(dtype=float)
        if len(other)
        else np.zeros((0, len(names)), dtype=float)
    )

    y_train = None
    if target in train.columns:
        y_raw = train[target].to_numpy(dtype=float)
        y_train = np.log1p(y_raw) if log_target else y_raw

    y_other = None
    if target in other.columns and len(other):
        y_raw_o = other[target].to_numpy(dtype=float)
        y_other = np.log1p(y_raw_o) if log_target else y_raw_o

    schema = FeatureSchema(
        feature_names=names,
        numeric_cols=numeric_cols,
        ordinal_cols=[c for c in ORDINAL_QUALITY_COLS if c in names],
        neighborhood_levels=neigh_levels,
        neighborhood_means=neigh_means,
        global_target_mean=global_mean,
        log_target=log_target,
        include_interactions=include_interactions,
        include_neighborhood=include_neighborhood,
    )

    train_ids = train["Id"].to_numpy() if "Id" in train.columns else np.arange(len(train))
    other_ids = other["Id"].to_numpy() if "Id" in other.columns else np.arange(len(other))

    return (
        EngineeredData(X=X_train, y=y_train, ids=train_ids, schema=schema, frame=train),
        EngineeredData(X=X_other, y=y_other, ids=other_ids, schema=schema, frame=other),
    )


def transform_with_schema(frame: pd.DataFrame, schema: FeatureSchema) -> np.ndarray:
    """Apply a fitted ``FeatureSchema`` to a new dataframe (e.g. Kaggle test)."""
    df = _map_ordinals(frame.copy())
    cols = []
    for name in schema.feature_names:
        if name == "NeighborhoodEnc":
            enc = (
                df["Neighborhood"]
                .astype(str)
                .map(lambda v: schema.neighborhood_means.get(v, schema.global_target_mean))
                .astype(float)
            )
            cols.append(enc.to_numpy())
        elif name == "OverallQual_x_GrLivArea":
            cols.append((df["OverallQual"].fillna(0) * df["GrLivArea"].fillna(0)).to_numpy(dtype=float))
        else:
            series = df[name] if name in df.columns else pd.Series(0.0, index=df.index)
            cols.append(series.fillna(0).to_numpy(dtype=float))
    return np.column_stack(cols) if cols else np.zeros((len(df), 0))
