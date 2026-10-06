"""Feature engineering for Ames Housing: ordinals, safe target encoding, one-hots, interactions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .data import DEFAULT_FEATURES

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

# Nominal categoricals → one-hot (kept small for linear + tree models)
ONEHOT_COLS: List[str] = ["MSZoning", "SaleCondition", "GarageType"]


@dataclass
class FeatureSchema:
    """Serializable description of the engineered feature matrix."""

    feature_names: List[str]
    numeric_cols: List[str]
    ordinal_cols: List[str]
    onehot_cols: List[str]
    onehot_levels: Dict[str, List[str]]
    neighborhood_levels: List[str]
    neighborhood_means: Dict[str, float]
    global_target_mean: float
    numeric_medians: Dict[str, float]
    log_target: bool = True
    include_interactions: bool = True
    include_neighborhood: bool = True
    include_onehot: bool = True
    oof_target_encoding: bool = False


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


def _impute_numeric(
    train: pd.DataFrame, other: pd.DataFrame, cols: Sequence[str]
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    medians: Dict[str, float] = {}
    train_out = train.copy()
    other_out = other.copy()
    for col in cols:
        if col not in train_out.columns:
            train_out[col] = 0.0
        if col not in other_out.columns:
            other_out[col] = 0.0
        median = float(train_out[col].median()) if train_out[col].notna().any() else 0.0
        medians[col] = median
        train_out[col] = train_out[col].fillna(median)
        other_out[col] = other_out[col].fillna(median)
    return train_out, other_out, medians


def fit_target_means(
    categories: pd.Series,
    y: np.ndarray,
    min_count: int = 5,
) -> Tuple[Dict[str, float], float]:
    """Fit category → mean(y) map with global-mean fallback for rare levels."""
    y = np.asarray(y, dtype=float).reshape(-1)
    global_mean = float(np.mean(y))
    stats = (
        pd.DataFrame({"cat": categories.astype(str).to_numpy(), "y": y})
        .groupby("cat")["y"]
        .agg(["mean", "count"])
    )
    means: Dict[str, float] = {}
    for cat, row in stats.iterrows():
        means[str(cat)] = float(row["mean"]) if row["count"] >= min_count else global_mean
    return means, global_mean


def apply_target_means(
    categories: pd.Series,
    means: Dict[str, float],
    global_mean: float,
) -> np.ndarray:
    return categories.astype(str).map(lambda v: means.get(v, global_mean)).to_numpy(dtype=float)


def target_encode_oof(
    categories: pd.Series,
    y: np.ndarray,
    n_splits: int = 5,
    min_count: int = 5,
    random_state: int = 42,
) -> Tuple[np.ndarray, Dict[str, float], float]:
    """
    Out-of-fold target encoding (leakage-safe for the training matrix).

    Each row's encoded value is computed from folds that do **not** contain that row.
    Also returns the full-data means for transforming unseen frames.
    """
    y = np.asarray(y, dtype=float).reshape(-1)
    cats = categories.astype(str).reset_index(drop=True)
    n = len(cats)
    rng = np.random.default_rng(random_state)
    idx = rng.permutation(n)
    folds = np.array_split(idx, n_splits)
    encoded = np.zeros(n, dtype=float)

    for i in range(n_splits):
        val_idx = folds[i]
        train_idx = np.concatenate([folds[j] for j in range(n_splits) if j != i])
        means, gmean = fit_target_means(cats.iloc[train_idx], y[train_idx], min_count=min_count)
        encoded[val_idx] = apply_target_means(cats.iloc[val_idx], means, gmean)

    full_means, full_gmean = fit_target_means(cats, y, min_count=min_count)
    return encoded, full_means, full_gmean


def _onehot_fit_transform(
    train: pd.DataFrame,
    other: pd.DataFrame,
    cols: Sequence[str],
) -> Tuple[np.ndarray, np.ndarray, List[str], Dict[str, List[str]]]:
    """Fit one-hot levels on train; transform train and other with aligned columns."""
    levels: Dict[str, List[str]] = {}
    names: List[str] = []
    train_blocks: List[np.ndarray] = []
    other_blocks: List[np.ndarray] = []

    for col in cols:
        if col not in train.columns:
            continue
        tr = train[col].fillna("Missing").astype(str)
        ot = (
            other[col].fillna("Missing").astype(str)
            if col in other.columns
            else pd.Series(["Missing"] * len(other), dtype=str)
        )
        uniq = sorted(tr.unique().tolist())
        levels[col] = uniq
        for level in uniq:
            names.append(f"{col}__{level}")
            train_blocks.append((tr == level).to_numpy(dtype=float))
            other_blocks.append((ot == level).to_numpy(dtype=float) if len(other) else np.zeros(0))

    if not names:
        return (
            np.zeros((len(train), 0)),
            np.zeros((len(other), 0)),
            [],
            {},
        )
    return np.column_stack(train_blocks), np.column_stack(other_blocks), names, levels


def _onehot_transform(frame: pd.DataFrame, levels: Dict[str, List[str]]) -> np.ndarray:
    blocks = []
    for col, uniq in levels.items():
        series = (
            frame[col].fillna("Missing").astype(str)
            if col in frame.columns
            else pd.Series(["Missing"] * len(frame), dtype=str)
        )
        for level in uniq:
            blocks.append((series == level).to_numpy(dtype=float))
    return np.column_stack(blocks) if blocks else np.zeros((len(frame), 0))


def build_feature_frame(
    train: pd.DataFrame,
    other: Optional[pd.DataFrame] = None,
    *,
    target: str = "SalePrice",
    log_target: bool = True,
    include_interactions: bool = True,
    include_neighborhood: bool = True,
    include_onehot: bool = True,
    drop_outliers: bool = True,
    oof_target_encoding: bool = True,
    n_oof_splits: int = 5,
    random_state: int = 42,
) -> Tuple[EngineeredData, EngineeredData]:
    """
    Build aligned engineered matrices for train and ``other`` (val/test).

    When ``oof_target_encoding=True`` (default), neighborhood encodings on the
    **training** matrix are out-of-fold, so later CV / hold-out work does not
    leak target information. Full-data means are still stored on the schema for
    transforming ``other`` / production rows.
    """
    other = train.iloc[0:0].copy() if other is None else other.copy()
    train = train.copy()

    if drop_outliers and "GrLivArea" in train.columns and target in train.columns:
        train = train.loc[train["GrLivArea"] <= 4000].copy()

    train = _map_ordinals(train)
    other = _map_ordinals(other)

    numeric_cols = [c for c in NUMERIC_BASE if c in train.columns or c in other.columns]
    for col in ORDINAL_QUALITY_COLS:
        if col in train.columns:
            numeric_cols.append(col)
    seen: set = set()
    numeric_cols = [c for c in numeric_cols if not (c in seen or seen.add(c))]

    train, other, medians = _impute_numeric(train, other, numeric_cols)

    parts_tr: List[np.ndarray] = []
    parts_ot: List[np.ndarray] = []
    names: List[str] = []

    for col in numeric_cols:
        parts_tr.append(train[col].to_numpy(dtype=float))
        parts_ot.append(other[col].to_numpy(dtype=float) if len(other) else np.zeros(0))
        names.append(col)

    y_train = None
    if target in train.columns:
        y_raw = train[target].to_numpy(dtype=float)
        y_train = np.log1p(y_raw) if log_target else y_raw

    neigh_means: Dict[str, float] = {}
    global_mean = 0.0
    neigh_levels: List[str] = []
    if include_neighborhood and "Neighborhood" in train.columns and y_train is not None:
        if oof_target_encoding:
            tr_enc, neigh_means, global_mean = target_encode_oof(
                train["Neighborhood"],
                y_train,
                n_splits=n_oof_splits,
                random_state=random_state,
            )
        else:
            neigh_means, global_mean = fit_target_means(train["Neighborhood"], y_train)
            tr_enc = apply_target_means(train["Neighborhood"], neigh_means, global_mean)
        ot_enc = (
            apply_target_means(other["Neighborhood"], neigh_means, global_mean)
            if len(other)
            else np.zeros(0)
        )
        parts_tr.append(tr_enc)
        parts_ot.append(ot_enc)
        names.append("NeighborhoodEnc")
        neigh_levels = sorted(neigh_means.keys())

    if include_interactions and "OverallQual" in train.columns and "GrLivArea" in train.columns:
        parts_tr.append((train["OverallQual"] * train["GrLivArea"]).to_numpy(dtype=float))
        parts_ot.append(
            (other["OverallQual"] * other["GrLivArea"]).to_numpy(dtype=float)
            if len(other)
            else np.zeros(0)
        )
        names.append("OverallQual_x_GrLivArea")

    onehot_names: List[str] = []
    onehot_levels: Dict[str, List[str]] = {}
    if include_onehot:
        oh_tr, oh_ot, onehot_names, onehot_levels = _onehot_fit_transform(train, other, ONEHOT_COLS)
        if onehot_names:
            # append columns individually for naming consistency
            for j, name in enumerate(onehot_names):
                parts_tr.append(oh_tr[:, j])
                parts_ot.append(oh_ot[:, j] if len(other) else np.zeros(0))
                names.append(name)

    X_train = np.column_stack(parts_tr) if parts_tr else np.zeros((len(train), 0))
    X_other = np.column_stack(parts_ot) if parts_ot and len(other) else np.zeros((len(other), len(names)))

    y_other = None
    if target in other.columns and len(other):
        y_raw_o = other[target].to_numpy(dtype=float)
        y_other = np.log1p(y_raw_o) if log_target else y_raw_o

    schema = FeatureSchema(
        feature_names=names,
        numeric_cols=numeric_cols,
        ordinal_cols=[c for c in ORDINAL_QUALITY_COLS if c in names],
        onehot_cols=list(onehot_levels.keys()),
        onehot_levels=onehot_levels,
        neighborhood_levels=neigh_levels,
        neighborhood_means=neigh_means,
        global_target_mean=global_mean,
        numeric_medians=medians,
        log_target=log_target,
        include_interactions=include_interactions,
        include_neighborhood=include_neighborhood,
        include_onehot=include_onehot,
        oof_target_encoding=oof_target_encoding,
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
            cols.append(
                apply_target_means(
                    df["Neighborhood"] if "Neighborhood" in df.columns else pd.Series(["Missing"] * len(df)),
                    schema.neighborhood_means,
                    schema.global_target_mean,
                )
            )
        elif name == "OverallQual_x_GrLivArea":
            oq = df["OverallQual"] if "OverallQual" in df.columns else 0
            gl = df["GrLivArea"] if "GrLivArea" in df.columns else 0
            cols.append((pd.Series(oq).fillna(0) * pd.Series(gl).fillna(0)).to_numpy(dtype=float))
        elif "__" in name and name.split("__", 1)[0] in schema.onehot_levels:
            col, level = name.split("__", 1)
            series = (
                df[col].fillna("Missing").astype(str)
                if col in df.columns
                else pd.Series(["Missing"] * len(df), dtype=str)
            )
            cols.append((series == level).to_numpy(dtype=float))
        else:
            if name in df.columns:
                series = df[name]
                if not pd.api.types.is_numeric_dtype(series):
                    series = pd.to_numeric(series, errors="coerce")
                fill = schema.numeric_medians.get(name, 0.0)
                cols.append(series.fillna(fill).to_numpy(dtype=float))
            else:
                cols.append(np.full(len(df), schema.numeric_medians.get(name, 0.0), dtype=float))
    return np.column_stack(cols) if cols else np.zeros((len(df), 0))
