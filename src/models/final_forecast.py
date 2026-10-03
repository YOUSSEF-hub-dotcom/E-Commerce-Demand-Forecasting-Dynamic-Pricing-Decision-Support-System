"""Frozen Phase-3 final forecasting pipeline (production candidate).

Selected by the controlled-experiments cycle (05/06): **log-space LightGBM +
the 7 regime-aware features from EXP 2**. This module is the single source of
truth for the frozen candidate:

* exact feature stack (13 base features + 73 one-hot categories + 7 regime)
* exact temporal split (70/15/15, chronological, no leakage)
* exact imputation contract
* exact LogGBM hyperparameters (Optuna winner run ``4c9ad7...``)
* exact target transform (log1p) / prediction transform (expm1)
* exact sample-weight scheme and evaluation suite (incl. B/B2/C cohorts and MASE)

Nothing in this module reads, writes, or tunes on the protected test split;
scoring is delegated to ``finalize_lifecycle`` which unlocks it exactly once.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.imputation import impute_splits
from src.data.target_engineering import load_demand_matrix
from src.data.temporal_split import assert_chronological_order, temporal_split
from src.features.feature_engineering import (
    add_calendar_features,
    add_demand_drivers,
    add_lag_features,
    add_rolling_features,
    add_target_transform,
    PHASE3_LAGS,
    PHASE3_ROLLING_AGGS,
    PHASE3_ROLLING_WINDOWS,
)
from src.models.data_prep import (
    design_matrix_rows,
    make_sample_weights,
    prepare_model_matrix,
)
from src.utils.logging_setup import get_logger
from src.utils.metrics import asymmetric_loss, mae, rmse, wape

LOGGER = get_logger(__name__)

# --------------------------------------------------------------------------
# Freeze constants (identical to the controlled experiment EXP 2)
# --------------------------------------------------------------------------

# Exact Optuna-selected winner hyperparameters (MLflow run `4c9ad7...`).
WINNER_PARAMS: dict = dict(
    colsample_bytree=0.5290,
    learning_rate=0.008468,
    max_depth=10,
    min_child_samples=38,
    n_estimators=600,
    num_leaves=223,
    reg_alpha=2.1423,
    reg_lambda=0.1013,
    subsample=0.6624,
    random_state=1,
    n_jobs=-1,
    verbose=-1,
)

# The 7 regime-aware features added by EXP 2 (all strictly past-only).
REGIME_COLS: list[str] = [
    "rolling_demand_mean_8w",
    "rolling_demand_mean_12w",
    "rolling_demand_mean_16w",
    "rolling_demand_std_8w",
    "regime_ratio_4_12",
    "regime_ratio_8_16",
    "positive_rate_8w",
]
EPS: float = 1e-6  # denominator guard for regime ratios (as in the experiment).

NUMERIC_IMPUTE: list[str] = [
    "avg_price", "avg_freight", "price_prev", "price_chg_1w", "freight_prev",
]

TARGET = "order_demand"
LOG_TARGET = "demand_log1p"
WEIGHT_COL = "sample_weight"
IDENTITY_COLS = ["product_id", "week", TARGET, LOG_TARGET, WEIGHT_COL]

SPLIT_RATIOS = (0.70, 0.15, 0.15)

# Cohort definitions (identical to notebooks 05/06).
B_MIN_POS_WEEKS = 6
B_MIN_AVG_WEEKLY = 1.0
B2_MIN_AVG_WEEKLY = 0.5

FEATURE_VERSION = "phase3-exp2-v1"
TARGET_TRANSFORM = "log1p"
PREDICTION_TRANSFORM = "expm1 (with physical floor at 0)"
WEIGHT_STRATEGY = "1+log1p"
CALIBRATION_CONFIG = "none (deferred; recipe documented, fitted on train-derived CV only)"


def inverse_log1p(
    pred_log,
    floor: float = 0.0,
    finite_fill: float = 0.0,
) -> np.ndarray:
    """Convert log-domain predictions to business (unit) demand.

    Applies ``expm1`` and then enforces the **physical target constraint**
    ``prediction >= 0`` as part of the inference contract:

    * values below ``floor`` clamp to ``floor`` (demand can never be negative),
    * any non-finite result (``NaN``/``Inf`` from overflow or degenerate input)
      becomes ``finite_fill`` so served predictions are always finite.

    This mirrors the documented ``prediction_transform`` of the frozen
    candidate, now with an explicit demand-floor guarantee.
    """
    values = np.expm1(np.asarray(pred_log, dtype=float))
    out = np.where(np.isfinite(values), values, finite_fill).astype("float64")
    return np.clip(out, a_min=floor, a_max=None)


def freeze_config() -> dict:
    """Return the machine-readable freeze configuration."""
    return {
        "feature_version": FEATURE_VERSION,
        "model_type": "lightgbm",
        "objective": "regression",
        "params": dict(WINNER_PARAMS),
        "target_transform": TARGET_TRANSFORM,
        "prediction_transform": PREDICTION_TRANSFORM,
        "weight_strategy": WEIGHT_STRATEGY,
        "calibration": CALIBRATION_CONFIG,
        "split_ratios": list(SPLIT_RATIOS),
        "regime_features": list(REGIME_COLS),
        "numeric_impute": list(NUMERIC_IMPUTE),
        "cohorts": {
            "B": {"min_pos_weeks": B_MIN_POS_WEEKS, "min_avg_weekly": B_MIN_AVG_WEEKLY},
            "B2": {"min_avg_weekly": B2_MIN_AVG_WEEKLY},
        },
    }


# --------------------------------------------------------------------------
# Frozen feature pipeline
# --------------------------------------------------------------------------

def build_frozen_features(matrix: pd.DataFrame) -> pd.DataFrame:
    """Phase-3 feature stack + the 7 EXP2 regime features (one call)."""
    df = matrix.copy()
    df = add_calendar_features(df)
    df = add_demand_drivers(df)
    df = add_lag_features(df, lags=PHASE3_LAGS)
    df = add_rolling_features(df, windows=PHASE3_ROLLING_WINDOWS, aggs=PHASE3_ROLLING_AGGS)
    df = add_target_transform(df)
    df = add_regime_features(df)
    return df


def add_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add the EXP2 regime feature columns (strictly past-only)."""
    df = df.copy()
    df = add_rolling_features(df, windows=(8, 12, 16), aggs=("mean",))
    df = add_rolling_features(df, windows=(8,), aggs=("std",))
    df["regime_ratio_4_12"] = df["rolling_demand_mean_4w"] / (df["rolling_demand_mean_12w"] + EPS)
    df["regime_ratio_8_16"] = df["rolling_demand_mean_8w"] / (df["rolling_demand_mean_16w"] + EPS)
    shifted = df.groupby("product_id", observed=True)["order_demand"].shift(1)
    pos8 = shifted.gt(0).groupby(df["product_id"], observed=True).rolling(8).mean()
    df["positive_rate_8w"] = pos8.reset_index(level=0, drop=True)
    for col in REGIME_COLS:
        df[col] = df[col].astype("float64").fillna(0.0)
    return df


def run_frozen_split(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Temporal (no-leakage) split + train-only imputation of the full frame."""
    splits = temporal_split(frame, "week", ratios=SPLIT_RATIOS)
    assert_chronological_order(splits, "week")
    splits = impute_splits(
        splits,
        categorical_cols=["product_category_name"],
        numeric_cols=NUMERIC_IMPUTE,
    )
    return splits


def prepare_frozen(splits: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Prepare train/val/test design matrices (incl. regime columns) at 93 columns.

    One-hot vocabulary and OHE schema are derived from the **TRAIN split only**
    and then frozen; validation and test are aligned to that frozen layout.
    Because conditional imputation already maps any unseen validation/test
    category to the reserved ``unknown`` label (which exists in the train
    vocabulary), the frozen schema can never be expanded by a later split.
    """
    train_vocab = sorted(splits["train"]["product_category_name"].astype(str).unique())
    for k in ("val", "test"):
        extra = set(splits[k]["product_category_name"].astype(str)) - set(train_vocab)
        if extra:
            raise RuntimeError(
                f"prepare_frozen: {k} contains categories unseen in train: {sorted(extra)}"
            )
    prepared: dict[str, pd.DataFrame] = {}
    for k in ("train", "val", "test"):
        out = prepare_model_matrix(splits[k], categories=train_vocab)
        vals = splits[k][REGIME_COLS].reset_index(drop=True).astype("float64")
        assert not vals.isna().any().any(), f"NaN in regime features ({k})"
        out[REGIME_COLS] = vals
        prepared[k] = out
    return prepared


def freeze_schema(prepared_train: pd.DataFrame) -> list[str]:
    """The exact ordered model-input schema (13 + 73 one-hot + 7 regime = 93)."""
    schema = [c for c in prepared_train.columns if c not in IDENTITY_COLS]
    if len(schema) != 93:
        raise RuntimeError(f"freeze_schema: expected 93 columns, got {len(schema)}.")
    return schema


# --------------------------------------------------------------------------
# Cohort + metric machinery (identical to the experiment notebooks)
# --------------------------------------------------------------------------

def cohort_flags(prepared_train: pd.DataFrame, pids) -> np.ndarray:
    """Assign B-sustained / B2-mid / C-long-tail flags from *training* stats."""
    tr = prepared_train[["product_id", TARGET]].copy()
    tr["product_id"] = tr["product_id"].astype(str)
    g = tr.groupby("product_id")[TARGET]
    stats = pd.DataFrame(
        {"pos": g.apply(lambda s: int((s > 0).sum())), "weeks": g.size(), "demand": g.sum()}
    )
    stats["avg_wkly"] = stats["demand"] / stats["weeks"].clip(lower=1)
    pid = pd.Series(np.asarray(pids, dtype=object))
    pos = pid.map(stats["pos"]).to_numpy()
    avg = pid.map(stats["avg_wkly"]).to_numpy()
    is_b = (pos >= B_MIN_POS_WEEKS) & (avg >= B_MIN_AVG_WEEKLY)
    is_b2 = (pos >= B_MIN_POS_WEEKS) & (avg >= B2_MIN_AVG_WEEKLY) & ~is_b
    return np.where(is_b, "B-sustained", np.where(is_b2, "B2-mid", "C-long-tail"))


def compute_mase_denominator(prepared_train: pd.DataFrame) -> float:
    """Train 1-step naive MAE - the MASE scaling denominator (unchanged)."""
    ins = prepared_train[["product_id", "week", TARGET]].sort_values(
        ["product_id", "week"], kind="mergesort"
    )
    ins["prev"] = ins.groupby("product_id", observed=True)[TARGET].shift(1)
    ins = ins[ins["prev"].notna()]
    return mae(ins[TARGET], ins["prev"])


def full_metric_suite(actual, pred, cohort: np.ndarray, mase_denom: float) -> dict:
    """The full evaluation suite aligned to the experiment notebooks."""
    ya = np.asarray(actual, dtype=float)
    yp = np.asarray(pred, dtype=float)
    global_mae = mae(ya, yp)
    pos = ya > 0
    b = cohort == "B-sustained"
    b2 = cohort == "B2-mid"
    c = cohort == "C-long-tail"

    def cwape(mask):
        return wape(ya[mask], yp[mask])

    def cbias(mask):
        s = ya[mask].sum()
        return float((yp[mask].sum() / s - 1) * 100) if s > 0 else np.nan

    return {
        "wape": wape(ya, yp),
        "mae": global_mae,
        "rmse": rmse(ya, yp),
        "mase": global_mae / mase_denom,
        "bias_pct": float((yp.sum() / ya.sum() - 1) * 100),
        "asym21": asymmetric_loss(ya, yp),
        "pos_wape": cwape(pos),
        "B_wape": cwape(b),
        "B_bias_pct": cbias(b),
        "B_units": float(ya[b].sum() - yp[b].sum()),
        "B2_wape": cwape(b2),
        "B2_bias_pct": cbias(b2),
        "C_wape": cwape(c),
        "C_bias_pct": cbias(c),
    }