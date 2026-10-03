"""Model-matrix preparation: encoding, target variants, and sample weights.

Converts an imputed Phase-3 feature frame into a model-ready matrix shared by
every candidate algorithm (LightGBM / XGBoost):

* **Drops** row-identity and current-week monetary columns (``product_id``,
  ``week``, ``avg_price``, ``avg_freight``) so the model sees only past-only
  features - and so the artefact generalizes to unseen products (cold start via
  category + drivers, never via identity lookup).
* **One-hot encodes** the category (71-73 levels) - a single code path works for
  both LightGBM and XGBoost.
* **Target variants** - ``order_demand`` (business scale) and ``demand_log1p``
  (model scale, skew-normalized per the blueprint).
* **Sample weights** - ``1 + log(1 + demand)``: sub-linear emphasis on high
  volume weeks without letting the long tail dominate, implemented under the
  log-domain objective.
"""

import numpy as np
import pandas as pd

from src.utils.logging_setup import get_logger

LOGGER = get_logger(__name__)

# Chronological + identity columns that must never enter the design matrix.
ROW_IDENTITY_COLS = ["product_id", "week", "product_category_name"]
# Current-week monetary columns excluded from the past-only model matrix.
CURRENT_WEEK_COLS = ["avg_price", "avg_freight"]
# Past-only driver + calendar + lag columns consumed by the models.
FEATURE_COLS = [
    "week_of_year", "month", "quarter", "year",
    "demand_lag_1", "demand_lag_2", "demand_lag_4",
    "rolling_demand_mean_4w", "rolling_demand_std_4w",
    "price_prev", "price_chg_1w", "freight_prev", "category_share_prev",
]
TARGET_COL = "order_demand"
LOG_TARGET_COL = "demand_log1p"
WEIGHT_COL = "sample_weight"
CATEGORY_COL = "product_category_name"


def make_sample_weights(demand: pd.Series) -> np.ndarray:
    """Compute per-row sample weights from observed demand.

    ``w = 1 + log(1 + demand)`` grows sub-linearly with demand so busy weeks
    are emphasized without letting extreme-value weeks dominate the objective.

    Parameters
    ----------
    demand : pd.Series
        Observed ``order_demand`` values.

    Returns
    -------
    np.ndarray
        Non-negative weight per row.
    """
    weights = 1.0 + np.log1p(np.asarray(demand, dtype=float))
    return weights.astype(np.float64)


def prepare_model_matrix(frame: pd.DataFrame, categories: list[str] | None = None) -> pd.DataFrame:
    """Turn an imputed feature frame into the shared model design matrix.

    Parameters
    ----------
    frame : pd.DataFrame
        Imputed Phase-3 feature frame (train/val/test), containing the
        category column, target columns, and all ``FEATURE_COLS``.
    categories : list[str] | None, optional
        Frozen category levels `<prefix><label>` learned from the **train split
        only** and applied identically to every split. When provided, each
        split's one-hot block shares the identical column set - otherwise a
        category seen only in training would silently disappear from the
        validation matrix and break the design schema. The frozen pipeline
        (`final_forecast.prepare_frozen`) derives this list from train alone.

    Returns
    -------
    pd.DataFrame
        Model-ready matrix with ``FEATURE_COLS`` + one-hot category columns,
        plus ``TARGET_COL``, ``LOG_TARGET_COL``, and ``WEIGHT_COL``.

    Raises
    ------
    ValueError
        If any required column is missing or NaN values remain in the design
        columns.
    """
    required = set(FEATURE_COLS) | {CATEGORY_COL, TARGET_COL}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"prepare_model_matrix: missing columns {sorted(missing)}")

    df = frame.reset_index(drop=True)

    dummies = pd.get_dummies(df[CATEGORY_COL], prefix="cat", dtype=np.float32)
    if categories is not None:
        # Structural no-expansion guarantee: when a frozen schema is supplied,
        # the encoded block must never gain columns beyond it (which would
        # change the model's input dimensionality downstream). Unseen labels
        # should already have been mapped to the reserved ``unknown`` label by
        # imputation, so any extra column here is a pipeline contract breach.
        frozen_vocab = {f"cat_{c}" for c in sorted(categories)}
        extra = set(dummies.columns) - frozen_vocab
        if extra:
            raise RuntimeError(
                f"prepare_model_matrix: encoded block escapes the frozen schema "
                f"({sorted(extra)}). Ensure unseen labels route to 'unknown' before OHE."
            )
        dummies = dummies.reindex(columns=sorted(frozen_vocab), fill_value=np.float32(0.0))
    dummies = dummies.loc[:, sorted(dummies.columns)]

    design_cols = list(FEATURE_COLS) + list(dummies.columns)
    X = pd.concat([df[FEATURE_COLS].reset_index(drop=True), dummies], axis=1)

    if X.isna().any().any():
        bad = X.columns[X.isna().any()].tolist()
        raise ValueError(f"prepare_model_matrix: NaN in design columns {bad}")

    out = df[["product_id", "week", TARGET_COL, LOG_TARGET_COL]].reset_index(drop=True)
    out = pd.concat([out, X], axis=1)
    out[WEIGHT_COL] = make_sample_weights(out[TARGET_COL])
    LOGGER.info(
        "prepare_model_matrix: %d rows x %d design cols (incl. %d one-hot categories).",
        len(out), X.shape[1], len(dummies.columns),
    )
    return out


def design_matrix_rows(prepared: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Split a prepared frame into ``(X, y_log, weights)`` for model fitting.

    Parameters
    ----------
    prepared : pd.DataFrame
        Output of ``prepare_model_matrix``.

    Returns
    -------
    tuple[pd.DataFrame, pd.Series, pd.Series]
        Design matrix ``X``, log-target ``y``, and sample weights.
    """
    y = prepared[LOG_TARGET_COL].astype("float64")
    weights = prepared[WEIGHT_COL].astype("float64")
    X = prepared.drop(columns=[TARGET_COL, LOG_TARGET_COL, WEIGHT_COL, "product_id", "week"])
    return X, y, weights