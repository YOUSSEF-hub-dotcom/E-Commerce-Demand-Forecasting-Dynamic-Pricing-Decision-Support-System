"""Conditional imputation that is fit strictly on the training split.

Blueprint Part 2 rule: *statistical imputation is applied strictly within the
training set* to eliminate lookahead bias. This module exposes a single
pipeline step that:

1. **Categorical columns** - learns the category vocabulary on the **train**
   split; values in validation/test that never appeared in training are mapped
   to the reserved label ``unknown`` and missing values are filled with it.
2. **Numeric columns** - ``sklearn.SimpleImputer`` fitted on the **train**
   split and applied to both train and validation/test.

The only legitimate exception to train-only fitting is a *stateless* per-row
fill such as forward-fill of a last-known price, which uses strictly past
information (documented in ``feature_engineering`` and NOT performed here).
"""

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

from src.utils.logging_setup import get_logger

LOGGER = get_logger(__name__)

UNKNOWN_LABEL = "unknown"


def _impute_categorical(frames: dict[str, pd.DataFrame], cols: list[str]) -> dict[str, pd.DataFrame]:
    """Impute categorical columns using the train vocabulary.

    Parameters
    ----------
    frames : dict[str, pd.DataFrame]
        Mapping ``{"train": ..., "val": ..., "test": ...}``.
    cols : list[str]
        Categorical column names to process.

    Returns
    -------
    dict[str, pd.DataFrame]
        Frames with unknown/absent categories relabeled to ``UNKNOWN_LABEL``.
    """
    train_vocab = set()
    for col in cols:
        train_vocab |= set(frames["train"][col].dropna().astype(str).unique())

    for name, frame in frames.items():
        for col in cols:
            frame = frame.copy()
            frame[col] = frame[col].astype("object").fillna(UNKNOWN_LABEL)
            frame.loc[~frame[col].isin(train_vocab), col] = UNKNOWN_LABEL
            frames[name] = frame
    return frames


def _impute_numeric(frames: dict[str, pd.DataFrame], cols: list[str]) -> dict[str, pd.DataFrame]:
    """Median-impute numeric columns (strategy fit on the train split only).

    Parameters
    ----------
    frames : dict[str, pd.DataFrame]
        Mapping ``{"train": ..., "val": ..., "test": ...}``.
    cols : list[str]
        Numeric column names to process.

    Returns
    -------
    dict[str, pd.DataFrame]
        Frames with numeric NaNs replaced by the train-fitted median.
    """
    if not cols:
        return frames
    imputer = SimpleImputer(strategy="median")
    train_values = frames["train"][cols].to_numpy(dtype=float)
    for col in cols:
        if frames["train"][col].isna().all():
            LOGGER.warning("_impute_numeric: column '%s' is fully NaN on train; skipping.", col)
    imputer.fit(train_values)
    for name, frame in frames.items():
        out = imputer.transform(frame[cols].to_numpy(dtype=float))
        frames[name] = frame.copy()
        frames[name][cols] = out
    return frames


def impute_splits(
    frames: dict[str, pd.DataFrame],
    categorical_cols: list[str] | None = None,
    numeric_cols: list[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """Apply the full conditional imputation to train/val/test frames.

    Parameters
    ----------
    frames : dict[str, pd.DataFrame]
        Mapping of split name to DataFrame (keys ``train``/``val``/``test``).
    categorical_cols : list[str] | None, optional
        Categorical columns to impute with the reserved ``unknown`` label.
    numeric_cols : list[str] | None, optional
        Numeric columns to median-impute (statistics fit on train only).

    Returns
    -------
    dict[str, pd.DataFrame]
        Imputed frames, preserving split structure.

    Raises
    ------
    ValueError
        If the input mapping lacks any of train/val/test.
    """
    required = {"train", "val", "test"}
    if not required.issubset(frames.keys()):
        raise ValueError(f"impute_splits: frames must contain {sorted(required)}")

    frames = {k: v.copy() for k, v in frames.items()}

    if categorical_cols:
        frames = _impute_categorical(frames, categorical_cols)
    if numeric_cols:
        frames = _impute_numeric(frames, numeric_cols)

    for name in required:
        LOGGER.info("impute_splits: %s remaining NaNs=%d", name, int(frames[name].isna().sum().sum()))
    return frames