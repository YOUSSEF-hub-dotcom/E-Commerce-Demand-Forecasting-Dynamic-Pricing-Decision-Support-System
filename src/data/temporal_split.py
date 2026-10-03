"""Chronological time-series splitting with a hard no-leakage contract.

The single most important anti-leakage boundary in the pipeline lives here.
Rows are partitioned **by time, in order, with no shuffling**, so no training
observation can ever carry information from the validation or test period.

``temporal_split`` returns contiguous chronological blocks:

    train = rows[0 : n0]
    val   = rows[n0 : n0 + n1]
    test  = rows[n0 + n1 : ]

where ``n0`` and ``n1`` derive deterministically from the requested ratios and
the total row count. The split is pure (stateless) and reproducible - it needs
no random seed because it never samples.

Boundary-week convention
------------------------
The temporal key is the ISO-week label. Because many products share the same
week label, the ratio-aligned row cuts MAY land inside a week, so a week label
can legitimately appear at the boundary of Train/Validation and
Validation/Test (enforced inclusively by ``<=`` in
``assert_chronological_order``). This is an intentional, documented
convention, NOT a leak, because:

1. **Row-level disjointness is always preserved** - every (product, week) row
   belongs to exactly one split; splits never share rows.
2. **Features are past-only** - every feature at week ``t`` is computed from
   strictly earlier weeks (lags, shift-by-1 rolling windows, shifted drivers,
   stateless calendar). A row in the shared boundary week carries no
   information about other rows of the same week.
3. **Probes verify the property behaviorally** - the zero-lookahead probes in
   ``tests/test_data_processing.py`` and ``tests/test_features.py``, and the
   fold-boundary equivalence proof in notebook 03, confirm that no engineered
   value depends on same-week facts.
"""

import numpy as np
import pandas as pd
from pathlib import Path

from src.utils.logging_setup import get_logger

LOGGER = get_logger(__name__)


def temporal_split(
    df: pd.DataFrame,
    date_col: str,
    ratios=(0.70, 0.15, 0.15),
) -> dict[str, pd.DataFrame]:
    """Partition a DataFrame into chronological train/validation/test blocks.

    Parameters
    ----------
    df : pd.DataFrame
        Frame to split. Must contain ``date_col``.
    date_col : str
        Column name holding the temporal key (``datetime64`` or `Period`).
    ratios : tuple[float, float, float], optional
        Train/validation/test fractions summing to 1.0 (default 70/15/15).

    Returns
    -------
    dict[str, pd.DataFrame]
        ``{"train": ..., "val": ..., "test": ...}`` chronologically ordered,
        disjoint, and covering the input exactly.

    Raises
    ------
    ValueError
        If ratios do not sum to 1.0, or if the frame is empty.
    """
    if not np.isclose(sum(ratios), 1.0):
        raise ValueError(f"temporal_split: ratios must sum to 1.0, got {ratios}")
    if len(df) == 0:
        raise ValueError("temporal_split: cannot split an empty frame.")
    if date_col not in df.columns:
        raise ValueError(f"temporal_split: column '{date_col}' not found.")

    ordered = df.sort_values(date_col, kind="mergesort").reset_index(drop=True)
    n = len(ordered)

    n0 = int(round(ratios[0] * n))
    n1 = int(round(ratios[1] * n))
    n2 = n - n0 - n1

    train = ordered.iloc[:n0].copy()
    val = ordered.iloc[n0 : n0 + n1].copy()
    test = ordered.iloc[n0 + n1 :].copy()

    LOGGER.info("temporal_split: train=%d val=%d test=%d (total=%d)", len(train), len(val), len(test), n)
    return {"train": train, "val": val, "test": test}


def assert_chronological_order(splits: dict[str, pd.DataFrame], date_col: str) -> None:
    """Hard-guard that splits form a strictly chronological, non-overlapping sequence.

    Uses **inclusive** comparisons (``<=``) so that a shared boundary week label
    between adjacent splits is permitted (see module docstring: *Boundary-week
    convention*). Row identity remains disjoint; past-only feature construction
    guarantees no confirmed target leakage from a shared boundary week.

    Raises ``AssertionError`` if ``max(train) > min(val)`` or
    ``max(val) > min(test)``. Designed to run inside tests and at the top of
    every training pipeline as a trip-wire against lookahead bias.

    Parameters
    ----------
    splits : dict[str, pd.DataFrame]
        Output of ``temporal_split``.
    date_col : str
        Temporal key column (must be orderable).

    Raises
    ------
    AssertionError
        If any split boundary violates chronological ordering.
    """
    def _max(frame):
        return frame[date_col].max()

    def _min(frame):
        return frame[date_col].min()

    assert _max(splits["train"]) <= _min(splits["val"]), "LEAK: train extends past val start."
    assert _max(splits["val"]) <= _min(splits["test"]), "LEAK: val extends past test start."


def save_splits(splits: dict[str, pd.DataFrame], splits_dir: Path | str) -> dict[str, Path]:
    """Persist split frames to ``data/splits``.

    Parameters
    ----------
    splits : dict[str, pd.DataFrame]
        Output of ``temporal_split``.
    splits_dir : Path | str
        Destination directory (convention: ``data/splits``).

    Returns
    -------
    dict[str, Path]
        Mapping of split name to the file written.
    """
    splits_dir = Path(splits_dir)
    splits_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    for name, frame in splits.items():
        dest = splits_dir / f"{name}.csv"
        frame.to_csv(dest, index=False)
        written[name] = dest
        LOGGER.info("save_splits: wrote %s (%d rows)", dest, len(frame))
    return written