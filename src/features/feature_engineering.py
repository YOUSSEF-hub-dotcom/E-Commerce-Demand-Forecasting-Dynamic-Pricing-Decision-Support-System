"""Past-only feature engineering for the demand forecasting engine.

Every feature produced here is **strictly backward-looking**: a feature at week
``t`` may only depend on information observed strictly before ``t``. This is
enforced by construction:

* **Lag features** - ``groupby(product).shift(k)`` yields the value from k weeks
  earlier.
* **Rolling statistics** - the series is shifted by one week *first* and the
  window is computed over that shifted series, so ``rolling_mean_4w`` at week
  ``t`` covers weeks ``[t-4, t-1]``.
* **Demand drivers** (price/freight/share) - all lagged one week before use.
* **Calendar features** - stateless derivations of the week label.

The module never receives validation/test labels and never fits a statistic on
the full series; the only "fill" operations are either ``0`` (semantically "no
prior demand") or future imputation handled train-only by ``src/data/imputation``.
"""

import numpy as np
import pandas as pd

from src.utils.logging_setup import get_logger

LOGGER = get_logger(__name__)

DEMAND_LAGS = (1, 2, 3, 4)
ROLLING_WINDOWS = (4,)
WEEK_FREQ = "W"
# Phase-3 default feature set per PROJECT_BLUEPRINT Part 2.
PHASE3_LAGS = (1, 2, 4)
PHASE3_ROLLING_WINDOWS = (4,)
PHASE3_ROLLING_AGGS = ("mean", "std")


def _sort_for_shift(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy deterministically sorted per product, week ascending.

    Grouped ``shift`` relies on row order within groups; sorting by the
    temporal key guarantees reproducible lag/rolling semantics.
    """
    return df.sort_values(["product_id", "week"], kind="mergesort").reset_index(drop=True)


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive stateless calendar/temporal features from the week label.

    Parameters
    ----------
    df : pd.DataFrame
        Frame with a weekly ``Period`` column ``week``.

    Returns
    -------
    pd.DataFrame
        Frame extended with ``week_of_year``, ``month``, ``quarter`` and
        ``year``.
    """
    df = df.copy()
    starts = df["week"].dt.start_time
    df["week_of_year"] = starts.dt.isocalendar().week.astype("int64")
    df["month"] = starts.dt.month.astype("int64")
    df["quarter"] = starts.dt.quarter.astype("int64")
    df["year"] = starts.dt.year.astype("int64")
    return df


def add_lag_features(df: pd.DataFrame, lags: tuple[int, ...] = DEMAND_LAGS) -> pd.DataFrame:
    """Add demand lag features at ``t-k`` per product.

    Parameters
    ----------
    df : pd.DataFrame
        Dense demand matrix (must contain ``order_demand``).
    lags : tuple[int, ...], optional
        Lag offsets to generate (default ``(1, 2, 3, 4)``).

    Returns
    -------
    pd.DataFrame
        Frame with ``demand_lag_<k>`` columns. Missing lags (before a product's
        first sale week) are filled with ``0`` - "no prior demand" - which is
        the correct semantics for a zero-mean demand series.
    """
    df = _sort_for_shift(df)
    for k in lags:
        df[f"demand_lag_{k}"] = df.groupby("product_id", observed=True)["order_demand"].shift(k)
    for k in lags:
        df[f"demand_lag_{k}"] = df[f"demand_lag_{k}"].fillna(0).astype("float64")
    LOGGER.info("add_lag_features: lags %s added.", lags)
    return df


def add_rolling_features(
    df: pd.DataFrame,
    windows: tuple[int, ...] = ROLLING_WINDOWS,
    aggs: tuple[str, ...] = ("mean", "max"),
) -> pd.DataFrame:
    """Add rolling-window demand statistics computed strictly on the past.

    The demand series is shifted by one week *before* rolling, so the window at
    week ``t`` covers demand from ``[t-window, t-1]`` - never ``t`` itself.

    Parameters
    ----------
    df : pd.DataFrame
        Frame sorted per product (output of ``add_lag_features`` is fine).
    windows : tuple[int, ...], optional
        Window sizes (default ``(4,)``).
    aggs : tuple[str, ...], optional
        Aggregations to compute per window, any subset of ``mean``, ``max``,
        ``std`` (default ``("mean", "max")``).

    Returns
    -------
    pd.DataFrame
        Frame with ``rolling_demand_<agg>_<w>w`` columns (NaN-free,
        zero-filled for mean/max; std filled with 0 when insuffprising history).
    """
    df = _sort_for_shift(df)
    shifted = df.groupby("product_id", observed=True)["order_demand"].shift(1)
    generated: list[str] = []
    for w in windows:
        roll = shifted.groupby(df["product_id"], observed=True).rolling(w)
        for agg in aggs:
            col = f"rolling_demand_{agg}_{w}w"
            val = roll.agg(agg)
            val = val.reset_index(level=0, drop=True).sort_index().to_numpy()
            df[col] = val.astype("float64")
            # mean/max: "no prior demand" -> 0 ; std: single/first observation
            # window has no spread -> 0.
            df[col] = df[col].fillna(0.0)
            generated.append(col)
    LOGGER.info("add_rolling_features: windows %s aggs %s added (shifted-by-1, past-only): %s",
                windows, aggs, generated)
    return df


def add_demand_drivers(df: pd.DataFrame) -> pd.DataFrame:
    """Add lagged price/freight drivers and the category-demand share.

    All drivers reference strictly prior weeks:

    * ``price_prev``  - average price of the product last week.
    * ``price_chg_1w`` - one-week relative price change (both terms lagged).
    * ``freight_prev`` - average freight value last week.
    * ``category_share_prev`` - product demand as a fraction of its category's
      total demand last week.

    Parameters
    ----------
    df : pd.DataFrame
        Dense demand matrix (needs ``avg_price``, ``avg_freight``,
        ``order_demand`` and ``product_category_name``).

    Returns
    -------
    pd.DataFrame
        Frame extended with the four driver columns. Price/freight lags keep
        their NaNs (resolved by train-only imputation); share NaNs are filled
        with ``0`` ("no prior category demand").
    """
    df = _sort_for_shift(df)
    g_price = df.groupby("product_id", observed=True)["avg_price"]
    g_freight = df.groupby("product_id", observed=True)["avg_freight"]

    df["price_prev"] = g_price.shift(1)
    df["freight_prev"] = g_freight.shift(1)
    df["price_2w_prev"] = df.groupby("product_id", observed=True)["price_prev"].shift(1)

    with np.errstate(divide="ignore", invalid="ignore"):
        df["price_chg_1w"] = (df["price_prev"] - df["price_2w_prev"]) / df["price_2w_prev"]
    df["price_chg_1w"] = df["price_chg_1w"].replace([np.inf, -np.inf], np.nan)

    df["cat_total"] = df.groupby(["product_category_name", "week"], observed=True)["order_demand"].transform("sum")
    product_demand_prev = df.groupby("product_id", observed=True)["order_demand"].shift(1)
    # A product belongs to exactly one category, so shifting the category-week
    # total within the product group yields the previous week's category total.
    cat_total_prev = df.groupby("product_id", observed=True)["cat_total"].shift(1)

    with np.errstate(divide="ignore", invalid="ignore"):
        share = product_demand_prev / cat_total_prev
    df["category_share_prev"] = share.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    df = df.drop(columns=["price_2w_prev", "cat_total"])
    LOGGER.info("add_demand_drivers: price_prev, price_chg_1w, freight_prev, category_share_prev added.")
    return df


def add_target_transform(df: pd.DataFrame) -> pd.DataFrame:
    """Add the ``log1p`` transform of the demand target for skewed-regression support.

    Stateless and safe. Used later by Phase 3 experiments that model
    ``log(1 + order_demand)`` per the blueprint's target-imbalance strategy.

    Parameters
    ----------
    df : pd.DataFrame
        Frame containing ``order_demand``.

    Returns
    -------
    pd.DataFrame
        Frame with ``demand_log1p`` added.
    """
    df = df.copy()
    df["demand_log1p"] = np.log1p(df["order_demand"].astype("float64"))
    return df


def build_feature_matrix(matrix: pd.DataFrame) -> pd.DataFrame:
    """One-call orchestration of the Phase-2 feature stack.

    Parameters
    ----------
    matrix : pd.DataFrame
        Dense demand matrix from ``target_engineering.build_demand_matrix``.

    Returns
    -------
    pd.DataFrame
        Full feature frame: calendar + lags + rolling + demand drivers +
        target transform. Rows sorted per product, week ascending.
    """
    LOGGER.info("build_feature_matrix: starting feature construction (%d rows).", len(matrix))
    df = matrix.copy()
    df = add_calendar_features(df)
    df = add_demand_drivers(df)
    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = add_target_transform(df)
    LOGGER.info("build_feature_matrix: completed -> %d x %d", *df.shape)
    return df


def restrict_to_window(matrix: pd.DataFrame, max_week) -> pd.DataFrame:
    """Truncate a demand matrix to a chronological cutoff (inclusive).

    Used to recompute a fold's feature stack using *only data observed up to*
    the fold's validation boundary - the procedural formulation of the
    "features computed strictly on training-set boundaries" rule.

    Parameters
    ----------
    matrix : pd.DataFrame
        Dense demand matrix (must contain a ``week`` Period column).
    max_week : Period | str
        Last week to keep (inclusive).

    Returns
    -------
    pd.DataFrame
        Copy of ``matrix`` restricted to weeks ``<= max_week``.
    """
    if "week" not in matrix.columns or not isinstance(matrix["week"].dtype, pd.PeriodDtype):
        raise ValueError("restrict_to_window: matrix requires a weekly Period column 'week'.")
    cutoff = pd.Period(max_week, freq=WEEK_FREQ)
    return matrix[matrix["week"] <= cutoff].copy()


def build_phase3_feature_matrix(matrix: pd.DataFrame) -> pd.DataFrame:
    """Phase-3 feature stack: lags {1,2,4} + rolling mean/std(4w) + drivers.

    Implements the refined Phase-3 feature set specified by the blueprint:
    lag features at 1/2/4 weeks, rolling mean and standard deviation over the
    4-week window, calendar/seasonal features, elasticity demand drivers, and
    the ``log1p`` target transform. All features remain strictly past-only.

    Parameters
    ----------
    matrix : pd.DataFrame
        Dense demand matrix from ``target_engineering.build_demand_matrix``.

    Returns
    -------
    pd.DataFrame
        Feature frame for Phase-3 model training.
    """
    LOGGER.info("build_phase3_feature_matrix: starting (%d rows).", len(matrix))
    df = matrix.copy()
    df = add_calendar_features(df)
    df = add_demand_drivers(df)
    df = add_lag_features(df, lags=PHASE3_LAGS)
    df = add_rolling_features(df, windows=PHASE3_ROLLING_WINDOWS, aggs=PHASE3_ROLLING_AGGS)
    df = add_target_transform(df)
    LOGGER.info("build_phase3_feature_matrix: completed -> %d x %d", *df.shape)
    return df