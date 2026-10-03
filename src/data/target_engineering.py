"""Demand target engineering: weekly aggregation and dense demand matrix.

Builds the forecasting target ``order_demand`` (units sold) at the
**product x ISO-week** grain, plus the pricing/freight demand drivers observed
during each week.

* ``build_weekly_demand`` - sparse aggregation: one row per (product, week)
  that actually contains sales.
* ``build_demand_matrix`` - dense reindex: every (product, week) combination
  across the observed timeline, zero-filled for demand. A dense matrix is the
  correct forecasting artifact because a product/week with *no* sales is
  meaningful ``0`` demand, indistinguishable from a missing row in the sparse
  form.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.logging_setup import get_logger

LOGGER = get_logger(__name__)

WEEK_FREQ = "W"  # pandas weekly ending Sunday.


def _to_week(series: pd.Series) -> pd.Series:
    """Convert a datetime64 Series to weekly Period labels (``W`` freq)."""
    return series.dt.to_period(WEEK_FREQ)


def build_weekly_demand(fact: pd.DataFrame) -> pd.DataFrame:
    """Aggregate the fact table into sparse product-week demand rows.

    Parameters
    ----------
    fact : pd.DataFrame
        Valid-order item-level fact table (from ``data_processing``).

    Returns
    -------
    pd.DataFrame
        Columns:
          product_id, week (Period), order_demand, avg_price, avg_freight,
          product_category_name.

    Notes
    -----
    Each item row represents one unit (``order_item_id`` position), so demand
    is the row count per (product, week). ``avg_price`` / ``avg_freight`` are
    the mean price and freight charged that week - the demand drivers consumed
    by the pricing engine in later phases.
    """
    if "order_purchase_timestamp" not in fact.columns:
        raise ValueError("build_weekly_demand: fact table requires order_purchase_timestamp.")

    fact = fact.copy()
    fact["week"] = _to_week(fact["order_purchase_timestamp"])
    if fact["week"].isna().any():
        raise ValueError("build_weekly_demand: unparseable purchase timestamps present.")

    agg = (
        fact.groupby(["product_id", "week"], observed=True)
        .agg(
            order_demand=("order_item_id", "count"),
            avg_price=("price", "mean"),
            avg_freight=("freight_value", "mean"),
        )
        .reset_index()
    )

    category_lookup = fact[["product_id", "product_category_name"]].drop_duplicates("product_id")
    agg = agg.merge(category_lookup, on="product_id", how="left", validate="many_to_one")

    agg = agg.sort_values(["product_id", "week"]).reset_index(drop=True)
    LOGGER.info("build_weekly_demand: %d product-week rows (%.2f of all possible)",
                len(agg), 0.0)
    return agg


def build_demand_matrix(demand: pd.DataFrame) -> pd.DataFrame:
    """Densify the sparse demand table into a full product x week matrix.

    Parameters
    ----------
    demand : pd.DataFrame
        Output of ``build_weekly_demand``.

    Returns
    -------
    pd.DataFrame
        One row per (product, week) for every week between the first and last
        sale. Zero-filled ``order_demand``; ``avg_price``/``avg_freight``/category
        are ``NaN`` for sale-free weeks (handled by conditional imputation).
        ``product_id`` and ``product_category_name`` are stored as pandas
        ``category`` dtype to bound memory.
    """
    required = {"product_id", "week", "order_demand"}
    if not required.issubset(demand.columns):
        raise ValueError(f"build_demand_matrix: missing columns, got {list(demand.columns)}")

    demand = demand.copy()
    demand["product_id"] = demand["product_id"].astype("category")
    if "product_category_name" in demand.columns:
        demand["product_category_name"] = demand["product_category_name"].astype("category")

    products = sorted(demand["product_id"].cat.categories)
    weeks = pd.period_range(demand["week"].min(), demand["week"].max(), freq=WEEK_FREQ)

    idx = pd.MultiIndex.from_product([products, weeks], names=["product_id", "week"])
    matrix = demand.set_index(["product_id", "week"]).reindex(idx).reset_index()
    matrix["order_demand"] = matrix["order_demand"].fillna(0).astype("int64")

    LOGGER.info("build_demand_matrix: %d products x %d weeks = %d rows",
                len(products), len(weeks), len(matrix))
    return matrix


def save_demand_matrix(matrix: pd.DataFrame, out_path: Path | str) -> Path:
    """Persist the demand matrix to ``data/processed`` as compressed CSV.

    Parameters
    ----------
    matrix : pd.DataFrame
        Dense product-week demand matrix.
    out_path : Path | str
        Destination (convention: ``data/processed/demand_matrix_weekly.csv.gz``).

    Returns
    -------
    Path
        Resolved path written.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    matrix.to_csv(out_path, index=False)
    LOGGER.info("save_demand_matrix: wrote %s (%.1f MB)", out_path, out_path.stat().st_size / 1e6)
    return out_path


def load_demand_matrix(path: Path | str) -> pd.DataFrame:
    """Reload a persisted dense demand matrix with dtypes restored.

    Parameters
    ----------
    path : Path | str
        Path produced by ``save_demand_matrix``.

    Returns
    -------
    pd.DataFrame
        Dense matrix with ``week`` recovered as a weekly ``Period`` column,
        ``product_id`` as a categorical, and demand as ``int64``.
    """
    df = pd.read_csv(path)
    starts = df["week"].astype(str).str.split("/").str[0]
    df["week"] = pd.to_datetime(starts).dt.to_period(WEEK_FREQ)
    df["product_id"] = df["product_id"].astype("category")
    if "product_category_name" in df.columns:
        df["product_category_name"] = df["product_category_name"].astype("category")
    df["order_demand"] = df["order_demand"].astype("int64")
    LOGGER.info("load_demand_matrix: loaded %d rows x %d cols.", *df.shape)
    return df