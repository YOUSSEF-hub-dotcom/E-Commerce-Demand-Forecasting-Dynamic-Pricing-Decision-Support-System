"""Part-4 weekly pricing panel: data loading, period boundaries, cost input.

Sources the dense weekly demand matrix (``data/processed/demand_matrix_weekly.csv.gz``)
and derives deterministic TRAIN/VAL/TEST periods from the **week-label** column so that
Part-4 fitting uses only the training period and validation uses the validation period.

**Anti-leakage contract.** None of the Part-4 modules ever read
``data/splits/test.csv``. Period assignment is a pure function of the week label and
the fixed boundary constants below; the test period is never used for fitting,
model selection, or repeated inspection.

**Cost input.** Olist provides no cost field (see ``part4_data_feasibility``). Profit
optimization is therefore gated on an *optional external cost source*; when absent the
engine reports ``PROFIT_NOT_COMPUTABLE`` and never invents a margin.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.utils.logging_setup import get_logger

LOGGER = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MATRIX_PATH = PROJECT_ROOT / "data" / "processed" / "demand_matrix_weekly.csv.gz"

# Week-start labels (ISO Monday) that partition the Part-3 split into Part-4 periods.
TRAIN_MAX_WEEK = "2018-01-22"   # weeks <= this are Part-4 TRAIN
VAL_START_WEEK = "2018-01-29"   # Part-4 VAL period
TEST_START_WEEK = "2018-05-21"  # weeks >= this are Part-4 TEST (never used)
LAST_VAL_WEEK = "2018-05-14"    # default reference week for price recommendations

PRODUCT_COL = "product_id"
WEEK_COL = "week"
DEMAND_COL = "order_demand"
PRICE_COL = "avg_price"
FREIGHT_COL = "avg_freight"
CATEGORY_COL = "product_category_name"
PERIOD_COL = "part4_period"


def week_start_label(week: object) -> str:
    """Normalize a matrix week label to its Monday ISO start ``YYYY-MM-DD``."""
    return str(week).split("/")[0]


def assign_period(week_label: str) -> str:
    """Deterministic period assignment from a week-start label (no test access)."""
    start = week_start_label(week_label)
    if start <= TRAIN_MAX_WEEK:
        return "train"
    if start < TEST_START_WEEK:
        return "val"
    return "test"


def load_matrix(path: str | Path | None = None) -> pd.DataFrame:
    """Load the dense weekly matrix with a normalized week-start label column."""
    src = Path(path) if path is not None else MATRIX_PATH
    df = pd.read_csv(src)
    df[WEEK_COL] = df[WEEK_COL].astype(str).map(week_start_label).astype("category")
    df[PERIOD_COL] = df[WEEK_COL].astype(str).map(assign_period)
    return df


def train_val_only(df: pd.DataFrame) -> pd.DataFrame:
    """Subset to Part-4 train + val periods (test period is excluded entirely)."""
    out = df[df[PERIOD_COL].isin(("train", "val"))].copy()
    if df[PERIOD_COL].eq("test").any():
        LOGGER.info("pricing_data: excluded %d test-period rows from the panel.", int(df[PERIOD_COL].eq("test").sum()))
    return out


def split_periods(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Return ``{"train": ..., "val": ..., "test": ...}`` views of the panel."""
    return {p: df[df[PERIOD_COL] == p].copy() for p in ("train", "val", "test")}


def require_periods(df: pd.DataFrame, allowed: tuple[str, ...] = ("train", "val")) -> None:
    """Assert no fitted/validated frame contains rows outside the allowed periods."""
    present = set(df[PERIOD_COL].unique())
    bad = present - set(allowed)
    if bad:
        raise AssertionError(f"pricing_data: panel contains forbidden period(s): {sorted(bad)}")


def load_sale_prices(df: pd.DataFrame) -> pd.DataFrame:
    """Price observations restricted to sale weeks (price > 0, demand non-null).

    Used for elasticity: only weeks with an observed transaction price can inform
    the demand response.
    """
    out = df[(df[PRICE_COL].notna()) & (df[PRICE_COL] > 0)].copy()
    return out


def load_costs(path: str | Path, product_col: str = "product_id", cost_col: str = "cost") -> dict[str, float]:
    """Load an optional external cost map ``{product_id: cost}``.

    The file must carry ``product_id`` and a strictly positive numeric cost column.
    Returns a plain dict; validates positivity and non-duplicate product keys.
    """
    costs = pd.read_csv(path)
    if product_col not in costs.columns or cost_col not in costs.columns:
        raise ValueError(
            f"pricing_data: cost file must contain '{product_col}' and '{cost_col}' "
            f"columns; got {list(costs.columns)}"
        )
    costs[cost_col] = pd.to_numeric(costs[cost_col], errors="coerce")
    costs = costs.dropna(subset=[cost_col])
    if not (costs[cost_col] > 0).all():
        raise ValueError("pricing_data: cost values must be strictly positive")
    if costs[product_col].duplicated().any():
        raise ValueError("pricing_data: cost file has duplicate product_id entries")
    return {str(k): float(v) for k, v in zip(costs[product_col], costs[cost_col])}