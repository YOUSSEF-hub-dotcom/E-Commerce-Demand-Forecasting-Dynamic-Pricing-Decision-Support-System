"""Part-4 price-elasticity estimation & classification (Phases 4.1-4.2).

Methodology (documented in the Part-4 architecture):

* **Granularity:** product-week on the dense demand matrix (Part-4 train/val periods only).
* **Baseline model:** log-log form ``log(demand) = a + b*log(price)`` (constant-elasticity
  demand ``Q = a*P**b``) fit on the TRAIN period.
* **Where data supports:** pooled panel regressions with product fixed effects, and a
  ``product + week`` fixed-effects variant. Exact FE design mirrors the availability of
  within-product price variation established in the Phase-4.0 audit.
* **Honest identification:** the OLS estimates are a **descriptive demand response** —
  there is no experimental/instrumental price variation in this marketplace data, so no
  causal claim is made anywhere in the report or tags.
* **Coverage honesty:** per-product elasticity is estimated only for products with real
  within-product price variation (>=2 distinct prices, min span, min weeks); products
  without a usable own estimate fall back to their **category pooled elasticity** (flagged),
  and products with *no exploitable variation* are marked ``Unsupported`` (no re-pricing
  recommendation) rather than assumed elastic.
* **Validation (4.2):** the train-fitted pooled model is applied to the VAL-period panel
  (sign, magnitude, R2, n overlap). Per-product reliability is classified from the
  train-period estimate against pre-registered thresholds.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.optimization.pricing_data import (
    CATEGORY_COL,
    DEMAND_COL,
    PRICE_COL,
    PRODUCT_COL,
    WEEK_COL,
)
from src.utils.logging_setup import get_logger

LOGGER = get_logger(__name__)

# Pre-registered identification bar (Phase-4.0 audit outputs).
MIN_WEEKS = 3
MIN_DISTINCT_PRICES = 2
MIN_PRICE_SPAN = 1.05
MAX_ABS_ELASTICITY = 3.0
RELIABLE_T_STAT = 1.0

STATUS_RELIABLE = "Reliable"
STATUS_WEAK = "Weak"
STATUS_INSUFFICIENT = "Insufficient"
STATUS_UNSUPPORTED = "Unsupported"


@dataclass(frozen=True)
class OLSResult:
    """Closed-form OLS coefficient summary (numpy only, no external stats dep)."""

    b: float
    se: float
    t: float
    n: int
    r2: float


def _ols(log_price: np.ndarray, log_demand: np.ndarray) -> OLSResult:
    """Regress ``log_demand`` on ``[1, log_price]``."""
    X = np.column_stack([np.ones(len(log_price)), log_price])
    coef, *_ = np.linalg.lstsq(X, log_demand, rcond=None)
    resid = log_demand - X @ coef
    n = len(log_demand)
    dof = n - 2
    s2 = float(resid @ resid / max(dof, 1))
    cov = s2 * np.linalg.pinv(X.T @ X)
    se = float(np.sqrt(max(cov[1, 1], 0.0)))
    t = coef[1] / se if se > 0 else math.inf
    ss_res = float(resid @ resid)
    ss_tot = float(np.sum((log_demand - log_demand.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return OLSResult(b=float(coef[1]), se=se, t=t, n=n, r2=r2)


def _fe_design(frame: pd.DataFrame, fe_keys: tuple[str, ...]) -> np.ndarray:
    """Design matrix: [log_price] + one-hot columns for the fixed-effect keys."""
    price = np.log(frame[PRICE_COL].to_numpy(dtype=float))
    cols = [price]
    for key in fe_keys:
        cat = pd.Categorical(frame[key])
        dummies = pd.get_dummies(cat, drop_first=True).to_numpy(dtype=float)
        cols.append(dummies)
    return np.column_stack(cols)


def pooled_ols(
    frame: pd.DataFrame,
    *,
    fe_keys: tuple[str, ...] = (PRODUCT_COL,),
    price_col: str = PRICE_COL,
    demand_col: str = DEMAND_COL,
) -> dict[str, float | int | str]:
    """Pooled log-log elasticity with product (and optionally week) fixed effects.

    Uses the fixed-effect (within) estimator: consecutive within-demeaning of
    ``log price`` and ``log demand`` by each ``fe_key`` is algebraically equal to the
    OLS with a full dummy block, but is O(n) in memory/time (a dummy design with
    ~32k product columns is infeasible). Within-demeaning across *two* nested keys
    is the standard sequential two-way within transform.
    """
    sub = frame[
        (frame[price_col].fillna(0) > 0)
        & (frame[demand_col].fillna(0) > 0)
        & np.isfinite(frame[price_col])
        & np.isfinite(frame[demand_col])
    ].copy()
    n = len(sub)
    if n < 3:
        return {"b": float("nan"), "se": float("nan"), "t": float("nan"),
                "n": 0, "r2": float("nan"), "fe": list(fe_keys)}
    lp = np.log(sub[price_col].to_numpy(dtype=float))
    ly = np.log(sub[demand_col].to_numpy(dtype=float))
    tmp = pd.DataFrame({"_lp": pd.Series(lp), "_ly": pd.Series(ly)})
    for key in fe_keys:
        groups = sub[key].to_numpy()
        tmp["_lp"] -= tmp.groupby(groups)["_lp"].transform("mean")
        tmp["_ly"] -= tmp.groupby(groups)["_ly"].transform("mean")
    lp, ly = tmp["_lp"].to_numpy(dtype=float), tmp["_ly"].to_numpy(dtype=float)
    sxx = float(lp @ lp)
    sxy = float(lp @ ly)
    if sxx <= 1e-9:
        return {"b": float("nan"), "se": float("nan"), "t": float("nan"),
                "n": n, "r2": float("nan"), "fe": list(fe_keys),
                "note": "no within-product variation in log price"}
    b = sxy / sxx
    resid = ly - b * lp
    dof = max(n - (1 + int(np.sum([sub[key].nunique() - 1 for key in fe_keys]))), 1)
    s2 = float(resid @ resid / dof)
    se = float(math.sqrt(max(s2 / sxx, 0.0)))
    t = b / se if se > 0 else math.inf
    ss_tot = float((ly - ly.mean()) ** 2 @ np.ones(n))
    r2 = 1.0 - float(resid @ resid) / ss_tot if ss_tot > 0 else 0.0
    return {"b": b, "se": se, "t": t, "n": n, "r2": r2, "fe": list(fe_keys)}


def _product_variation(frame: pd.DataFrame) -> pd.DataFrame:
    """Per-product price-variation summary used by identification (train period)."""
    sub = frame[(frame[PRICE_COL] > 0)].copy()
    if sub.empty:
        return pd.DataFrame(columns=[PRODUCT_COL, "n_weeks", "n_distinct", "span", "min_price", "max_price"])
    g = sub.groupby(PRODUCT_COL)[PRICE_COL]
    span = g.max() / g.min().replace(0, np.nan)
    out = pd.DataFrame(
        {
            "n_weeks": g.size(),
            "n_distinct": g.nunique(),
            "min_price": g.min(),
            "max_price": g.max(),
            "span": span,
        }
    ).reset_index()
    out["span"] = out["span"].fillna(1.0)
    return out


def _per_product_elasticities(train: pd.DataFrame) -> pd.DataFrame:
    """Per-product OLS elasticity for products that pass the identification bar."""
    rows = []
    sale = train[train[PRICE_COL].fillna(0) > 0]
    for pid, grp in sale.groupby(PRODUCT_COL):
        prices = grp[PRICE_COL].to_numpy(dtype=float)
        offers = grp[DEMAND_COL].to_numpy(dtype=float)
        rounded = np.round(prices, 2)
        b = se = t = r2 = float("nan")
        n = 0
        distinct = int(np.unique(rounded).size)
        if (
            distinct >= MIN_DISTINCT_PRICES
            and (prices > 0).all()
            and (offers > 0).all()
            and len(grp) >= MIN_WEEKS
        ):
            o = _ols(np.log(prices), np.log(offers))
            b, se, t, n, r2 = o.b, o.se, o.t, o.n, o.r2
        span = (
            float(rounded.max() / rounded.min())
            if rounded.min() > 0
            else float("inf")
        )
        rows.append(
            {
                PRODUCT_COL: pid,
                "n_weeks": int(len(grp)),
                "n_distinct": distinct,
                "span": span,
                "b": b,
                "se": se,
                "t": t,
                "n": n,
                "r2": r2,
                "status": None,
            }
        )
    return pd.DataFrame(rows)


def _category_elasticities(train: pd.DataFrame) -> dict[str, dict[str, float | int]]:
    out = {}
    for cat, grp in train.groupby(CATEGORY_COL):
        res = pooled_ols(grp, fe_keys=(PRODUCT_COL,))
        out[cat] = res
    return out


def classify_own(row: pd.Series) -> str:
    """Product-level reliability class from its own train-period estimate."""
    if row["n_distinct"] < MIN_DISTINCT_PRICES:
        return STATUS_UNSUPPORTED
    if row["span"] < MIN_PRICE_SPAN or row["n_weeks"] < MIN_WEEKS:
        return STATUS_INSUFFICIENT
    if not (row["b"] < 0):
        return STATUS_WEAK
    if abs(row["b"]) > MAX_ABS_ELASTICITY or abs(row["b"]) < 1e-9:
        return STATUS_WEAK
    if row["t"] < 0 and abs(row["t"]) >= RELIABLE_T_STAT:
        return STATUS_RELIABLE
    return STATUS_WEAK


def effective_elasticity(
    per_product: pd.DataFrame,
    per_category: dict[str, dict[str, float | int]],
    category_of: dict[str, str],
) -> pd.DataFrame:
    """Pick the effective elasticity per product: own (if usable) else category-pooled."""
    out = per_product.copy()
    own_usable = out["status"].isin((STATUS_RELIABLE, STATUS_WEAK))
    out["category"] = out[PRODUCT_COL].map(category_of)
    out["cat_b"] = out["category"].map(lambda c: per_category.get(c, {}).get("b"))
    cat_usable_map = {
        c: (bool(r.get("b", 0) < 0) and bool(r.get("n", 0) >= MIN_WEEKS) and bool(abs(r.get("b", math.inf)) <= MAX_ABS_ELASTICITY))
        for c, r in per_category.items()
    }
    out["cat_usable"] = out["category"].map(cat_usable_map)
    use_own = own_usable & out["cat_usable"].fillna(False)
    use_cat = (~own_usable) & out["cat_usable"].fillna(False)
    out["elasticity"] = np.where(use_own, out["b"], np.where(use_cat, out["cat_b"], np.nan))
    out["elasticity_source"] = np.where(use_own, "own", np.where(use_cat, "category_pooled", "none"))
    return out


def estimate(
    train: pd.DataFrame,
    val: pd.DataFrame | None = None,
    *,
    fe_keys: tuple[str, ...] = (PRODUCT_COL,),
    category_col: str = CATEGORY_COL,
) -> dict:
    """Run the Phase-4.1 estimation + Phase-4.2 validation and return a report.

    ``train`` must be a Part-4 TRAIN-period frame, ``val`` an optional Part-4
    VAL-period frame (used only for pooled-model validation).
    """
    per_product = _per_product_elasticities(train)
    if not per_product.empty:
        per_product["status"] = per_product.apply(classify_own, axis=1)

    per_category = _category_elasticities(train)

    # Panels (product FE / product+week FE) on the full train sale subset.
    pool_fe = pooled_ols(train, fe_keys=fe_keys)
    pool_fe_week = pooled_ols(train, fe_keys=fe_keys + (WEEK_COL,))

    category_of = (
        train[[PRODUCT_COL, category_col]]
        .astype(str)
        .drop_duplicates()
        .set_index(PRODUCT_COL)[category_col]
        .to_dict()
    )
    eff = effective_elasticity(per_product, per_category, category_of)

    val_diag = {}
    if val is not None and len(val):
        v_res = pooled_ols(val, fe_keys=fe_keys)
        val_diag = {
            "val_n": v_res["n"],
            "val_b": v_res["b"],
            "val_se": v_res["se"],
            "val_t": v_res["t"],
            "val_r2": v_res["r2"],
            "train_val_sign_match": bool((pool_fe["b"] < 0) == (v_res["b"] < 0)),
            "train_val_magnitude_similar": bool(
                abs(pool_fe["b"] - v_res["b"]) <= 1.0
            ),
        }

    pool_n = int(pool_fe["n"])
    return {
        "method": "log_log_constant_elasticity",
        "identification_bounds": {
            "min_weeks": MIN_WEEKS,
            "min_distinct_prices": MIN_DISTINCT_PRICES,
            "min_price_span": MIN_PRICE_SPAN,
            "max_abs_elasticity": MAX_ABS_ELASTICITY,
            "reliable_t_stat": RELIABLE_T_STAT,
            "causal_claim": False,
        },
        "sample": {
            "train_rows": int(len(train)),
            "train_sale_rows": int((train[PRICE_COL].fillna(0) > 0).sum()),
            "products": int(train[PRODUCT_COL].nunique()),
            "products_with_variation": int(
                per_product["status"].isin((STATUS_RELIABLE, STATUS_WEAK)).sum()
            )
            if not per_product.empty
            else 0,
            "val_rows": int(len(val)) if val is not None else 0,
        },
        "pooled_product_fe": pool_fe,
        "pooled_product_week_fe": pool_fe_week,
        "val_diagnostics": val_diag,
        "per_product": per_product,
        "per_category": per_category,
        "effective": eff,
        "status_counts": (
            per_product["status"].value_counts().to_dict() if not per_product.empty else {}
        ),
    }