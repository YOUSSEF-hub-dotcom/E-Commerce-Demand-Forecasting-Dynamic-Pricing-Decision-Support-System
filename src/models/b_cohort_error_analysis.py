"""Part-3 B-cohort error analysis (validation-only root-cause investigation).

DIAGNOSIS ONLY. No model modification, no retraining, no tuning, no Test access.

The frozen V2 model is loaded from the Model Registry (read-only) and scored
ONLY on the Validation split. The validation predictions are justified by
reproducing the official V2 validation metrics (global WAPE ~0.3964,
B WAPE ~0.5101) from the v2 history.

The Test split is intentionally never read by this module: all investigation
uses Train (cohort / history / volatility statistics) and Validation (the
development/evaluation set). See the Test Protection Statement embedded in the
emitted report.

Output (new file only):
    reports/forecasting/part3_b_cohort_error_analysis.json

Usage:
    python -m src.models.b_cohort_error_analysis
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.models.data_prep import design_matrix_rows
from src.models.final_forecast import (
    TARGET,
    cohort_flags,
    compute_mase_denominator,
    freeze_schema,
    full_metric_suite,
)
from src.models.finalize_lifecycle import build_prepared
from src.models.registry import MODEL_NAME, init_mlflow
from src.utils.logging_setup import get_logger

LOGGER = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = PROJECT_ROOT / "reports" / "forecasting"
REPORT_JSON = REPORTS_DIR / "part3_b_cohort_error_analysis.json"

MODEL_VERSION = 2
MODEL_URI = "models:/{}/{}".format(MODEL_NAME, MODEL_VERSION)

# Official V2 validation references (part3_v2_audit.json / v2 history).
VAL_GLOBAL_WAPE_REF = 0.3964
VAL_B_WAPE_REF = 0.5101460932701322
WAPE_REPRO_TOLERANCE = 2e-3

COHORT_ORDER = ["B-sustained", "B2-mid", "C-long-tail"]


def _machine(v):
    if isinstance(v, (np.floating, np.integer, np.bool_)):
        return v.item()
    if isinstance(v, pd.Series):
        return v.tolist()
    if isinstance(v, (list, tuple)):
        return [_machine(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _machine(x) for k, x in v.items()}
    return v


def _desc(series) -> dict:
    s = pd.Series(pd.to_numeric(series), dtype="float64").dropna()
    if s.empty:
        return {}
    return {
        "count": int(s.count()),
        "mean": _machine(s.mean()),
        "std": _machine(s.std()),
        "min": _machine(s.min()),
        "p25": _machine(s.quantile(0.25)),
        "median": _machine(s.median()),
        "p75": _machine(s.quantile(0.75)),
        "p95": _machine(s.quantile(0.95)),
        "p99": _machine(s.quantile(0.99)),
        "max": _machine(s.max()),
    }


def _wape_sub(sub: pd.DataFrame) -> dict:
    sa = float(sub["actual"].sum())
    se = float(sub["abs_error"].sum())
    return {
        "rows": int(len(sub)),
        "sum_actual": _machine(sa),
        "sum_abs_error": _machine(se),
        "wape": _machine(se / sa) if sa > 0 else None,
        "bias_pct": _machine((float(sub["error"].sum()) / sa) * 100) if sa > 0 else None,
    }


def _bucketed_wape(df: pd.DataFrame, key: str, edges) -> list[dict]:
    bucket = pd.cut(df[key], bins=edges, include_lowest=True)
    out = []
    for interval, g in df.groupby(bucket, observed=True):
        if g.empty:
            continue
        row = _wape_sub(g)
        row["bucket"] = str(interval)
        out.append(row)
    return out


def _cohort_profile(df: pd.DataFrame, train_stats: pd.DataFrame, dev_rows: int) -> dict:
    sa = float(df["actual"].sum())
    se = float(df["abs_error"].sum())
    hist = train_stats.reindex(df["product_id"].unique())
    return {
        "rows": int(len(df)),
        "products": int(df["product_id"].nunique()),
        "week_labels": int(df["week"].nunique()),
        "start_week": _machine(df["week"].min()),
        "end_week": _machine(df["week"].max()),
        "share_of_dev_rows": _machine(len(df) / dev_rows),
        "zero_demand_ratio": _machine(float((df["actual"] == 0).mean())),
        "mean_actual": _machine(df["actual"].mean()),
        "median_actual": _machine(df["actual"].median()),
        "sum_actual": _machine(sa),
        "sum_abs_error": _machine(se),
        "wape": _machine(se / sa) if sa > 0 else None,
        "bias_pct": _machine((float(df["error"].sum()) / sa) * 100) if sa > 0 else None,
        "signed_error_sum": _machine(float(df["error"].sum())),
        "cv_actual": _machine(float(df["actual"].std(ddof=0) / df["actual"].mean())) if sa > 0 else None,
        "mean_train_history_rows": _machine(hist["train_rows"].mean()) if len(hist) else None,
        "mean_train_nonzero_weeks": _machine(hist["train_nonzero"].mean()) if len(hist) else None,
        "mean_train_avg_demand": _machine(hist["train_avg"].mean()) if len(hist) else None,
        "mean_train_cv": _machine(hist["train_cv"].mean()) if len(hist) else None,
    }


def analyze() -> dict:
    LOGGER.info("b_cohort: loading cached prepared splits (train + val only).")
    prepared = build_prepared(schema_len_expected=93)
    train, val = prepared["train"], prepared["val"]
    schema = freeze_schema(train)
    assert len(schema) == 93, "frozen schema must be 93 columns"
    dev_rows = int(len(train) + len(val))

    val_X, _, _ = design_matrix_rows(val)
    val_X = val_X.astype("float64")
    assert list(val_X.columns) == schema, "val design matrix != frozen schema order"

    LOGGER.info("b_cohort: loading frozen V2 (%s) and scoring VALIDATION only.", MODEL_URI)
    init_mlflow()
    from mlflow import pyfunc

    loaded = pyfunc.load_model(MODEL_URI)
    pred = loaded.predict(val_X)["pred_demand"].to_numpy(dtype="float64")
    actual = val[TARGET].astype("float64").to_numpy()
    assert len(pred) == len(actual)

    val_pids = val["product_id"].astype(str).to_numpy()
    cohort = cohort_flags(train, val_pids)
    pid_to_cohort = dict(zip(val_pids, cohort))
    mase_denom = compute_mase_denominator(train)

    global_metrics = full_metric_suite(actual, pred, cohort, mase_denom)
    reproducibility = {
        "global_wape_recomputed": _machine(global_metrics["wape"]),
        "global_wape_reference": VAL_GLOBAL_WAPE_REF,
        "B_wape_recomputed": _machine(global_metrics["B_wape"]),
        "B_wape_reference": VAL_B_WAPE_REF,
        "global_wape_matches": bool(abs(global_metrics["wape"] - VAL_GLOBAL_WAPE_REF) <= WAPE_REPRO_TOLERANCE),
        "B_wape_matches": bool(abs(global_metrics["B_wape"] - VAL_B_WAPE_REF) <= WAPE_REPRO_TOLERANCE),
    }
    LOGGER.info(
        "b_cohort: sanity reproduction global wape=%.4f (ref %.4f) | B wape=%.4f (ref %.4f).",
        global_metrics["wape"], VAL_GLOBAL_WAPE_REF,
        global_metrics["B_wape"], VAL_B_WAPE_REF,
    )

    # Error-analysis DataFrame over the full validation set.
    error = pred - actual
    cat_cols = [c for c in val.columns if c.startswith("cat_")]
    ohe = val[cat_cols].to_numpy(dtype="float64")
    labels = np.array([c[len("cat_"):] for c in cat_cols])
    cats = labels[ohe.argmax(axis=1)]

    df = pd.DataFrame(
        {
            "product_id": val_pids,
            "week": val["week"].astype(str).to_numpy(),
            "category": cats,
            "actual": actual,
            "prediction": pred,
            "error": error,
            "abs_error": np.abs(error),
        }
    )
    for c in schema:
        df[c] = val[c].astype("float64").to_numpy()
    df["relative_error"] = np.where(df["actual"] > 0, df["error"] / df["actual"], np.nan)
    df["cohort"] = df["product_id"].map(pid_to_cohort)

    # Train-side per-product history / volatility statistics (Train only).
    tr = train[["product_id", TARGET]].copy()
    tr["product_id"] = tr["product_id"].astype(str)
    grp = tr.groupby("product_id")[TARGET]
    train_stats = pd.DataFrame(
        {
            "train_rows": grp.size(),
            "train_nonzero": grp.apply(lambda s: int((s > 0).sum())),
            "train_sum": grp.sum(),
        }
    )
    train_stats["train_avg"] = train_stats["train_sum"] / train_stats["train_rows"].clip(lower=1)
    train_stats["train_cv"] = grp.std() / train_stats["train_avg"].clip(lower=1e-9)

    profile = {}
    for cname in COHORT_ORDER:
        profile[cname] = _cohort_profile(df[df["cohort"] == cname], train_stats, dev_rows)
    profile["overall"] = _cohort_profile(df, train_stats, dev_rows)

    # ---- B-specific analyses -------------------------------------------------
    b = df[df["cohort"] == "B-sustained"].copy()
    if len(b) == 0:
        raise RuntimeError("B cohort is empty on validation - unexpected.")
    out: dict = {"profile": profile, "reproducibility": reproducibility}
    saB = float(b["actual"].sum())
    seB = float(b["abs_error"].sum())

    # A. product-level
    prod = (
        b.groupby("product_id")
        .agg(
            rows=("actual", "size"),
            sum_actual=("actual", "sum"),
            sum_abs_error=("abs_error", "sum"),
            sum_error=("error", "sum"),
            nonzero_weeks_in_val=("actual", lambda s: int((s > 0).sum())),
        )
        .reset_index()
    )
    prod["wape_p"] = prod["sum_abs_error"] / prod["sum_actual"].clip(lower=1e-12)
    prod["bias_pct_p"] = (prod["sum_error"] / prod["sum_actual"].clip(lower=1e-12)) * 100
    prod["pct_of_B_error"] = prod["sum_abs_error"] / seB * 100
    prod["pct_of_B_actual"] = prod["sum_actual"] / saB * 100
    prod = prod.merge(
        train_stats[["train_rows", "train_nonzero", "train_avg", "train_cv"]],
        left_on="product_id", right_index=True, how="left",
    )
    prod = prod.replace({np.nan: None})

    out["B_top_abs_error_products"] = _machine(
        prod.sort_values("sum_abs_error", ascending=False).head(20).to_dict("records")
    )
    out["B_top_wape_contributors"] = _machine(
        prod.sort_values("pct_of_B_error", ascending=False).head(20).to_dict("records")
    )
    under = prod[prod["sum_error"] < 0].sort_values("sum_error")
    over = prod[prod["sum_error"] > 0].sort_values("sum_error", ascending=False)
    out["B_strongest_underprediction"] = _machine(under.head(10).to_dict("records"))
    out["B_strongest_overprediction"] = _machine(over.head(10).to_dict("records"))
    out["B_error_direction_summary"] = {
        "products_under": int(len(under)),
        "products_over": int(len(over)),
        "share_under_of_products": _machine(len(under) / len(prod)),
        "share_over_of_products": _machine(len(over) / len(prod)),
    }

    # Concentration diagnostics
    order = prod.sort_values("pct_of_B_error", ascending=False)
    cum_err = order["pct_of_B_error"].cumsum()
    cum_act = order["pct_of_B_actual"].cumsum()
    n_products = int(len(order))

    def top_share(pct_err: float) -> dict:
        hit = (cum_err >= pct_err)
        n_hit = int(hit.idxmax() + 1) if hit.any() else n_products
        return {
            "target_pct": pct_err,
            "products_needed": n_hit,
            "share_of_products": _machine(n_hit / n_products),
            "cum_error_share_at_hit": _machine(cum_err.iloc[n_hit - 1]),
            "cum_actual_share_at_hit": _machine(cum_act.iloc[n_hit - 1]),
        }

    out["B_concentration"] = {
        "products": n_products,
        "top1_error_share_pct": _machine(float(cum_err.iloc[0])),
        "top5_error_share_pct": _machine(float(cum_err.iloc[min(4, n_products - 1)])),
        "top10_error_share_pct": _machine(float(cum_err.iloc[min(9, n_products - 1)])),
        "top20_error_share_pct": _machine(float(cum_err.iloc[min(19, n_products - 1)])),
        "top1_actual_share_pct": _machine(float(cum_act.iloc[0])),
        "top5_actual_share_pct": _machine(float(cum_act.iloc[min(4, n_products - 1)])),
        "top10_actual_share_pct": _machine(float(cum_act.iloc[min(9, n_products - 1)])),
        "top20_actual_share_pct": _machine(float(cum_act.iloc[min(19, n_products - 1)])),
        "products_for_top_50pct_error": top_share(50.0),
        "products_for_top_80pct_error": top_share(80.0),
        "per_product_wape_distribution": _desc(prod["wape_p"]),
    }

    worst_id = order.iloc[0]["product_id"]
    mask = b["product_id"] != worst_id
    out["B_leave_one_out_worst_product"] = {
        "removed_product": worst_id,
        "B_wape_without": _machine(float(b[mask]["abs_error"].sum()) / float(b[mask]["actual"].sum())),
        "B_wape_full": _machine(seB / saB),
    }

    # B. demand-pattern analysis
    hist_max = b[["rolling_demand_mean_4w", "demand_lag_1"]].max(axis=1)
    b["spike"] = (b["actual"] > 0) & (b["actual"] > 2 * hist_max + 1e-9)
    b["collapse"] = (b["actual"] == 0) & (b["demand_lag_1"] > 0)
    b["zero_week"] = b["actual"] == 0

    full_metrics_b = {
        "rows": int(len(b)),
        "sum_actual": _machine(saB),
        "sum_abs_error": _machine(seB),
        "wape": _machine(seB / saB),
    }
    out["B_demand_pattern"] = {
        "full_B": full_metrics_b,
        "zero_weeks": _wape_sub(b[b["zero_week"]]),
        "spike_weeks": _wape_sub(b[b["spike"]]),
        "non_spike_nonzero_weeks": _wape_sub(b[(~b["spike"]) & (~b["zero_week"])]),
        "collapse_weeks": _wape_sub(b[b["collapse"]]),
        "n_zero_rows_in_val": int((b["actual"] == 0).sum()),
        "n_spike_rows_in_val": int(b["spike"].sum()),
        "n_products_with_zero_within_val": int(b[b["zero_week"]]["product_id"].nunique()),
        "per_product_val_cv_distribution": _desc(
            b.groupby("product_id")["actual"].std(ddof=0) / b.groupby("product_id")["actual"].mean().clip(lower=1e-12)
        ) if saB > 0 else None,
        "actual_distribution": _desc(b["actual"]),
    }

    b["_cv"] = b["product_id"].map(train_stats["train_cv"]).fillna(0.0)
    out["B_wape_by_train_cv_bucket"] = _bucketed_wape(b, "_cv", [0, 0.5, 1.0, 2.0, float("inf")])

    # C. error-direction analysis
    under_r = b[b["error"] < 0]
    over_r = b[b["error"] > 0]
    out["B_error_direction_rows"] = {
        "underprediction_rows": int(len(under_r)),
        "overprediction_rows": int(len(over_r)),
        "exact_rows": int(len(b) - len(under_r) - len(over_r)),
        "underprediction_share_of_B_error": _machine(float(under_r["abs_error"].sum()) / seB),
        "overprediction_share_of_B_error": _machine(float(over_r["abs_error"].sum()) / seB),
        "mean_error": _machine(float(b["error"].mean())),
        "median_error": _machine(float(b["error"].median())),
        "error_distribution": _desc(b["error"]),
        "abs_error_distribution": _desc(b["abs_error"]),
        "rel_error_distribution_positive_rows": _desc(b.loc[b["actual"] > 0, "relative_error"]),
        "underprediction": _wape_sub(under_r),
        "overprediction": _wape_sub(over_r),
    }

    # D. temporal analysis
    weekly = (
        b.groupby("week")
        .agg(sum_actual=("actual", "sum"), sum_abs_error=("abs_error", "sum"), sum_error=("error", "sum"))
        .reset_index()
    )
    weekly["wape"] = weekly["sum_abs_error"] / weekly["sum_actual"].clip(lower=1e-12)
    weekly["bias_pct"] = weekly["sum_error"] / weekly["sum_actual"].clip(lower=1e-12) * 100
    weeks_sorted = sorted(weekly["week"].unique().tolist())
    half = len(weeks_sorted) // 2
    early = set(weeks_sorted[:half])
    recent = set(weeks_sorted[half:])
    out["B_temporal"] = {
        "week_label_count": len(weeks_sorted),
        "weekly_metrics": _machine(weekly.replace({np.nan: None}).to_dict("records")),
        "worst_weeks": _machine(weekly.sort_values("wape", ascending=False).head(8).replace({np.nan: None}).to_dict("records")),
        "early_half": _wape_sub(b[b["week"].isin(early)]),
        "recent_half": _wape_sub(b[b["week"].isin(recent)]),
    }

    # E. feature-pattern analysis (descriptive buckets)
    out["B_wape_by_abs_price_chg_bucket"] = _bucketed_wape(
        b.assign(_px=b["price_chg_1w"].abs()), "_px", [float("-inf"), 0.0, 0.05, 0.15, float("inf")]
    )
    out["B_wape_by_positive_rate_bucket"] = _bucketed_wape(
        b.assign(_pr=b["positive_rate_8w"]), "_pr", [0.0, 0.25, 0.75, 1.0001]
    )
    out["B_lag1_zero_vs_nonzero"] = {
        "lag1_zero": _wape_sub(b[b["demand_lag_1"] == 0]),
        "lag1_nonzero": _wape_sub(b[b["demand_lag_1"] > 0]),
    }
    out["B_lag4_zero_vs_nonzero"] = {
        "lag4_zero": _wape_sub(b[b["demand_lag_4"] == 0]),
        "lag4_nonzero": _wape_sub(b[b["demand_lag_4"] > 0]),
    }
    out["B_wape_by_rolling_std_bucket"] = _bucketed_wape(
        b.assign(_rs=b["rolling_demand_std_4w"]), "_rs", [float("-inf"), 0.0, 0.5, 2.0, float("inf")]
    )
    level = b.assign(_lv=b["rolling_demand_mean_4w"])
    out["B_wape_by_rolling_mean_level_bucket"] = _bucketed_wape(
        level, "_lv", [float("-inf"), 0.0, 1.0, 5.0, float("inf")]
    )

    # F. category analysis within B
    cat = (
        b.groupby("category")
        .agg(rows=("actual", "size"), sum_actual=("actual", "sum"), sum_abs_error=("abs_error", "sum"))
        .reset_index()
    )
    cat["wape"] = cat["sum_abs_error"] / cat["sum_actual"].clip(lower=1e-12)
    cat["pct_of_B_error"] = cat["sum_abs_error"] / seB * 100
    cat["pct_of_B_actual"] = cat["sum_actual"] / saB * 100
    cat = cat.replace({np.nan: None}).sort_values("pct_of_B_error", ascending=False)
    out["B_by_category"] = _machine(cat.to_dict("records"))
    out["B_category_n"] = int(len(cat))
    high = cat[cat["wape"] > 0.6]
    out["B_category_high_error"] = {
        "n_categories_high": int(len(high)),
        "rows_in_high_cats": int(high["rows"].sum()),
        "actual_share_in_high_cats_pct": _machine(float(high["sum_actual"].sum()) / saB * 100),
        "error_share_in_high_cats_pct": _machine(float(high["sum_abs_error"].sum()) / seB * 100),
    }

    # G. product-history (bucket by train history rows)
    b["_hist"] = b["product_id"].map(train_stats["train_rows"]).fillna(0.0)
    hist_edges = [0, 30, 60, 90, float("inf")]
    out["B_wape_by_train_history_bucket"] = _bucketed_wape(b, "_hist", hist_edges)
    b["_nz"] = b["product_id"].map(train_stats["train_nonzero"]).fillna(0.0)
    out["B_wape_by_train_nonzero_bucket"] = _bucketed_wape(b, "_nz", [0, 5, 10, 20, float("inf")])

    out["test_protection_statement"] = TEST_PROTECTION_STATEMENT
    out["model"] = {"uri": MODEL_URI, "version": MODEL_VERSION}
    out["dev_rows"] = dev_rows
    out["feature_columns_used"] = list(schema)
    out["note"] = (
        "Diagnosis-only analysis on TRAIN + VALIDATION. The Test split is not "
        "read; the historical Test B WAPE=0.628239 (post-hoc evaluation) is "
        "preserved unchanged."
    )
    return _machine(out)


TEST_PROTECTION_STATEMENT = (
    "No new experimentation, tuning, feature selection, or model decision was "
    "based on the Test split. The Test split (data/splits/test.csv) is never "
    "read by this analysis; the B WAPE=0.628239 result from the post-hoc "
    "evaluation is preserved as fixed historical evidence, and Validation is "
    "used as the development/selection evaluation set."
)


def main() -> dict:
    result = analyze()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    LOGGER.info("b_cohort: analysis JSON written to %s.", REPORT_JSON)
    return result


if __name__ == "__main__":
    main()