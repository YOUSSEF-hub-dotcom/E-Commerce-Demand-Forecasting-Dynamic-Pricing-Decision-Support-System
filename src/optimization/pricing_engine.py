"""Part-4 pricing engine orchestrator (Phases 4.0-4.6 end-to-end driver).

Pipeline (architecture separation enforced):

    weekly panel (Part-4 train/val only)
        -> elasticity estimation + validation/classification   (``elasticity``)
        -> demand baseline from the Part-3 Production forecast (``demand_baseline``)
        -> price-simulation grid                              (``simulation``)
        -> profit/revenue optimizer P* = argmax               (``optimizer``)
        -> business guardrails                                (``guardrails``)
        -> recommended price table + reports

Anti-leakage: the engine fits/validates ONLY on Part-4 train/val periods (period labels
derived from the week column), never reads ``data/splits/test.csv``, and rejects a
reference week at/after the test boundary. Cost is an optional explicit input; without
it the engine reports ``PROFIT_NOT_COMPUTABLE`` and never invents margins.

Run:
    python -m src.optimization.pricing_engine --prepared <path> [--cost <csv>]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.optimization import elasticity as elasticity_mod
from src.optimization import pricing_data
from src.optimization import demand_baseline as demand_baseline_mod
from src.optimization.demand_baseline import (
    attach_current_price,
    baseline_frame_for_week,
    load_production_model,
    predict_baseline,
)
from src.optimization.guardrails import GuardrailConfig, apply_guardrails
from src.optimization.optimizer import optimal_price_from_panel
from src.optimization.simulation import simulate_panel
from src.utils.logging_setup import get_logger

LOGGER = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = PROJECT_ROOT / "reports" / "forecasting"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "optimization"

DEFAULT_PREPARED_CACHE = Path(
    os.environ.get(
        "PART3_PREPARED_CACHE",
        "C:/Users/HEDAYA~1/AppData/Local/Temp/opencode/experiments/"
        "final_lifecycle_phase3-exp2-v1_prepared.joblib",
    )
)


def _status_counts(eff: pd.DataFrame, all_products: pd.Series) -> dict:
    counts = {"total_products_scored": int(all_products.nunique())}
    if eff is not None and not eff.empty:
        status = eff.groupby(eff["status"]).size().to_dict()
        counts.update({f"elasticity_{k}": int(v) for k, v in status.items()})
        counts["with_effective_elasticity"] = int(eff["elasticity"].notna().sum())
    return counts


def _json_safe(obj):
    """Convert pandas frames inside the elasticity report to JSON records."""
    if isinstance(obj, pd.DataFrame):
        return obj.reset_index(drop=True).to_dict(orient="records")
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


def run_engine(
    *,
    prepared: str | Path | None = None,
    reference_week: str = pricing_data.LAST_VAL_WEEK,
    cost_path: str | Path | None = None,
    model_uri: str = "models:/demand_forecasting_final/2",
    cfg: GuardrailConfig | None = None,
    out_dir: str | Path | None = None,
) -> dict:
    """Run the full Part-4 pricing engine; returns the machine-readable report."""
    prepared = Path(prepared) if prepared is not None else DEFAULT_PREPARED_CACHE
    feature_source = "on-the-fly-frozen-val"
    if not Path(prepared).exists():
        if prepared != DEFAULT_PREPARED_CACHE:
            raise FileNotFoundError(
                f"pricing_engine: prepared lifecycle frames not found at {prepared}; "
                "pass --prepared <path> (e.g. the frozen-lifecycle prepared cache)."
            )
        LOGGER.warning(
            "pricing_engine: prepared cache missing at %s; will build frozen "
            "reference-week features from the matrix (train-only stats).",
            prepared,
        )
        prepared = None
    if reference_week >= pricing_data.TEST_START_WEEK:
        raise AssertionError(
            f"pricing_engine: reference week {reference_week} is in the Part-4 TEST "
            "period; not allowed (anti-leakage)."
        )
    cfg = cfg or GuardrailConfig()

    LOGGER.info("pricing_engine: loading weekly matrix + period labels.")
    matrix = pricing_data.load_matrix()
    periods = pricing_data.split_periods(matrix)
    train = periods["train"]
    val = periods["val"]
    pricing_data.require_periods(train, allowed=("train",))
    pricing_data.require_periods(pd.concat([train, val]), allowed=("train", "val"))

    LOGGER.info("pricing_engine: estimating elasticity on Part-4 train/val periods.")
    elast = elasticity_mod.estimate(train, val)
    eff = elast["effective"]

    LOGGER.info("pricing_engine: demand baseline from Production model for week %s.", reference_week)
    model = load_production_model(model_uri)
    if prepared is not None:
        features = baseline_frame_for_week(prepared, reference_week)
        feature_source = "frozen-prepared-cache(val)"
    else:
        features = demand_baseline_mod.frozen_features_for_week(matrix, reference_week, model=model)
    base = predict_baseline(model, features)
    matrix_ref = matrix[
        (matrix[pricing_data.WEEK_COL].astype(str) == reference_week)
        & (matrix[pricing_data.PRICE_COL].fillna(0) > 0)
    ].copy()
    merged = attach_current_price(base, matrix_ref)

    eff_map = {
        str(r[pricing_data.PRODUCT_COL]): r["elasticity"]
        for _, r in eff.iterrows()
        if pd.notna(r["elasticity"])
    }
    status_map = {
        str(r[pricing_data.PRODUCT_COL]): r["status"]
        for _, r in eff.iterrows()
    }
    source_map = {
        str(r[pricing_data.PRODUCT_COL]): r["elasticity_source"]
        for _, r in eff.iterrows()
    }

    costs = pricing_data.load_costs(cost_path) if cost_path is not None else {}

    scored = merged[merged["current_price"].notna()].copy()
    skipped_no_price = int(merged["current_price"].isna().sum())

    baselines_for_sim = scored[[pricing_data.PRODUCT_COL, "baseline_demand", "current_price"]].dropna()
    sim = simulate_panel(baselines_for_sim, eff_map, cost_by_product=costs)
    recs = optimal_price_from_panel(sim, cost_by_product=costs)

    guardrail_rows = []
    for r in recs.itertuples(index=False):
        cost = costs.get(str(r.product_id))
        out = apply_guardrails(
            float(r.optimal_price),
            float(r.current_price),
            cost=cost,
            cfg=cfg,
        )
        guardrail_rows.append(
            {
                pricing_data.PRODUCT_COL: str(r.product_id),
                "recommended_price": out.recommended_price,
                "guardrail_margin_floor_applied": out.margin_floor_applied,
                "guardrail_margin_floor_inactive_no_cost": out.margin_floor_inactive_no_cost,
                "guardrail_ceiling_applied": out.ceiling_applied,
                "guardrail_floor_applied": out.floor_applied,
                "guardrail_smoothed": out.smoothed,
                "guardrail_notes": list(out.notes),
            }
        )
    guards = pd.DataFrame(guardrail_rows)
    status_cols = pd.DataFrame(
        {
            pricing_data.PRODUCT_COL: scored[pricing_data.PRODUCT_COL].astype(str),
            "elasticity_status": scored[pricing_data.PRODUCT_COL].astype(str).map(status_map),
            "elasticity_source": scored[pricing_data.PRODUCT_COL].astype(str).map(source_map),
        }
    )

    result = scored.merge(status_cols, on=pricing_data.PRODUCT_COL, how="left")
    result = result.merge(
        recs.drop(columns=["current_price", "baseline_demand"], errors="ignore"),
        on=pricing_data.PRODUCT_COL,
        how="left",
    )
    if not guards.empty:
        result = result.merge(guards, on=pricing_data.PRODUCT_COL, how="left")

    has_elasticity = result["elasticity_source"].notna() & (result["elasticity_source"] != "none")
    result["profit_computable"] = result[pricing_data.PRODUCT_COL].astype(str).isin(costs) & has_elasticity

    def reason(row):
        if pd.isna(row.get("current_price")):
            return "NO_PRICE_OBSERVATION"
        src = row.get("elasticity_source")
        if src is None or (isinstance(src, float) and pd.isna(src)) or src == "none":
            return "NO_ELASTICITY"
        if not row.get("profit_computable", False):
            return "PROFIT_NOT_COMPUTABLE (no cost supplied)"
        return "optimized"

    result["reason"] = result.apply(reason, axis=1)
    result["recommended_price"] = result.get("recommended_price")

    def _flag_count(name: str) -> int:
        col = result.get(name)
        if col is None:
            return 0
        if col.dtype == bool:
            return int(col.sum())
        return int(col.astype("boolean").fillna(False).sum())

    n_changed = int((result["reason"] == "optimized").sum())
    n_profit_ok = int(result["profit_computable"].sum())
    n_optimized = int(result["reason"].eq("optimized").sum())
    n_extrap = _flag_count("extrapolation_required")

    report = {
        "phase": "part4_dynamic_pricing",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "reference_week": reference_week,
        "feature_source": feature_source,
        "test_split_accessed": False,
        "cost_source": str(cost_path) if cost_path else None,
        "profit_computed": bool(cost_path is not None),
        "elasticity": _json_safe(elast),
        "summary": {
            "products_scored": int(scored[pricing_data.PRODUCT_COL].nunique()),
            "products_skipped_no_price": skipped_no_price,
            "products_with_baseline_price": int(result["current_price"].notna().sum()),
            "products_with_effective_elasticity": int(
                result["reason"].ne("NO_ELASTICITY").sum()
            ),
            "products_recommended_optimized": n_optimized,
            "products_price_recommended_change": n_changed,
            "products_profit_computable": n_profit_ok,
            "products_extrapolation_flagged": n_extrap,
            "guardrail": {
                "margin_floor_applied": _flag_count("guardrail_margin_floor_applied"),
                "margin_floor_inactive_no_cost": _flag_count("guardrail_margin_floor_inactive_no_cost"),
                "ceiling_applied": _flag_count("guardrail_ceiling_applied"),
                "floor_applied": _flag_count("guardrail_floor_applied"),
                "smoothed": _flag_count("guardrail_smoothed"),
            },
        },
    }

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    out_dir = Path(out_dir) if out_dir else REPORT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    keep = [
        pricing_data.PRODUCT_COL, "category", "baseline_demand", "current_price",
        "elasticity_status", "elasticity_source", "elasticity", "optimal_price",
        "optimal_multiplier", "objective", "recommended_price", "reason",
        "profit_computable", "extrapolation_required", "optimal_revenue",
        "optimal_profit", "guardrail_margin_floor_applied",
        "guardrail_margin_floor_inactive_no_cost", "guardrail_ceiling_applied",
        "guardrail_floor_applied", "guardrail_smoothed",
    ]
    export_cols = [c for c in keep if c in result.columns]
    export = result[export_cols].copy()
    export.to_csv(ARTIFACT_DIR / "recommended_prices.csv", index=False)

    (out_dir / "part4_pricing_engine.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8"
    )
    _render_markdown(report, result, out_dir / "part4_pricing_engine_report.md")

    LOGGER.info(
        "pricing_engine: report done - scored %d products, %d optimized, profit %s.",
        int(scored[pricing_data.PRODUCT_COL].nunique()),
        n_optimized,
        "computed" if costs else "NOT computed (no cost)",
    )
    return report


def _render_markdown(report: dict, result: pd.DataFrame, path: Path) -> None:
    s = report["summary"]
    lines = [
        "# Part-4 Pricing Engine Report (Dynamic Pricing)",
        "",
        f"* Generated: {report['generated_utc']}",
        f"* Reference week: `{report['reference_week']}` (Part-4 VAL period, test untouched: {report['test_split_accessed']})",
        f"* Cost source: `{report['cost_source'] or 'NONE - profit NOT computable'}`",
        "",
        "## Summary",
        "",
        "| Metric | value |",
        "|---|---|",
    ]
    for k, v in s.items():
        if k != "guardrail":
            lines.append(f"| {k} | {v} |")
    lines.append("")
    lines.append("| Guardrail | applied count |")
    lines.append("|---|---|")
    for k, v in s["guardrail"].items():
        lines.append(f"| {k} | {v} |")
    lines.append("")
    lines.append("## Elasticity (pooled, product FE)")
    pe = report["elasticity"]["pooled_product_fe"]
    lines.append(
        f"- pooled log-log b={pe['b']:.4f}, se={pe['se']:.4f}, t={pe['t']:.3f}, "
        f"n={pe['n']}, R2={pe['r2']:.4f}"
    )
    lines.append("- **Descriptive** demand response; no causal claim (no experimental/instrumental price variation).")
    lines.append("")
    lines.append("## Recommended-price sample (top 10 absolute change)")
    opt = result[result["recommended_price"].notna()].sort_values(
        "recommended_price", ascending=False
    ).head(10)
    if not opt.empty:
        tbl = opt[["product_id", "category", "reason", "elasticity_source",
                   "current_price", "optimal_price", "recommended_price", "objective"]]
        lines.append("```")
        lines.append(tbl.to_string(index=False))
        lines.append("```")
    lines.append("")
    lines.append("> Profit optimization is NOT executable until a valid cost source is supplied; until then")
    lines.append("> the revenue-optimal scenario is returned and flagged `PROFIT_NOT_COMPUTABLE`. Elasticity")
    lines.append("> estimates are descriptive, not causal.")
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Part-4 dynamic pricing engine.")
    parser.add_argument("--prepared", default=str(DEFAULT_PREPARED_CACHE),
                        help="path to frozen-lifecycle prepared frames (joblib)")
    parser.add_argument("--reference-week", default=pricing_data.LAST_VAL_WEEK,
                        help="reference week (week-start label) must be < test boundary")
    parser.add_argument("--cost", default=None, help="optional cost CSV (product_id, cost)")
    parser.add_argument("--model-uri", default="models:/demand_forecasting_final/2")
    args = parser.parse_args(argv)
    report = run_engine(
        prepared=args.prepared,
        reference_week=args.reference_week,
        cost_path=args.cost,
        model_uri=args.model_uri,
    )
    print(json.dumps(report["summary"], indent=2, default=str))


if __name__ == "__main__":
    main()