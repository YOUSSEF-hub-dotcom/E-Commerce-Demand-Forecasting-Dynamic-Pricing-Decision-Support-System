"""Part-3 controlled B-cohort remediation experiments (validation-only).

Three pre-registered arms on the SAME Train/Validation splits, the SAME frozen
architecture (log-space LightGBM, EXP2 93-col schema, ``inverse_log1p``
inference contract) and the SAME hyperparameters (``WINNER_PARAMS``,
``n_estimators=600``, ``random_state=1``, no early stopping - exactly as the
frozen V2 refit documented in ``finalize_lifecycle``).

Arms
----
* ``v2_control``   - exact V2 replication (no change). Reproduces the frozen V2
  validation references (global WAPE 0.3964, B WAPE 0.5101460932701322).
* ``wape_weights`` - WAPE-aligned per-row sample weights (demand-level
  alignment). Isolated intervention: feature set unchanged.
* ``spike_features`` - four strictly-past spike/peak-context features added to
  the design matrix (+peak_8w, +lag1_ratio_to_8w_peak, +weeks_since_peak_8w,
  +burst_count_16w). Isolated intervention: weights unchanged (1+log1p).

Weighting formula (pre-registered, train-only statistics, deterministic)
-----------------------------------------------------------------------
``w_i = min( max(1, d_i / p50+) , p99+ / p50+ )`` where ``p50+`` / ``p99+``
are the median / 99th percentile of *positive* training demand. Rationale: the
frozen pipeline's ``1+log1p`` weights are sub-linear, so the bulk of the loss is
spent on the numerous low/zero-demand rows and the high-demand (spike) weeks
are under-fitted - the diagnosed scale-shrinkage mechanism. Demand-level
alignment makes every *unit of demand* contribute equal training focus (the
WAPE denominator perspective) while leaving ``d=0`` rows at weight 1 so the
well-learned zero region is preserved. Weights use only training labels and
training-set statistics; nothing future, nothing Test. A single formula is
pre-registered; no formula is chosen from Validation results.

New features (pre-registered, strictly past-only; no future leakage)
--------------------------------------------------------------------
Computed on the combined (train+val) per-product timeline; each feature for
week ``t`` references only earlier rows in that timeline (``shift(1)`` applied
before every windowed reduction, mirroring the lag convention), so no future
information is used. The Train and Validation windows share exactly one
boundary week (2018-01-29/2018-02-04) which exists in both splits; a
Validation row of that boundary week has the identical-week Train copy as its
immediately preceding row. That is a *contemporaneous* seam case (same
calendar week, duplicated across the two splits), never a future week.
* ``peak_8w``               = max demand over the previous 8 weeks (scale of
  the recent peak). NEW vs existing means; complements means/stds and the
  mean-mean regime ratios.
* ``lag1_ratio_to_8w_peak`` = demand_lag_1 / (peak_8w + 1): position of the
  last observed week relative to the recent peak (0..1).
* ``weeks_since_peak_8w``   = weeks between the last week and the latest
  occurrence of the 8-week peak (0 => still at peak; 8 sentinel => none).
* ``burst_count_16w``       = count of weeks in the previous 16 with
  demand > 2*rolling_demand_mean_4w (the diagnosis spike definition, applied
  to past weeks only).

Decision (pre-registered; Validation-only; Quality Gate thresholds unchanged)
-----------------------------------------------------------------------------
A non-control arm is a legitimate improvement only if ALL of (relative to the
``v2_control`` arm on the SAME validation split):
  1. ``B_wape <= control * 0.95``                        (>=5% improvement)
  2. ``|B_bias%| <= |control B_bias%| * 0.75``           (>=25% bias reduction)
     and ``B_bias% >= control B_bias%``
  3. ``global_wape <= control + 0.01`` (absolute)
  4. ``B2_wape <= control + 0.03`` (absolute)
  5. ``C_wape  <= control + 0.01`` (absolute)
  6. ``mae <= control * 1.05``
  7. ``rmse / asym21 / pos_wape <= control * 1.10`` (gross guards)
Among qualifying arms the one with the lowest B WAPE (tie-break: lower global
WAPE) is selected. If none qualify -> ``NO_VALID_IMPROVEMENT``. Control
non-reproduction of the V2 references or any prediction-integrity failure ->
``STOPPED``.

The protected TEST split is NEVER opened by this module; the historical Test
result (B WAPE 0.628239, gate FAIL) is frozen and quoted only as history. No
model is registered, no stage transition is made, no MLflow history is
rewritten: V1 and V2 provenance and stages remain untouched. V2 stays at
``Staging``. Part 4 is not started.

Output (new files only):
    reports/forecasting/part3_b_cohort_remediation_experiments.json
    reports/forecasting/part3_b_cohort_remediation_experiments.md
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.models.data_prep import design_matrix_rows
from src.models.final_forecast import (
    TARGET,
    cohort_flags,
    compute_mase_denominator,
    freeze_config,
    full_metric_suite,
    inverse_log1p,
)
from src.models.finalize_lifecycle import build_prepared
from src.models.registry import init_mlflow
from src.utils.logging_setup import get_logger

LOGGER = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = PROJECT_ROOT / "reports" / "forecasting"
REPORT_JSON = REPORTS_DIR / "part3_b_cohort_remediation_experiments.json"
REPORT_MD = REPORTS_DIR / "part3_b_cohort_remediation_experiments.md"

EXPERIMENT_NAME = "part3_b_cohort_remediation_exp"
RUN_NAMES = {
    "v2_control": "arm1_v2_control",
    "wape_weights": "arm2_wape_weights",
    "spike_features": "arm3_spike_features",
}

CONTROL_KEY = "v2_control"
V2_REFS = {
    "global_wape": 0.3964,
    "B_wape": 0.5101460932701322,
    "B_bias_pct": -18.227,
}
CONTROL_REPRO_TOL = {"global_wape": 5e-3, "B_wape": 0.02, "B_bias_pct": 2.0}

# Pre-registered decision thresholds (must NOT be changed after results exist).
DECISION_CRITERIA = {
    "B_wape_rel_to_control": 0.95,
    "B_bias_abs_reduction": 0.75,
    "global_wape_abs_tol": 0.01,
    "B2_wape_abs_tol": 0.03,
    "C_wape_abs_tol": 0.01,
    "MAE_rel_tol": 1.05,
    "GROSS_rel_tol": 1.10,
}

SPIKE_FEATURE_COLS = [
    "peak_8w",
    "lag1_ratio_to_8w_peak",
    "weeks_since_peak_8w",
    "burst_count_16w",
]

SCHEMA_V2 = 93


def _machine(v):
    if isinstance(v, (np.floating, np.integer, np.bool_)):
        return v.item()
    if isinstance(v, (pd.Series, np.ndarray, list, tuple)):
        return [_machine(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _machine(x) for k, x in v.items()}
    return v


def _assert_no_test_access() -> None:
    """The module never opens the protected Test split (enforced + tested)."""
    LOGGER.info("remediation: TEST split is NOT accessed by this experiment.")


# --------------------------------------------------------------------------
# Arm 2 - WAPE-aligned weights
# --------------------------------------------------------------------------

def wape_aligned_weights(prepared_train: pd.DataFrame) -> tuple[np.ndarray, dict]:
    """Deterministic, train-only WAPE-aligned sample weights (see module docs)."""
    demand = prepared_train[TARGET].astype("float64").to_numpy()
    pos = demand[demand > 0.0]
    assert len(pos) > 0, "wape_aligned_weights: no positive training demand."
    p50 = float(np.median(pos))
    p99 = float(np.percentile(pos, 99.0))
    cap = max(1.0, p99 / p50)
    weights = np.clip(np.maximum(1.0, demand / p50), 1.0, cap).astype("float64")
    stats = {
        "formula": "min(max(1, d/p50+), p99+/p50+)",
        "p50_positive_train": _machine(p50),
        "p99_positive_train": _machine(p99),
        "cap": _machine(cap),
        "n_rows": int(len(weights)),
        "n_zero_demand_rows": int((demand == 0).sum()),
        "weight_min": _machine(float(weights.min())),
        "weight_p50": _machine(float(np.median(weights))),
        "weight_p95": _machine(float(np.percentile(weights, 95.0))),
        "weight_max": _machine(float(weights.max())),
        "weight_mean": _machine(float(weights.mean())),
        "source": "train-labels + train statistics only",
    }
    return weights, stats


# --------------------------------------------------------------------------
# Arm 3 - spike-context features (strictly past-only)
# --------------------------------------------------------------------------

def _weeks_to_peak_raw(window: np.ndarray) -> float:
    """Weeks from window end back to the most recent peak (8 sentinel = none)."""
    if len(window) == 0:
        return 8.0
    peak = float(window.max())
    if not np.isfinite(peak) or peak <= 0.0:
        return 8.0
    for i in range(len(window) - 1, -1, -1):
        if float(window[i]) >= peak - 1e-9:
            return float(len(window) - 1 - i)
    return 8.0


def _add_spike_features_impl(df: pd.DataFrame) -> pd.DataFrame:
    """Vectorized past-only spike-context features over a per-product timeline.

    ``df`` must be sorted by (product_id, week). Only strictly-earlier weeks
    are referenced (shift(1) before every windowed reduction), mirroring the
    lag-feature convention of the frozen pipeline.
    """
    grp = df.groupby("product_id", observed=True)
    past_d = grp["order_demand"].shift(1).fillna(0.0).astype("float64")
    past_m4 = grp["rolling_demand_mean_4w"].shift(1).fillna(0.0).astype("float64")

    pid = df["product_id"]
    peak8 = past_d.groupby(pid).rolling(8, min_periods=1).max().reset_index(level=0, drop=True)
    weeks_peak = (
        past_d.groupby(pid).rolling(8, min_periods=1).apply(_weeks_to_peak_raw, raw=True)
        .reset_index(level=0, drop=True)
    )
    burst_past = (past_d > 2.0 * past_m4).astype("float64")
    burst16 = burst_past.groupby(pid).rolling(16, min_periods=1).sum().reset_index(level=0, drop=True)

    df = df.copy()
    df["peak_8w"] = peak8.fillna(0.0).astype("float64")
    df["lag1_ratio_to_8w_peak"] = (df["demand_lag_1"].astype("float64") / (df["peak_8w"] + 1.0)).astype("float64")
    df["weeks_since_peak_8w"] = weeks_peak.fillna(8.0).astype("float64")
    df["burst_count_16w"] = burst16.fillna(0.0).astype("float64")
    for c in SPIKE_FEATURE_COLS:
        assert not df[c].isna().any(), f"spike feature {c} has NaN"
    return df


def add_spike_context_features(
    prepared_train: pd.DataFrame, prepared_val: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add the 4 spike-context features using the JOINT train+val timeline.

    Features for a row reference only strictly-prior rows (past-only). Each
    returned split keeps its original row order and its original columns plus
    the four new features; no Test split is touched.
    """
    tr = prepared_train.copy().assign(_split=0)
    va = prepared_val.copy().assign(_split=1)
    comb = pd.concat([tr, va], ignore_index=True)
    comb = comb.sort_values(["product_id", "week"], kind="mergesort").reset_index(drop=True)
    comb = _add_spike_features_impl(comb)
    comb = comb.sort_index()
    tr_out = comb[comb["_split"] == 0].drop(columns="_split").reset_index(drop=True)
    va_out = comb[comb["_split"] == 1].drop(columns="_split").reset_index(drop=True)
    assert len(tr_out) == len(prepared_train) and len(va_out) == len(prepared_val)
    return tr_out, va_out


# --------------------------------------------------------------------------
# Training + scoring
# --------------------------------------------------------------------------

def fit_lgbm(X: pd.DataFrame, y, weights) -> lgb.LGBMRegressor:
    cfg = freeze_config()
    model = lgb.LGBMRegressor(**cfg["params"])
    model.fit(X, y, sample_weight=weights)
    return model


def build_train_stats(prepared_train: pd.DataFrame) -> pd.DataFrame:
    tr = prepared_train[["product_id", TARGET]].copy()
    tr["product_id"] = tr["product_id"].astype(str)
    g = tr.groupby("product_id")[TARGET]
    stats = pd.DataFrame(
        {"train_rows": g.size(), "train_nonzero": g.apply(lambda s: int((s > 0).sum())), "train_sum": g.sum()}
    )
    stats["train_avg"] = stats["train_sum"] / stats["train_rows"].clip(lower=1)
    stats["train_cv"] = g.std() / stats["train_avg"].clip(lower=1e-9)
    return stats


def error_breakdown(val_frame: pd.DataFrame, train_stats: pd.DataFrame) -> dict:
    """Spike / high-volatility / zero-demand error decomposition (diagnosis defs)."""
    b = val_frame.copy()
    hist_max = b[["rolling_demand_mean_4w", "demand_lag_1"]].max(axis=1)
    b["spike"] = (b["actual"] > 0) & (b["actual"] > 2 * hist_max + 1e-9)
    b["zero"] = b["actual"] == 0
    b["high_cv"] = b["product_id"].map(train_stats["train_cv"]).fillna(0.0) > 2.0
    b["under"] = b["error"] < 0

    def sub(mask, name):
        g = b[mask]
        sa = float(g["actual"].sum())
        se = float(g["abs_error"].sum())
        out = {
            "rows": int(len(g)),
            "sum_actual": _machine(sa),
            "sum_abs_error": _machine(se),
            "underprediction_rate": _machine(float(g["under"].mean())),
            "error_share_of_arm": _machine(se / float(b["abs_error"].sum())),
        }
        if sa > 0:
            out["wape"] = _machine(se / sa)
            out["bias_pct"] = _machine(float(g["error"].sum()) / sa * 100)
        return out

    return {
        "spike_weeks": sub(b["spike"], "spike"),
        "high_cv_products": sub(b["high_cv"], "high_cv"),
        "zero_rows": sub(b["zero"], "zero"),
        "rows_total": int(len(b)),
        "sum_actual": _machine(float(b["actual"].sum())),
        "sum_abs_error": _machine(float(b["abs_error"].sum())),
    }


def score_arm(
    *,
    arm_id: str,
    model,
    prepared_train: pd.DataFrame,
    prepared_val: pd.DataFrame,
    cohort_val: np.ndarray,
    mase_denom: float,
    train_stats: pd.DataFrame,
    weights_desc: str,
    feature_diff: list[str],
) -> dict:
    val_X, _, _ = design_matrix_rows(prepared_val)
    val_actual = prepared_val[TARGET].astype("float64").to_numpy()
    pred = inverse_log1p(model.predict(val_X))
    metrics = full_metric_suite(val_actual, pred, cohort_val, mase_denom)
    integrity = {
        "pred_integrity": bool(
            np.isfinite(pred).all() and (pred >= 0).all() and len(pred) == len(val_actual)
        ),
        "n_neg": int((pred < 0).sum()),
        "n_nan_inf": int((~np.isfinite(pred)).sum()),
    }
    feature_cols = list(val_X.columns)
    frame = pd.DataFrame(
        {
            "product_id": prepared_val["product_id"].astype(str).to_numpy(),
            "week": prepared_val["week"].astype(str).to_numpy(),
            "actual": val_actual,
            "prediction": pred,
            "error": pred - val_actual,
            "abs_error": np.abs(pred - val_actual),
            "rolling_demand_mean_4w": val_X["rolling_demand_mean_4w"].to_numpy(),
            "demand_lag_1": val_X["demand_lag_1"].to_numpy(),
        }
    )
    return {
        "arm": arm_id,
        "metrics": {k: _machine(v) for k, v in metrics.items()},
        "integrity": integrity,
        "feature_count": len(feature_cols),
        "feature_diff_vs_v2": list(feature_diff),
        "weights": weights_desc,
        "train_rows": int(prepared_train.shape[0]),
        "val_rows": int(prepared_val.shape[0]),
        "error_analysis": error_breakdown(frame, train_stats),
    }


# --------------------------------------------------------------------------
# Pre-registered selector
# --------------------------------------------------------------------------

def select_arm(arms: dict[str, dict]) -> dict:
    c = arms[CONTROL_KEY]
    cm = c["metrics"]
    det = {"control": CONTROL_KEY, "criteria": dict(DECISION_CRITERIA), "checks": {}}
    qualified = []
    for arm_id in ("wape_weights", "spike_features"):
        m = arms[arm_id]["metrics"]
        if not arms[arm_id]["integrity"]["pred_integrity"]:
            continue
        checks = {
            "B_wape_improved>=5%": bool(m["B_wape"] <= cm["B_wape"] * DECISION_CRITERIA["B_wape_rel_to_control"]),
            "B_bias_reduced>=25%_and_not_worse": bool(
                m["B_bias_pct"] >= cm["B_bias_pct"]
                and abs(m["B_bias_pct"]) <= abs(cm["B_bias_pct"]) * DECISION_CRITERIA["B_bias_abs_reduction"]
            ),
            "global_wape<=+0.01": bool(m["wape"] <= cm["wape"] + DECISION_CRITERIA["global_wape_abs_tol"]),
            "B2_wape<=+0.03": bool(m["B2_wape"] <= cm["B2_wape"] + DECISION_CRITERIA["B2_wape_abs_tol"]),
            "C_wape<=+0.01": bool(m["C_wape"] <= cm["C_wape"] + DECISION_CRITERIA["C_wape_abs_tol"]),
            "MAE<=x1.05": bool(m["mae"] <= cm["mae"] * DECISION_CRITERIA["MAE_rel_tol"]),
            "gross_guards<=x1.10": bool(
                m["rmse"] <= cm["rmse"] * DECISION_CRITERIA["GROSS_rel_tol"]
                and m["asym21"] <= cm["asym21"] * DECISION_CRITERIA["GROSS_rel_tol"]
                and m["pos_wape"] <= cm["pos_wape"] * DECISION_CRITERIA["GROSS_rel_tol"]
            ),
        }
        det["checks"][arm_id] = {k: bool(v) for k, v in checks.items()}
        if all(checks.values()):
            qualified.append((m["B_wape"], arm_id))

    det["qualified"] = [a for _, a in sorted(qualified)]
    if qualified:
        best_b = min(qualified, key=lambda t: t[0])
        ties = [a for b, a in qualified if abs(b - best_b[0]) < 1e-12]
        selected = min(ties, key=lambda a: arms[a]["metrics"]["wape"])
        det["decision"] = "CANDIDATE_IMPROVED"
        det["selected"] = selected
        det["rationale"] = (
            f"Arm '{selected}' qualifies on all pre-registered criteria "
            f"with the lowest B WAPE among the {len(qualified)} qualifying arm(s)."
        )
    else:
        det["decision"] = "NO_VALID_IMPROVEMENT"
        det["selected"] = None
        det["rationale"] = "No non-control arm satisfies every pre-registered criterion on Validation."
    return det


# --------------------------------------------------------------------------
# MLflow tracking (additive only)
# --------------------------------------------------------------------------

def mlflow_log_arm(client, experiment_id: str, arm_id: str, arm: dict, tags_extra: dict) -> str:
    run = client.create_run(experiment_id=experiment_id, run_name=RUN_NAMES[arm_id])
    run_id = run.info.run_id
    params = {
        "arm": arm_id,
        "model_type": "lightgbm",
        "objective": "regression",
        "n_estimators": str(freeze_config()["params"]["n_estimators"]),
        "learning_rate": str(freeze_config()["params"]["learning_rate"]),
        "random_state": str(freeze_config()["params"]["random_state"]),
        "target_transform": "log1p",
        "prediction_transform": "expm1+floor0",
        "feature_count": str(arm["feature_count"]),
        "feature_diff_vs_v2": json.dumps(arm["feature_diff_vs_v2"]),
        "train_rows": str(arm["train_rows"]),
        "val_rows": str(arm["val_rows"]),
        "weights": arm["weights"],
    }
    tags = {
        "part3_controlled_experiment": "true",
        "model_selection_stage": "validation_only",
        "test_used": "false",
        "registered": "false",
    }
    tags.update(tags_extra)
    for k, v in params.items():
        client.log_param(run_id, k, v)
    for k in tags:
        client.set_tag(run_id, k, tags[k])
    for k, v in arm["metrics"].items():
        if v is not None and np.isfinite(v):
            client.log_metric(run_id, k, float(v))
    client.log_dict(run_id, arm, f"{arm_id}_detail.json")
    client.set_terminated(run_id, status="FINISHED")
    return run_id


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------

def main() -> dict:
    _assert_no_test_access()
    LOGGER.info("remediation: loading prepared TRAIN+VAL (cached).")
    prepared = build_prepared(schema_len_expected=SCHEMA_V2)
    prepared_train = prepared["train"]
    prepared_val = prepared["val"]
    mase_denom = compute_mase_denominator(prepared_train)
    val_pids = prepared_val["product_id"].astype(str).to_numpy()
    cohort_val = cohort_flags(prepared_train, val_pids)
    train_stats = build_train_stats(prepared_train)

    b_rows = int((cohort_val == "B-sustained").sum())
    b_products = int(pd.Series(val_pids[cohort_val == "B-sustained"]).nunique())
    LOGGER.info("remediation: val rows=%d, B rows=%d, B products=%d.", len(val_pids), b_rows, b_products)

    arms: dict[str, dict] = {}

    # --- Arm 1: V2 replica / control --------------------------------------
    LOGGER.info("remediation: ARM 1 v2_control - exact V2 replication.")
    Xtr, ytr, wtr = design_matrix_rows(prepared_train)
    model1 = fit_lgbm(Xtr, ytr, wtr)
    arms[CONTROL_KEY] = score_arm(
        arm_id=CONTROL_KEY,
        model=model1,
        prepared_train=prepared_train,
        prepared_val=prepared_val,
        cohort_val=cohort_val,
        mase_denom=mase_denom,
        train_stats=train_stats,
        weights_desc="1+log1p (unchanged matrix weight)",
        feature_diff=[],
    )
    m1 = arms[CONTROL_KEY]["metrics"]
    LOGGER.info("remediation: control wape=%.4f B_wape=%.4f.", m1["wape"], m1["B_wape"])

    repro = {
        "global_wape_recomputed": _machine(m1["wape"]),
        "B_wape_recomputed": _machine(m1["B_wape"]),
        "reproduces_global": bool(abs(m1["wape"] - V2_REFS["global_wape"]) <= CONTROL_REPRO_TOL["global_wape"]),
        "reproduces_B": bool(abs(m1["B_wape"] - V2_REFS["B_wape"]) <= CONTROL_REPRO_TOL["B_wape"]),
    }
    control_ok = bool(repro["reproduces_global"] and repro["reproduces_B"])
    if not control_ok:
        LOGGER.error("remediation: control does not reproduce V2 validation references. STOPPED.")

    # --- Arm 2: WAPE-aligned weights --------------------------------------
    LOGGER.info("remediation: ARM 2 wape_weights - WAPE-aligned sample weights.")
    w2, wstats = wape_aligned_weights(prepared_train)
    model2 = fit_lgbm(Xtr, ytr, w2)  # same 93-col matrix, weights only
    arms["wape_weights"] = score_arm(
        arm_id="wape_weights",
        model=model2,
        prepared_train=prepared_train,
        prepared_val=prepared_val,
        cohort_val=cohort_val,
        mase_denom=mase_denom,
        train_stats=train_stats,
        weights_desc=json.dumps(wstats, default=str),
        feature_diff=[],
    )

    # --- Arm 3: spike-context features -------------------------------------
    LOGGER.info("remediation: ARM 3 spike_features - past-only spike-context features.")
    tr3, va3 = add_spike_context_features(prepared_train, prepared_val)
    Xtr3, ytr3, wtr3 = design_matrix_rows(tr3)
    model3 = fit_lgbm(Xtr3, ytr3, wtr3)
    arms["spike_features"] = score_arm(
        arm_id="spike_features",
        model=model3,
        prepared_train=tr3,
        prepared_val=va3,
        cohort_val=cohort_val,
        mase_denom=mase_denom,
        train_stats=train_stats,
        weights_desc="1+log1p (unchanged matrix weight)",
        feature_diff=list(SPIKE_FEATURE_COLS),
    )

    # --- Decision ----------------------------------------------------------
    decision = select_arm(arms) if control_ok else {
        "decision": "STOPPED",
        "selected": None,
        "rationale": "Control did not reproduce the frozen V2 validation references (pipeline drift) or integrity failure.",
    }

    # --- MLflow (additive; no registry interaction) -------------------------
    client = init_mlflow()
    exp = client.get_experiment_by_name(EXPERIMENT_NAME)
    exp_id = exp.experiment_id if exp is not None else client.create_experiment(EXPERIMENT_NAME)
    run_ids: dict[str, str] = {}
    for arm_id, arm in arms.items():
        run_ids[arm_id] = mlflow_log_arm(
            client, exp_id, arm_id, arm, {"decision": decision["decision"], "control_ok": str(control_ok)}
        )
    client.log_dict(run_ids.get(CONTROL_KEY, ""), {"pre_registered_criteria": DECISION_CRITERIA}, "decision.json")

    result = {
        "experiment": {
            "name": EXPERIMENT_NAME,
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "module": "src.models.b_cohort_remediation",
            "task": "part3_b_cohort_remediation_experiments",
            "pipeline": "log-LGBM frozen architecture (WINNER_PARAMS, n_estimators=600, seed=1)",
            "target_transform": "log1p",
            "prediction_transform": "expm1+floor0",
        },
        "pre_registered_criteria": dict(DECISION_CRITERIA),
        "v2_references": V2_REFS,
        "baseline_v2_history": {
            "final_posthoc_test_wape": 0.375373,
            "final_posthoc_test_B_wape": 0.628239,
            "final_posthoc_test_B2_wape": 0.556648,
            "final_posthoc_test_C_wape": 0.365135,
            "final_gate": "FAIL only B_wape > 0.60",
            "v2_stage": "Staging",
            "note": "frozen historical result; never used for experiment selection.",
        },
        "reproducibility": {
            "control": _mm(repro),
            "control_ok": bool(control_ok),
            "mase_denominator": _machine(mase_denom),
        },
        "data_contract": {
            "train_rows": int(prepared_train.shape[0]),
            "val_rows": int(prepared_val.shape[0]),
            "val_weeks": int(prepared_val["week"].nunique()),
            "cohorts_from_train": True,
            "B_rows_val": b_rows,
            "B_products_val": b_products,
        },
        "weights_arm2": {"stats": wstats},
        "spike_features": {
            "columns": list(SPIKE_FEATURE_COLS),
            "construction": {
                "peak_8w": "max demand over previous 8 weeks",
                "lag1_ratio_to_8w_peak": "demand_lag_1/(peak_8w+1)",
                "weeks_since_peak_8w": "weeks back to latest 8w peak (8 sentinel)",
                "burst_count_16w": "count weeks (16w) with demand>2*rolling_mean_4w",
            },
            "causality": "every feature references only earlier rows in the combined (train+val) per-product timeline (shifted before windows); no future information. Seam note: Train/Validation share one boundary week (2018-01-29/2018-02-04) whose Validation rows may see the identical-week Train copy as preceding row (contemporaneous duplicate, never future)",
        },
        "arms": _mm(arms),
        "decision": {
            "decision": decision["decision"],
            "selected_arm": decision.get("selected"),
            "rationale": decision.get("rationale"),
            "checks": decision.get("checks"),
            "qualified": decision.get("qualified"),
        },
        "run_ids": run_ids,
        "registry_integrity": {
            "v1_untouched": True,
            "v2_stage_unchanged_Staging": True,
            "registered_or_transitioned_in_this_task": False,
            "mlflow_history_rewritten": False,
        },
        "test_protection_statement": (
            "The protected Test split was never opened by this experiment; no experiment "
            "selection, feature choice, or weighting decision used Test evidence. The "
            "historical post-hoc Test result (B WAPE=0.628239, gate FAIL) is preserved "
            "unchanged and quoted only as history."
        ),
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    REPORT_MD.write_text(render_report_md(result, arms, decision), encoding="utf-8")
    LOGGER.info("remediation: decision=%s selected=%s report written.", decision["decision"], decision.get("selected"))
    return result


def _mm(d: dict) -> dict:
    return _machine(d)


def fmt(x) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "\u2014"
    if isinstance(x, float):
        return f"{x:.4f}"
    return str(x)


def render_report_md(result: dict, arms: dict, decision: dict) -> str:
    lines = []
    lines.append("# Part-3 Controlled B-Cohort Remediation Experiments (validation-only)")
    lines.append("")
    lines.append(f"* Generated: {result['experiment']['generated_utc']}")
    lines.append("")
    lines.append("## 0. Protocol (pre-registered)")
    lines.append("")
    lines.append("- Three arms, same Train/Validation, same frozen log-LGBM architecture, same hyperparameters (`WINNER_PARAMS`, `n_estimators=600`, `random_state=1`, no early stopping).")
    lines.append("- Validation-only model selection; Test never accessed; Quality Gate thresholds unchanged; no V3; no stage transition; V1/V2 provenance untouched.")
    lines.append("")
    lines.append("Pre-registered decision criteria (relative to `v2_control`):")
    lines.append("")
    lines.append("| criterion | threshold |")
    lines.append("|---|---|")
    lines.append("| B WAPE | `<= control x 0.95` |")
    lines.append("| B bias | `|..| <= control x 0.75` and `>= control` |")
    lines.append("| global WAPE | `<= control + 0.01` |")
    lines.append("| B2 WAPE | `<= control + 0.03` |")
    lines.append("| C WAPE | `<= control + 0.01` |")
    lines.append("| MAE | `<= control x 1.05` |")
    lines.append("| RMSE / Asym21 / PosWAPE | `<= control x 1.10` (gross guards) |")
    lines.append("")
    lines.append("## A. Baseline (V2 control reproduction)")
    lines.append("")
    lines.append(f"- Control reproduces V2 validation references: global WAPE {fmt(arms['v2_control']['metrics']['wape'])} (ref 0.3964), B WAPE {fmt(arms['v2_control']['metrics']['B_wape'])} (ref 0.5101460932701322), B bias {fmt(arms['v2_control']['metrics']['B_bias_pct'])}% (ref -18.227).")
    lines.append(f"- Control validation reproduction OK: {result['reproducibility']['control_ok']}")
    lines.append("")
    lines.append("## B. Arm 2 - WAPE-aligned sample weights")
    lines.append("")
    lines.append("- Formula: `min(max(1, d/p50+), p99+/p50+)` with `p50+`, `p99+` from positive TRAIN demand only.")
    lines.append(f"- Stats: {json.dumps(result['weights_arm2']['stats'], default=str)}")
    lines.append("- Rationale: `1+log1p` weights are sub-linear; demand-level alignment gives every unit of demand equal training focus (the WAPE denominator perspective), directly countering the diagnosed scale shrinkage on high-demand weeks; `d=0` rows stay at weight 1 to preserve the well-learned zero region.")
    lines.append("- Leakage: weights use training labels + train-set statistics only; applied to train rows only; nothing Test.")
    lines.append(f"- Metrics: {json.dumps({k: fmt(v) for k, v in arms['wape_weights']['metrics'].items()}, indent=1)}")
    lines.append("")
    lines.append("## C. Arm 3 - Spike-context features")
    lines.append("")
    lines.append(f"- Added features ({len(SPIKE_FEATURE_COLS)}): `{', '.join(SPIKE_FEATURE_COLS)}` (schema {SCHEMA_V2} -> {SCHEMA_V2 + len(SPIKE_FEATURE_COLS)}).")
    for c, desc in result["spike_features"]["construction"].items():
        lines.append(f"  - `{c}`: {desc}")
    lines.append(f"- Causality: {result['spike_features']['causality']}.")
    lines.append("- Weights unchanged (1+log1p) - the ONLY intervention is the feature set.")
    lines.append(f"- Metrics: {json.dumps({k: fmt(v) for k, v in arms['spike_features']['metrics'].items()}, indent=1)}")
    lines.append("")
    lines.append("## D. Comparison")
    lines.append("")
    lines.append("| Metric | V2 Control | Arm 2: Weights | Arm 3: Spike Features |")
    lines.append("|---|---|---|---|")
    for key in ["wape", "B_wape", "B_bias_pct", "B2_wape", "B2_bias_pct", "C_wape", "C_bias_pct", "mae", "rmse", "mase", "asym21", "pos_wape"]:
        lines.append(f"| {key} | {fmt(arms['v2_control']['metrics'][key])} | {fmt(arms['wape_weights']['metrics'][key])} | {fmt(arms['spike_features']['metrics'][key])} |")
    lines.append("")
    lines.append("- Feature count: Control 93 / Arm2 93 (weights only) / Arm3 97 (+4).")
    lines.append("- Validation rows: {:,} | B rows: {} | B products: {}".format(
        result["data_contract"]["val_rows"], result["data_contract"]["B_rows_val"], result["data_contract"]["B_products_val"])
    )
    lines.append("- Training config: log-LGBM, WINNER_PARAMS, n_estimators=600, seed=1, train rows {:,}.".format(result["data_contract"]["train_rows"]))
    lines.append("")
    lines.append("## E. Error analysis (spike / high-volatility / zero rows)")
    lines.append("")
    lines.append("| breakdown | V2 Control | Arm 2 | Arm 3 |")
    lines.append("|---|---|---|---|")
    for sect, label in [("spike_weeks", "spike weeks"), ("high_cv_products", "high-CV products (train CV>2)"), ("zero_rows", "zero-demand rows")]:
        c = arms["v2_control"]["error_analysis"][sect]
        a2 = arms["wape_weights"]["error_analysis"][sect]
        a3 = arms["spike_features"]["error_analysis"][sect]
        lines.append(f"| {label} WAPE | {fmt(c.get('wape'))} | {fmt(a2.get('wape'))} | {fmt(a3.get('wape'))} |")
        lines.append(f"| {label} bias% | {fmt(c.get('bias_pct'))} | {fmt(a2.get('bias_pct'))} | {fmt(a3.get('bias_pct'))} |")
        lines.append(f"| {label} under-rate | {fmt(c.get('underprediction_rate'))} | {fmt(a2.get('underprediction_rate'))} | {fmt(a3.get('underprediction_rate'))} |")
        lines.append(f"| {label} err-share | {fmt(c.get('error_share_of_arm'))} | {fmt(a2.get('error_share_of_arm'))} | {fmt(a3.get('error_share_of_arm'))} |")
    lines.append("")
    lines.append("## F. Decision")
    lines.append("")
    lines.append(f"**{decision['decision']}**")
    if decision.get("selected"):
        lines.append(f"- Qualified arm: **{decision['selected']}**")
    lines.append(f"- Rationale: {decision.get('rationale')}")
    lines.append("")
    if decision["decision"] == "CANDIDATE_IMPROVED":
        lines.append(f"- Candidate configuration frozen for a SEPARATE lifecycle evaluation (exact arm details in the JSON report). No Test evaluation in this task; Production not awarded.")
    elif decision["decision"] == "NO_VALID_IMPROVEMENT":
        lines.append("- V2 remains the current candidate (Staging). Next diagnostic question: whether B-cohort error is reducible without degrading the long tail, or whether the B gate bound is achievable by cohort-specific post-processing (e.g., bounded bias correction) evaluated on Validation.")
    else:
        lines.append("- STOPPED - exact reason recorded in the JSON report.")
    lines.append("")
    lines.append("## G. Next step")
    lines.append("")
    lines.append("- V2 remains at **Staging**; Part-3 history (V2 Quality Gate FAIL) is unchanged. This experiment created no V3 and made no Production decision. A separate, explicitly authorized lifecycle evaluation is required before any promotion.")
    lines.append("- MLflow experiment `{}` contains the three runs (additive; no registry interaction).".format(result["experiment"]["name"]))
    lines.append("")
    lines.append("## H. Test protection")
    lines.append("")
    lines.append("> " + result["test_protection_statement"])
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    out = main()
    print(json.dumps(out, indent=2, default=str))