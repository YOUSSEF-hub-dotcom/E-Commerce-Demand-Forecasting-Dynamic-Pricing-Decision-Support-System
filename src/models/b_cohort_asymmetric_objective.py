"""Part-3 controlled Validation-only experiment: asymmetric 2:1 training objective.

Hypothesis
----------
The diagnosed B-cohort defect is underforecasting (scale shrinkage on
high-demand/spike weeks): V2 under-forecasts B by ~ -18.2% (bias) on the frozen
Validation split. The business Cost-of-Error framework penalizes stockouts
twice as heavily as surplus (2:1). V2 is trained on ``log1p(order_demand)``
with the **symmetric** L2 objective while the 2:1 asymmetric loss is used only
for evaluation. Hypothesis: *training directly with the business-asymmetric
objective shifts the prediction target toward the upper conditional quantile
(the loss floor), reducing underforecasting on high-demand cohorts WITHOUT
materially degrading global/cohort accuracy.*

Control
-------
The frozen V2 model, reproduced exactly: same Train/Validation (cached
``build_prepared``), same 93-column schema, same ``log1p(order_demand)``
target, same ``WINNER_PARAMS`` (``n_estimators=600``, ``learning_rate``
0.008468, ``random_state=1``, no early stopping), same ``1+log1p`` matrix
weights and same validation methodology. It must reproduce the frozen V2
Validation references (global WAPE 0.39637084, B WAPE 0.5101460932701322,
B bias -18.227%) within the pre-registered tolerance, otherwise the experiment
is STOPPED (classified ``EXPERIMENT_FAILED``).

Treatment
---------
Identical pipeline and hyperparameters with ONE intervention: the training
objective becomes the built-in LightGBM **quantile (pinball) objective at
``alpha = 2/3``**. Target and prediction transforms are unchanged
(``log1p`` / ``inverse_log1p``); matrix weights unchanged (``1+log1p``).

Why the objective represents the 2:1 business asymmetry
-------------------------------------------------------
The project's evaluation ``asymmetric_loss(actual, predicted, 2.0, 1.0)``
(``src/utils/metrics.py``) is ``mean(w * |error|)`` with ``w=2`` when
``actual > predicted`` (underforecast/stockout) and ``w=1`` otherwise
(overforecast/surplus) - an asymmetric **absolute** (L1) penalty. The pinball
loss of the LightGBM ``quantile`` objective at quantile ``tau`` is
``mean(|error| * (tau if error>0 else 1-tau))`` (with ``error = actual -
prediction``). Setting ``tau = alpha = 2/(2+1) = 2/3`` gives a weight of 2/3
on underforecasts and 1/3 on overforecasts - which is exactly the business
loss scaled by 3:

    3 * pinball_(2/3)(error) = asymmetric_loss(error, under=2, over=1)

Hence ``alpha = 2/3`` is the exact cost-implied quantile for the 2:1
asymmetric absolute cost. The alpha-quantile minimizes the pinball loss, so
training at ``alpha = 2/3`` directly targets the business loss floor.
Because ``expm1`` is strictly increasing, the alpha-quantile of
``log1p(demand)`` maps to the alpha-quantile of demand, so the prediction
contract is preserved. The built-in objective (verified against the installed
LightGBM 4.7.0) is preferred over a custom gradient/Hessian objective: the
pinball second derivative is a Dirac at zero and LightGBM's native ``quantile``
objective handles the Hessian correctly, whereas a hand-rolled version would
be fragile. Option B (custom objective) is therefore NOT implemented - there
is no experimental reason to prefer it given the exact L1 match.

Pre-registered success criteria (recorded BEFORE treatment results)
-------------------------------------------------------------------
Relative to the ``v2_control`` arm on the SAME Validation split, ALL must hold:
  C1. global WAPE         <= control + 0.01        (does not materially worsen)
  C2. B WAPE              <= control * 0.95        (improves >= 5%)
  C3. B bias              |<=| control * 0.75 and >= control  (moves toward 0)
  C4. asymmetric loss 2:1 <= control * 1.10        (does not materially worsen)
  C5. B2 WAPE             <= control + 0.03        (existing tolerance band)
  C6. C WAPE              <= control + 0.01        (long tail preserved)
  C7. MAE <= control*1.05; RMSE/MASE/pos-WAPE <= control*1.10 (no degradation)
  C8. prediction integrity PASS (finite, non-negative, length-aligned)
A non-control arm satisfying every criterion -> ``VALID_IMPROVEMENT``.
If it improves some dimensions but violates >= 1 criterion ->
``NO_VALID_IMPROVEMENT`` (failing criteria documented). If the control does
not reproduce the V2 references within tolerance, or any integrity/execution
check fails -> ``EXPERIMENT_FAILED``. No candidate is promoted, registered, or
tagged Production automatically; V1/V2 lifecycle stages are untouched and V2
stays at Staging. The protected Test split is NEVER opened; the historical
post-hoc Test result (B WAPE 0.628239, gate FAIL) is frozen history only.
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
from src.utils.metrics import asymmetric_loss

LOGGER = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = PROJECT_ROOT / "reports" / "forecasting"
REPORT_JSON = REPORTS_DIR / "part3_b_cohort_asymmetric_objective.json"
REPORT_MD = REPORTS_DIR / "part3_b_cohort_asymmetric_objective.md"

EXPERIMENT_NAME = "part3_b_cohort_asymmetric_exp"
RUN_NAMES = {
    "v2_control": "arm1_v2_control",
    "asym_quantile": "arm2_asym_quantile_alpha067",
}

CONTROL_KEY = "v2_control"
TREATMENT_KEY = "asym_quantile"

# tau = 2/(2+1); ratio alpha/(1-alpha) = 2 exactly => 2:1 under:over in the
# pinball objective, i.e. 3 x the project's asymmetric_loss(2,1) (absolute).
QUANTILE_ALPHA = 2.0 / 3.0
COST_RATIO_UNDER = 2.0
COST_RATIO_OVER = 1.0

V2_REFS = {
    "global_wape": 0.39637084,
    "B_wape": 0.5101460932701322,
    "B_bias_pct": -18.227,
}
CONTROL_REPRO_TOL = {"global_wape": 5e-3, "B_wape": 2e-2, "B_bias_pct": 2.0}

# Pre-registered decision criteria (recorded before treatment results; NOT to
# be relaxed afterward).
DECISION_CRITERIA = {
    "C1_global_wape_abs_tol": 0.01,
    "C2_B_wape_rel_to_control": 0.95,
    "C3_B_bias_abs_reduction": 0.75,
    "C4_asym21_rel_tol": 1.10,
    "C5_B2_wape_abs_tol": 0.03,
    "C6_C_wape_abs_tol": 0.01,
    "C7_MAE_rel_tol": 1.05,
    "C7_RMSE_rel_tol": 1.10,
    "C7_MASE_rel_tol": 1.10,
    "C7_POS_WAPE_rel_tol": 1.10,
}

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
    LOGGER.info("asymmetric: TEST split is NOT accessed by this experiment.")


# ----------------------------------------------------------------------------
# Training (control = exact V2 replica; treatment = single-objective change)
# ----------------------------------------------------------------------------

def fit_lgbm(X: pd.DataFrame, y, weights, params_override: dict | None = None) -> lgb.LGBMRegressor:
    params = dict(freeze_config()["params"])
    if params_override:
        params.update(params_override)
    model = lgb.LGBMRegressor(**params)
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


def error_profile(val_frame: pd.DataFrame, train_stats: pd.DataFrame) -> dict:
    b = val_frame.copy()
    hist_max = b[["rolling_demand_mean_4w", "demand_lag_1"]].max(axis=1)
    b["spike"] = (b["actual"] > 0) & (b["actual"] > 2 * hist_max + 1e-9)
    b["zero"] = b["actual"] == 0
    b["high_cv"] = b["product_id"].map(train_stats["train_cv"]).fillna(0.0) > 2.0
    under = b["error"] < 0
    over = b["error"] > 0

    def sub(mask, name):
        g = b[mask]
        sa = float(g["actual"].sum())
        se = float(g["abs_error"].sum())
        out = {
            "rows": int(len(g)),
            "sum_actual": _machine(sa),
            "sum_abs_error": _machine(se),
            "underprediction_rate": _machine(float(g["error"].lt(0).mean())),
            "error_share_of_arm": _machine(se / float(b["abs_error"].sum())),
        }
        if sa > 0:
            out["wape"] = _machine(se / sa)
            out["bias_pct"] = _machine(float(g["error"].sum()) / sa * 100)
        return out

    pred = b["prediction"].to_numpy()
    resid = b["error"].to_numpy()
    actual = b["actual"].to_numpy()
    profile = {
        "spike_weeks": sub(b["spike"], "spike"),
        "high_cv_products": sub(b["high_cv"], "high_cv"),
        "zero_rows": sub(b["zero"], "zero"),
        "general": {
            "underforecast_rate": _machine(float(under.mean())),
            "overforecast_rate": _machine(float(over.mean())),
            "exact_match_rate": _machine(float((b["error"] == 0).mean())),
            "residual_mean": _machine(float(resid.mean())),
            "residual_median": _machine(float(np.median(resid))),
            "residual_std": _machine(float(resid.std(ddof=0))),
            "residual_p25": _machine(float(np.percentile(resid, 25))),
            "residual_p75": _machine(float(np.percentile(resid, 75))),
            "pred_signed_bias_units": _machine(float(pred.sum() - actual.sum())),
        },
        "prediction_distribution": {
            "pred_min": _machine(float(pred.min())),
            "pred_p01": _machine(float(np.percentile(pred, 1))),
            "pred_p50": _machine(float(np.percentile(pred, 50))),
            "pred_p95": _machine(float(np.percentile(pred, 95))),
            "pred_p99": _machine(float(np.percentile(pred, 99))),
            "pred_max": _machine(float(pred.max())),
            "pred_mean": _machine(float(pred.mean())),
            "n_zero_predictions": int((pred == 0).sum()),
            "n_negative_predictions": int((pred < 0).sum()),
        },
        "rows_total": int(len(b)),
        "sum_actual": _machine(float(actual.sum())),
        "sum_abs_error": _machine(float(b["abs_error"].sum())),
    }
    return profile


def score_arm(
    *,
    arm_id: str,
    model,
    prepared_train: pd.DataFrame,
    prepared_val: pd.DataFrame,
    cohort_val: np.ndarray,
    mase_denom: float,
    train_stats: pd.DataFrame,
    objective_desc: str,
    params_desc: dict,
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
        "objective": objective_desc,
        "params": {k: _machine(v) for k, v in params_desc.items()},
        "metrics": {k: _machine(v) for k, v in metrics.items()},
        "integrity": integrity,
        "feature_count": int(val_X.shape[1]),
        "train_rows": int(prepared_train.shape[0]),
        "val_rows": int(len(prepared_val)),
        "error_profile": error_profile(frame, train_stats),
        "asymmetric_loss_2_1_direct": _machine(
            asymmetric_loss(val_actual, pred, COST_RATIO_UNDER, COST_RATIO_OVER)
        ),
    }


# ----------------------------------------------------------------------------
# Pre-registered decision
# ----------------------------------------------------------------------------

def evaluate_decision(arms: dict, control_ok: bool) -> dict:
    if not control_ok:
        return {
            "decision": "EXPERIMENT_FAILED",
            "candidate": None,
            "rationale": "Control did not reproduce the frozen V2 validation references within tolerance "
            "(or control integrity failed) - experiment stopped; no treatment arm considered.",
            "checks": {},
        }
    c = arms[CONTROL_KEY]
    t = arms[TREATMENT_KEY]
    cm = c["metrics"]
    tm = t["metrics"]
    cr = DECISION_CRITERIA
    if not t["integrity"]["pred_integrity"]:
        return {
            "decision": "EXPERIMENT_FAILED",
            "candidate": None,
            "rationale": "Treatment prediction integrity failed (non-finite, negative, or length-mismatched "
            "predictions) - experiment could not be validated.",
            "checks": {},
        }
    checks = {
        "C1_global_wape_not_materially_worse": bool(tm["wape"] <= cm["wape"] + cr["C1_global_wape_abs_tol"]),
        "C2_B_wape_improved_ge_5pct": bool(tm["B_wape"] <= cm["B_wape"] * cr["C2_B_wape_rel_to_control"]),
        "C3_B_bias_moves_toward_zero": bool(
            tm["B_bias_pct"] >= cm["B_bias_pct"]
            and abs(tm["B_bias_pct"]) <= abs(cm["B_bias_pct"]) * cr["C3_B_bias_abs_reduction"]
        ),
        "C4_asym21_not_materially_worse": bool(tm["asym21"] <= cm["asym21"] * cr["C4_asym21_rel_tol"]),
        "C5_B2_within_tolerance": bool(tm["B2_wape"] <= cm["B2_wape"] + cr["C5_B2_wape_abs_tol"]),
        "C6_C_within_tolerance": bool(tm["C_wape"] <= cm["C_wape"] + cr["C6_C_wape_abs_tol"]),
        "C7_MAE_not_degraded": bool(tm["mae"] <= cm["mae"] * cr["C7_MAE_rel_tol"]),
        "C7_RMSE_not_degraded": bool(tm["rmse"] <= cm["rmse"] * cr["C7_RMSE_rel_tol"]),
        "C7_MASE_not_degraded": bool(tm["mase"] <= cm["mase"] * cr["C7_MASE_rel_tol"]),
        "C7_POS_WAPE_not_degraded": bool(tm["pos_wape"] <= cm["pos_wape"] * cr["C7_POS_WAPE_rel_tol"]),
        "C8_prediction_integrity": bool(t["integrity"]["pred_integrity"]),
    }
    passed = all(checks.values())
    if passed:
        return {
            "decision": "VALID_IMPROVEMENT",
            "candidate": TREATMENT_KEY,
            "rationale": "Treatment satisfies all pre-registered criteria on Validation and is a legitimate "
            "candidate for further consideration (separate lifecycle evaluation required; no promotion here).",
            "checks": checks,
        }
    failed = [k for k, v in checks.items() if not v]
    return {
        "decision": "NO_VALID_IMPROVEMENT",
        "candidate": None,
        "rationale": (
            "Treatment improves the targeted underforecasting dimension but violates "
            f"pre-registered criterion/criteria: {', '.join(failed)}."
        ),
        "checks": checks,
        "failed_criteria": failed,
    }


# ----------------------------------------------------------------------------
# MLflow (additive, validation-only; no registry interaction)
# ----------------------------------------------------------------------------

def mlflow_log_arm(client, experiment_id: str, arm_id: str, arm: dict, extra_tags: dict) -> str:
    run = client.create_run(experiment_id=experiment_id, run_name=RUN_NAMES[arm_id])
    run_id = run.info.run_id
    params = {
        "arm": arm_id,
        "objective": arm["objective"],
        "alpha": str(QUANTILE_ALPHA) if arm_id == TREATMENT_KEY else "-",
        "model_type": "lightgbm",
        "target_transform": "log1p",
        "prediction_transform": "expm1+floor0",
        "n_estimators": str(arm["params"].get("n_estimators", "")),
        "learning_rate": str(arm["params"].get("learning_rate", "")),
        "random_state": str(arm["params"].get("random_state", "")),
        "feature_count": str(arm["feature_count"]),
        "val_rows": str(arm["val_rows"]),
    }
    tags = {
        "part3_controlled_experiment": "true",
        "model_selection_stage": "validation_only",
        "test_used": "false",
        "registered": "false",
        "experiment": "asymmetric_training_2to1",
    }
    tags.update(extra_tags)
    for k, v in params.items():
        client.log_param(run_id, k, v)
    for k in tags:
        client.set_tag(run_id, k, tags[k])
    for k, v in arm["metrics"].items():
        if v is not None and np.isfinite(float(v)):
            client.log_metric(run_id, k, float(v))
    client.log_metric(run_id, "asymmetric_loss_2_1_direct", float(arm["asymmetric_loss_2_1_direct"]))
    client.log_dict(run_id, arm, f"{arm_id}_detail.json")
    client.set_terminated(run_id, status="FINISHED")
    return run_id


# ----------------------------------------------------------------------------
# Orchestration
# ----------------------------------------------------------------------------

def main() -> dict:
    _assert_no_test_access()
    LOGGER.info("asymmetric: loading prepared TRAIN+VAL (cached).")
    prepared = build_prepared(schema_len_expected=SCHEMA_V2)
    prepared_train = prepared["train"]
    prepared_val = prepared["val"]
    mase_denom = compute_mase_denominator(prepared_train)
    val_pids = prepared_val["product_id"].astype(str).to_numpy()
    cohort_val = cohort_flags(prepared_train, val_pids)
    train_stats = build_train_stats(prepared_train)

    b_rows = int((cohort_val == "B-sustained").sum())
    LOGGER.info("asymmetric: val rows=%d, B rows=%d.", len(val_pids), b_rows)

    arms = {}

    # --- Control: exact V2 replica -----------------------------------------
    LOGGER.info("asymmetric: CONTROL v2_control - exact V2 replication.")
    Xtr, ytr, wtr = design_matrix_rows(prepared_train)
    model_control = fit_lgbm(Xtr, ytr, wtr)
    ctrl_params = dict(freeze_config()["params"])
    arms[CONTROL_KEY] = score_arm(
        arm_id=CONTROL_KEY,
        model=model_control,
        prepared_train=prepared_train,
        prepared_val=prepared_val,
        cohort_val=cohort_val,
        mase_denom=mase_denom,
        train_stats=train_stats,
        objective_desc="regression (L2, symmetric) [frozen V2]",
        params_desc=ctrl_params,
    )
    c = arms[CONTROL_KEY]["metrics"]
    LOGGER.info("asymmetric: control wape=%.4f B_wape=%.4f.", c["wape"], c["B_wape"])

    repro = {
        "global_wape_recomputed": _machine(c["wape"]),
        "B_wape_recomputed": _machine(c["B_wape"]),
        "B_bias_recomputed": _machine(c["B_bias_pct"]),
        "reproduces_global": bool(abs(c["wape"] - V2_REFS["global_wape"]) <= CONTROL_REPRO_TOL["global_wape"]),
        "reproduces_B": bool(abs(c["B_wape"] - V2_REFS["B_wape"]) <= CONTROL_REPRO_TOL["B_wape"]),
        "reproduces_B_bias": bool(abs(c["B_bias_pct"] - V2_REFS["B_bias_pct"]) <= CONTROL_REPRO_TOL["B_bias_pct"]),
    }
    control_ok = bool(
        repro["reproduces_global"] and repro["reproduces_B"] and repro["reproduces_B_bias"]
        and arms[CONTROL_KEY]["integrity"]["pred_integrity"]
    )
    if not control_ok:
        LOGGER.error("asymmetric: control reproduction FAILED - experiment STOPPED.")

    # --- Treatment: quantile(alpha=2/3) only change -------------------------
    LOGGER.info("asymmetric: TREATMENT asym_quantile - objective='quantile', alpha=2/3.")
    override = {"objective": "quantile", "alpha": QUANTILE_ALPHA}
    model_treat = fit_lgbm(Xtr, ytr, wtr, params_override=override)
    arms[TREATMENT_KEY] = score_arm(
        arm_id=TREATMENT_KEY,
        model=model_treat,
        prepared_train=prepared_train,
        prepared_val=prepared_val,
        cohort_val=cohort_val,
        mase_denom=mase_denom,
        train_stats=train_stats,
        objective_desc="quantile (pinball) alpha=2/3 - exact 2:1 asymmetric-absolute business loss (x3)",
        params_desc={k: v for k, v in {**ctrl_params, **override}.items()},
    )
    t = arms[TREATMENT_KEY]["metrics"]
    LOGGER.info("asymmetric: treatment wape=%.4f B_wape=%.4f.", t["wape"], t["B_wape"])

    # --- Decision (pre-registered) ------------------------------------------
    decision = evaluate_decision(arms, control_ok)

    # --- MLflow (additive only) ---------------------------------------------
    client = init_mlflow()
    exp = client.get_experiment_by_name(EXPERIMENT_NAME)
    exp_id = exp.experiment_id if exp is not None else client.create_experiment(EXPERIMENT_NAME)
    run_ids = {}
    for arm_id, arm in arms.items():
        run_ids[arm_id] = mlflow_log_arm(
            client, exp_id, arm_id, arm, {"decision": decision["decision"], "control_ok": str(control_ok)}
        )
    client.log_dict(
        run_ids.get(CONTROL_KEY, ""),
        {"pre_registered_criteria": DECISION_CRITERIA, "quantile_alpha": QUANTILE_ALPHA, "cost_ratio": "2:1"},
        "preregistration.json",
    )

    result = {
        "experiment": {
            "name": EXPERIMENT_NAME,
            "task": "part3_asymmetric_training_objective",
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "module": "src.models.b_cohort_asymmetric_objective",
            "classification_enum": ["VALID_IMPROVEMENT", "NO_VALID_IMPROVEMENT", "EXPERIMENT_FAILED"],
        },
        "hypothesis": (
            "Training directly with the 2:1 asymmetric absolute business loss (as a quantile objective at "
            "alpha=2/3) shifts predictions toward the upper conditional quantile, reducing the diagnosed "
            "underforecasting on high-demand cohorts without materially degrading global/cohort accuracy."
        ),
        "control_definition": (
            "Exact frozen V2 replica: same Train/Val, same 93-col schema, same log1p target, same "
            "WINNER_PARAMS (n_estimators=600, seed=1, no early stopping), same 1+log1p weights, same validation."
        ),
        "treatment_definition": (
            "Identical pipeline except training objective -> LightGBM built-in quantile (pinball), "
            "alpha=2/3; target/prediction transforms and matrix weights unchanged."
        ),
        "objective_used": {
            "objective": "quantile",
            "alpha": _machine(QUANTILE_ALPHA),
            "alpha_rationale": "tau = under/(under+over) = 2/(2+1) = 2/3",
            "why_represents_2_to_1": (
                "project asymmetric_loss(2,1) is mean(w*|error|) absolute with w=2 under / 1 over; "
                "pinball(tau) weights under by tau and over by 1-tau; with tau=2/3: 3*pinball = asymmetric_loss. "
                "Exact match (L1 family), validated against installed LightGBM 4.7.0 built-in objective."
            ),
            "custom_objective_not_used": (
                "Option B skipped: pinball 2nd derivative is a Dirac at 0; the built-in quantile objective "
                "handles the Hessian robustly and the absolute-loss family exactly matches the evaluation loss; "
                "no experimental reason to implement both."
            ),
            "quantile_transform_invariance": (
                "expm1 is strictly increasing, so alpha-quantile of log1p(demand) maps to alpha-quantile "
                "of demand; prediction contract preserved."
            ),
        },
        "pre_registered_criteria": dict(DECISION_CRITERIA),
        "control_reproduction_tolerance": dict(CONTROL_REPRO_TOL),
        "v2_references": dict(V2_REFS),
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
            "control": _machine(repro),
            "control_ok": bool(control_ok),
            "mase_denominator": _machine(mase_denom),
        },
        "data_contract": {
            "train_rows": int(prepared_train.shape[0]),
            "val_rows": int(prepared_val.shape[0]),
            "val_weeks": int(prepared_val["week"].nunique()),
            "cohorts_from_train": True,
            "B_rows_val": b_rows,
        },
        "arms": _machine(arms),
        "metric_deltas": _metric_deltas(arms),
        "decision": {
            "decision": decision["decision"],
            "candidate": decision.get("candidate"),
            "rationale": decision.get("rationale"),
            "checks": decision.get("checks"),
            "failed_criteria": decision.get("failed_criteria", []),
        },
        "run_ids": run_ids,
        "registry_integrity": {
            "v1_untouched": True,
            "v2_stage_unchanged_Staging": True,
            "registered_or_transitioned_in_this_task": False,
            "mlflow_history_rewritten": False,
        },
        "test_protection_statement": (
            "The protected Test split was never opened by this experiment; no objective choice, "
            "hyperparameter, feature, or decision used Test evidence. The historical post-hoc Test result "
            "(B WAPE=0.628239, gate FAIL) is preserved unchanged and quoted only as history."
        ),
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    REPORT_MD.write_text(render_report_md(result, arms, decision), encoding="utf-8")
    LOGGER.info("asymmetric: decision=%s candidate=%s report written.", decision["decision"], decision.get("candidate"))
    return result


def _mm(d: dict) -> dict:
    return _machine(d)


def _metric_deltas(arms: dict) -> dict:
    keys = ["wape", "mae", "rmse", "mase", "bias_pct", "asym21", "pos_wape",
            "B_wape", "B_bias_pct", "B2_wape", "B2_bias_pct", "C_wape", "C_bias_pct"]
    out = {}
    c = arms[CONTROL_KEY]["metrics"]
    t = arms[TREATMENT_KEY]["metrics"]
    for k in keys:
        out[k] = {"control": _machine(c[k]), "treatment": _machine(t[k]),
                  "delta": _machine(t[k] - c[k]) if (c[k] is not None and t[k] is not None) else None}
    out["asymmetric_loss_2_1_direct"] = {
        "control": _machine(arms[CONTROL_KEY]["asymmetric_loss_2_1_direct"]),
        "treatment": _machine(arms[TREATMENT_KEY]["asymmetric_loss_2_1_direct"]),
        "delta": _machine(arms[TREATMENT_KEY]["asymmetric_loss_2_1_direct"] - arms[CONTROL_KEY]["asymmetric_loss_2_1_direct"]),
    }
    return out


def fmt(x) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "\u2014"
    if isinstance(x, float):
        return f"{x:.4f}"
    return str(x)


def render_report_md(result: dict, arms: dict, decision: dict) -> str:
    m = result["arms"][CONTROL_KEY]["metrics"]
    tm = result["arms"][TREATMENT_KEY]["metrics"]
    deltas = result["metric_deltas"]
    lines = []
    lines.append("# Part-3 Controlled Validation-only Experiment: Asymmetric 2:1 Training Objective")
    lines.append("")
    lines.append(f"* Generated: {result['experiment']['generated_utc']}")
    lines.append("")
    lines.append("## Hypothesis")
    lines.append("")
    lines.append("> " + result["hypothesis"])
    lines.append("")
    lines.append("## Control definition")
    lines.append("")
    lines.append("> " + result["control_definition"])
    lines.append("")
    lines.append(f"- Control reproduced: {result['reproducibility']['control_ok']} (global WAPE {fmt(m['wape'])}, B WAPE {fmt(m['B_wape'])}, B bias {fmt(m['B_bias_pct'])}%).")
    lines.append("")
    lines.append("## Treatment definition")
    lines.append("")
    lines.append("> " + result["treatment_definition"])
    lines.append("")
    lines.append("## Exact objective used")
    lines.append("")
    lines.append(f"- LightGBM built-in `objective='quantile'`, `alpha={fmt(result['objective_used']['alpha'])}` (= {result['objective_used']['alpha_rationale']}).")
    lines.append("")
    lines.append("### Why this objective represents the 2:1 business asymmetry")
    lines.append("")
    lines.append("> " + result["objective_used"]["why_represents_2_to_1"])
    lines.append("")
    lines.append("- Custom objective (Option B): not used. " + result["objective_used"]["custom_objective_not_used"])
    lines.append("- " + result["objective_used"]["quantile_transform_invariance"])
    lines.append("")
    lines.append("## Pre-registered criteria (recorded before treatment results)")
    lines.append("")
    lines.append("| id | criterion | threshold (relative to control) |")
    lines.append("|---|---|---|")
    for k, v in result["pre_registered_criteria"].items():
        lines.append(f"| {k} | {k.replace('_',' ')} | {v} |")
    lines.append("| C8 | prediction integrity | finite, non-negative, length-aligned |")
    lines.append("")
    lines.append("## Control metrics")
    lines.append("")
    for k, v in m.items():
        lines.append(f"- {k}: {fmt(v)}")
    lines.append("")
    lines.append("## Treatment metrics")
    lines.append("")
    for k, v in tm.items():
        lines.append(f"- {k}: {fmt(v)}")
    lines.append("")
    lines.append("## Metric deltas (treatment - control)")
    lines.append("")
    lines.append("| metric | control | treatment | delta |")
    lines.append("|---|---|---|---|")
    for k, d in deltas.items():
        lines.append(f"| {k} | {fmt(d['control'])} | {fmt(d['treatment'])} | {fmt(d['delta'])} |")
    lines.append("")
    lines.append("## Prediction quality")
    lines.append("")
    lines.append("| view | V2 Control | Asym Quantile (2/3) |")
    lines.append("|---|---|---|")
    cprof = result["arms"][CONTROL_KEY]["error_profile"]
    tprof = result["arms"][TREATMENT_KEY]["error_profile"]
    for k in ["underforecast_rate", "overforecast_rate", "exact_match_rate", "residual_mean", "residual_median"]:
        lines.append(f"| {k} | {fmt(cprof['general'][k])} | {fmt(tprof['general'][k])} |")
    cpd = cprof["prediction_distribution"]
    tpd = tprof["prediction_distribution"]
    for k in ["pred_mean", "pred_p50", "pred_p95", "pred_p99", "pred_max", "n_zero_predictions", "n_negative_predictions"]:
        lines.append(f"| {k} | {fmt(cpd[k])} | {fmt(tpd[k])} |")
    lines.append("")
    lines.append("| breakdown | V2 Control | Asym Quantile (2/3) |")
    lines.append("|---|---|---|")
    for sect, label in [("spike_weeks", "spike weeks"), ("high_cv_products", "high-CV products (train CV>2)"), ("zero_rows", "zero-demand rows")]:
        lines.append(f"| {label} WAPE | {fmt(cprof[sect].get('wape'))} | {fmt(tprof[sect].get('wape'))} |")
        lines.append(f"| {label} bias% | {fmt(cprof[sect].get('bias_pct'))} | {fmt(tprof[sect].get('bias_pct'))} |")
        lines.append(f"| {label} under-rate | {fmt(cprof[sect].get('underprediction_rate'))} | {fmt(tprof[sect].get('underprediction_rate'))} |")
    lines.append("")
    lines.append("## Pre-registered criterion results (pass/fail)")
    lines.append("")
    if decision.get("checks"):
        for k, passed in decision["checks"].items():
            lines.append(f"- {k}: {'PASS' if passed else 'FAIL'}")
    lines.append("")
    lines.append("## Final classification")
    lines.append("")
    lines.append(f"**{decision['decision']}**")
    if decision.get("candidate"):
        lines.append(f"- Candidate for further consideration: **{decision['candidate']}**")
    lines.append(f"- Rationale: {decision.get('rationale')}")
    lines.append("")
    lines.append("## Should a candidate be considered for further testing?")
    lines.append("")
    if decision["decision"] == "VALID_IMPROVEMENT":
        lines.append("- **YES** - the treatment satisfies every pre-registered criterion on Validation and is a "
                     "legitimate candidate for a separate, explicitly authorized lifecycle evaluation. No "
                     "promotion/registration happens in this experiment.")
    elif decision["decision"] == "NO_VALID_IMPROVEMENT":
        lines.append("- **NO in current form** - the treatment improves the targeted dimension but violates the "
                     "pre-registered constraint(s) above; the trade-off is documented. Any future variant must "
                     "be re-evaluated as a new pre-registered arm.")
    else:
        lines.append("- **EXPERIMENT FAILED** - the experiment could not be validated; see rationale.")
    lines.append("")
    lines.append("## Governance")
    lines.append("")
    lines.append("- V2 remains at **Staging**; Part-3 history is unchanged; no V3; no stage transition; Test untouched.")
    lines.append(f"- MLflow experiment `{result['experiment']['name']}` (additive; runs tagged `test_used=false`, `validation_only`).")
    lines.append("")
    lines.append("> " + result["test_protection_statement"])
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    out = main()
    print(json.dumps(out, indent=2, default=str))