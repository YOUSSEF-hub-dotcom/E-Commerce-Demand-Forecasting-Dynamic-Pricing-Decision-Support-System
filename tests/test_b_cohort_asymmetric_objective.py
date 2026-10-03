"""Part-3 asymmetric 2:1 training-objective experiment guards (Train + Validation only).

Encodes the authorized controlled experiment contract established by
``src/models/b_cohort_asymmetric_objective.py``:

* Validation-only selection: the module never reads ``data/splits/test.csv``,
  never predicts on Test, never registers a model, never transitions stages,
  and never creates V3 (V1/V2 lifecycle and the historical post-hoc B WAPE
  =0.628239 record are preserved untouched). Note: the LATER formal
  gate-policy revision (``part3_v2_gate_policy_revision``) separately
  authorized V2's promotion to Production under the revised policy -- this
  experiment itself never transitioned anything.
* The ``v2_control`` arm is the exact frozen V2 replica (93-col schema,
  ``log1p`` target, ``WINNER_PARAMS`` with ``n_estimators=600`` / seed 1 / no
  early stopping, ``1+log1p`` matrix weights) and must reproduce the frozen V2
  Validation references (global WAPE 0.39637084, B WAPE 0.5101460932701322,
  B bias -18.227%) within the pre-registered tolerance; otherwise the
  experiment is ``EXPERIMENT_FAILED``.
* The treatment arm changes ONLY the training objective: LightGBM built-in
  ``objective='quantile'`` with ``alpha = 2/(2+1) = 2/3``, which - because the
  project evaluation loss ``asymmetric_loss(_,_,2,1)`` is an *absolute* (L1)
  weighted penalty - is exactly the cost-implied pinball quantile
  (``3 * pinball_(2/3) = asymmetric_loss(2,1)`` elementwise).
* The pre-registered 8-criterion decision rules are frozen in
  ``DECISION_CRITERIA`` and drive the single-verdict classification
  ``VALID_IMPROVEMENT | NO_VALID_IMPROVEMENT | EXPERIMENT_FAILED``.
* MLflow receives exactly two additive runs in
  ``part3_b_cohort_asymmetric_exp`` (``arm1_v2_control``,
  ``arm2_asym_quantile_alpha067``) tagged ``test_used=false`` and
  ``validation_only``; the official lifecycle experiment and registry stay
  untouched.

Read-only: JSON/document reads, source inspection, small in-memory LightGBM /
loss identity checks, and read-only MLflow queries. No training on real data,
no registration, no file rewriting.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import src.models.b_cohort_asymmetric_objective as bc
import src.models.final_forecast as ff

ROOT = Path(__file__).resolve().parents[1]

REPORT_JSON = ROOT / "reports" / "forecasting" / "part3_b_cohort_asymmetric_objective.json"
REPORT_MD = ROOT / "reports" / "forecasting" / "part3_b_cohort_asymmetric_objective.md"
MODULE = ROOT / "src" / "models" / "b_cohort_asymmetric_objective.py"

VAL_GLOBAL_WAPE_REF = 0.39637084
VAL_B_WAPE_REF = 0.5101460932701322
VAL_B_BIAS_REF = -18.227
TEST_B_WAPE_RECORD = 0.6282389722943832
MASE_DENOM_REF = 0.03804341247443925


def read_report() -> dict:
    return json.loads(REPORT_JSON.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# 1. Control reproduction (frozen V2 validation references)
# --------------------------------------------------------------------------- #

def test_report_exists():
    assert REPORT_JSON.is_file() and REPORT_MD.is_file()
    json.loads(REPORT_JSON.read_text(encoding="utf-8"))


def test_control_reproduces_frozen_v2_validation_references():
    report = read_report()
    rep = report["reproducibility"]
    ctrl = report["arms"]["v2_control"]["metrics"]
    assert rep["control_ok"] is True
    assert rep["control"]["reproduces_global"] is True
    assert rep["control"]["reproduces_B"] is True
    assert rep["control"]["reproduces_B_bias"] is True
    assert abs(rep["control"]["global_wape_recomputed"] - VAL_GLOBAL_WAPE_REF) <= 5e-3
    assert abs(rep["control"]["B_wape_recomputed"] - VAL_B_WAPE_REF) <= 2e-2
    assert abs(rep["control"]["B_bias_recomputed"] - VAL_B_BIAS_REF) <= 2.0
    assert abs(ctrl["wape"] - VAL_GLOBAL_WAPE_REF) <= 1e-9
    assert abs(ctrl["B_wape"] - VAL_B_WAPE_REF) <= 1e-9
    assert abs(rep["mase_denominator"] - MASE_DENOM_REF) <= 1e-9


def test_control_and_treatment_share_data_contract():
    report = read_report()
    dc = report["data_contract"]
    assert dc["train_rows"] == 2_428_492
    assert dc["val_rows"] == 520_391
    assert dc["val_weeks"] == 17
    assert dc["cohorts_from_train"] is True
    assert dc["B_rows_val"] == 589
    for arm_id in ("v2_control", "asym_quantile"):
        arm = report["arms"][arm_id]
        assert arm["val_rows"] == 520_391
        assert arm["feature_count"] == 93


# --------------------------------------------------------------------------- #
# 2. Asymmetric objective correctness (the 2:1 -> alpha=2/3 mapping)
# --------------------------------------------------------------------------- #

def test_alpha_is_the_two_to_one_implied_quantile():
    assert bc.COST_RATIO_UNDER == 2.0 and bc.COST_RATIO_OVER == 1.0
    assert bc.QUANTILE_ALPHA == pytest.approx(2.0 / (2.0 + 1.0))
    # under/over weight ratio inside the pinball objective is exactly 2
    assert bc.QUANTILE_ALPHA / (1.0 - bc.QUANTILE_ALPHA) == pytest.approx(2.0)


def test_pinball_at_two_thirds_equals_scaled_asymmetric_loss():
    from src.utils.metrics import asymmetric_loss

    actual = np.array([0.0, 1.0, 3.0, 0.5, 8.0, 0.0])
    pred = np.array([0.2, 0.7, 4.0, 0.6, 5.0, 0.1])
    e = actual - pred  # positive e => underforecast
    tau = bc.QUANTILE_ALPHA
    pinball = np.mean(np.abs(e) * np.where(e > 0, tau, 1.0 - tau))
    asym = asymmetric_loss(actual, pred, under_weight=2.0, over_weight=1.0)
    assert asym == pytest.approx(3.0 * pinball)
    # and the recorded metrics agree with the direct business loss
    report = read_report()
    for arm_id in ("v2_control", "asym_quantile"):
        arm = report["arms"][arm_id]
        assert arm["metrics"]["asym21"] == pytest.approx(arm["asymmetric_loss_2_1_direct"], abs=1e-12)


def test_builtin_quantile_objective_accepted_by_installed_lightgbm():
    import lightgbm as lgb

    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 5))
    y = 2.0 * X[:, 0] + rng.normal(size=300)
    params = dict(ff.WINNER_PARAMS)
    params.update({"objective": "quantile", "alpha": bc.QUANTILE_ALPHA, "n_estimators": 10})
    model = lgb.LGBMRegressor(**params).fit(X, y, sample_weight=np.ones(300))
    assert model.objective_ == "quantile"
    p = model.predict(X[:5])
    assert np.isfinite(p).all() and p.shape == (5,)


def test_quantile_is_monotone_upper_shift_for_higher_alpha():
    import lightgbm as lgb

    rng = np.random.default_rng(1)
    X = rng.normal(size=(300, 5))
    y = 3.0 * X[:, 1] + rng.normal(size=300)
    base = dict(ff.WINNER_PARAMS)
    lo = lgb.LGBMRegressor(**{**base, "objective": "quantile", "alpha": 0.1, "n_estimators": 10}).fit(X, y, sample_weight=np.ones(300))
    hi = lgb.LGBMRegressor(**{**base, "objective": "quantile", "alpha": 0.9, "n_estimators": 10}).fit(X, y, sample_weight=np.ones(300))
    assert (hi.predict(X) >= lo.predict(X)).all()


# --------------------------------------------------------------------------- #
# 3. Treatment is a single isolated intervention (objective only)
# --------------------------------------------------------------------------- #

def test_treatment_changes_only_the_objective():
    report = read_report()
    ctrl = report["arms"]["v2_control"]
    treat = report["arms"]["asym_quantile"]
    ctrl_params = dict(ctrl["params"])
    treat_params = dict(treat["params"])
    assert "objective" not in ctrl_params and "alpha" not in ctrl_params
    objective = treat_params.pop("objective")
    alpha = treat_params.pop("alpha")
    assert objective == "quantile"
    assert alpha == pytest.approx(2.0 / 3.0)
    assert treat_params == ctrl_params, "treatment must ONLY change the objective"
    assert "regression (L2, symmetric)" in ctrl["objective"]
    assert "quantile" in treat["objective"]


# --------------------------------------------------------------------------- #
# 4. Prediction integrity
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("arm_id", ["v2_control", "asym_quantile"])
def test_prediction_integrity_for_all_arms(arm_id):
    report = read_report()
    arm = report["arms"][arm_id]
    assert arm["integrity"]["pred_integrity"] is True
    assert arm["integrity"]["n_neg"] == 0
    assert arm["integrity"]["n_nan_inf"] == 0
    assert arm["error_profile"]["prediction_distribution"]["n_negative_predictions"] == 0


# --------------------------------------------------------------------------- #
# 5. No Test access
# --------------------------------------------------------------------------- #

def test_module_never_reads_test_split():
    src = MODULE.read_text(encoding="utf-8")
    assert "read_csv" not in src and "read_parquet" not in src
    assert "test.csv" not in src
    assert "splits/test" not in src
    assert ".astype(\"float64\")" in src


def test_module_has_no_cli_or_force_flag():
    src = MODULE.read_text(encoding="utf-8")
    assert "argparse" not in src
    assert "add_argument" not in src
    assert "sys.argv" not in src


# --------------------------------------------------------------------------- #
# 6. Metric calculation + cohort metrics
# --------------------------------------------------------------------------- #

REQUIRED_METRICS = {
    "wape", "mae", "rmse", "mase", "bias_pct", "asym21", "pos_wape",
    "B_wape", "B_bias_pct", "B2_wape", "B2_bias_pct", "C_wape", "C_bias_pct",
}


def test_full_metric_suite_recorded_for_all_arms():
    report = read_report()
    for arm_id in ("v2_control", "asym_quantile"):
        metrics = report["arms"][arm_id]["metrics"]
        assert REQUIRED_METRICS <= set(metrics)


def test_cohort_metrics_present_and_finite():
    report = read_report()
    for arm_id in ("v2_control", "asym_quantile"):
        m = report["arms"][arm_id]["metrics"]
        for k in ("B_wape", "B_bias_pct", "B2_wape", "B2_bias_pct", "C_wape", "C_bias_pct"):
            assert m[k] is not None and np.isfinite(m[k]), f"{arm_id} {k}"


def test_metric_deltas_are_arithmetically_consistent():
    report = read_report()
    c = report["arms"]["v2_control"]["metrics"]
    t = report["arms"]["asym_quantile"]["metrics"]
    for k, d in report["metric_deltas"].items():
        if k == "asymmetric_loss_2_1_direct":
            assert d["delta"] == pytest.approx(d["treatment"] - d["control"], abs=1e-12)
        else:
            assert d["control"] == pytest.approx(c[k], abs=1e-12)
            assert d["treatment"] == pytest.approx(t[k], abs=1e-12)


def test_error_profile_reports_under_over_and_distribution():
    report = read_report()
    for arm_id in ("v2_control", "asym_quantile"):
        g = report["arms"][arm_id]["error_profile"]["general"]
        d = report["arms"][arm_id]["error_profile"]["prediction_distribution"]
        for k in ("underforecast_rate", "overforecast_rate", "residual_mean", "residual_median"):
            assert k in g and np.isfinite(g[k])
        for k in ("pred_min", "pred_p50", "pred_p99", "pred_max", "n_zero_predictions", "n_negative_predictions"):
            assert k in d


# --------------------------------------------------------------------------- #
# 7. Pre-registered criteria + decision classification
# --------------------------------------------------------------------------- #

def test_preregistered_criteria_matches_module_constants():
    report = read_report()
    assert report["pre_registered_criteria"] == bc.DECISION_CRITERIA
    assert report["objective_used"]["alpha"] == pytest.approx(2.0 / 3.0)


def test_decision_is_from_the_allowed_enum():
    report = read_report()
    assert report["decision"]["decision"] in {"VALID_IMPROVEMENT", "NO_VALID_IMPROVEMENT", "EXPERIMENT_FAILED"}
    assert set(report["experiment"]["classification_enum"]) == {
        "VALID_IMPROVEMENT", "NO_VALID_IMPROVEMENT", "EXPERIMENT_FAILED"
    }


def test_decision_checks_cover_c1_to_c8():
    report = read_report()
    checks = report["decision"]["checks"]
    assert set(checks) == {
        "C1_global_wape_not_materially_worse",
        "C2_B_wape_improved_ge_5pct",
        "C3_B_bias_moves_toward_zero",
        "C4_asym21_not_materially_worse",
        "C5_B2_within_tolerance",
        "C6_C_within_tolerance",
        "C7_MAE_not_degraded",
        "C7_RMSE_not_degraded",
        "C7_MASE_not_degraded",
        "C7_POS_WAPE_not_degraded",
        "C8_prediction_integrity",
    }
    fails = report["decision"].get("failed_criteria", [])
    assert isinstance(fails, list)
    for c, passed in checks.items():
        if c in fails:
            assert passed is False
        elif c not in fails:
            assert passed is True


def test_recorded_decision_matches_observed_metrics():
    report = read_report()
    c = report["arms"]["v2_control"]["metrics"]
    t = report["arms"]["asym_quantile"]["metrics"]
    cr = bc.DECISION_CRITERIA
    # Honest verdict: treatment improves global/bias/asym/C but fails C2 and C5.
    assert t["wape"] <= c["wape"] + cr["C1_global_wape_abs_tol"]
    assert t["B_wape"] > c["B_wape"] * cr["C2_B_wape_rel_to_control"]  # C2 FAIL
    assert abs(t["B_bias_pct"]) <= abs(c["B_bias_pct"]) * cr["C3_B_bias_abs_reduction"]
    assert t["asym21"] <= c["asym21"] * cr["C4_asym21_rel_tol"]
    assert t["B2_wape"] > c["B2_wape"] + cr["C5_B2_wape_abs_tol"]  # C5 FAIL
    assert t["C_wape"] <= c["C_wape"] + cr["C6_C_wape_abs_tol"]
    assert t["mae"] <= c["mae"] * cr["C7_MAE_rel_tol"]
    assert report["decision"]["decision"] == "NO_VALID_IMPROVEMENT"
    assert report["decision"]["candidate"] is None
    assert set(report["decision"]["failed_criteria"]) == {
        "C2_B_wape_improved_ge_5pct",
        "C5_B2_within_tolerance",
    }


# --------------------------------------------------------------------------- #
# 8. V1/V2 lifecycle untouched + experiment reproducibility
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def client():
    from src.models.registry import init_mlflow

    return init_mlflow()


def test_module_has_no_registration_or_stage_transition_paths():
    src = MODULE.read_text(encoding="utf-8")
    assert "register_model" not in src
    assert "transition_model_version_stage" not in src
    assert "log_model" not in src


def test_report_registry_integrity_flags():
    report = read_report()
    ri = report["registry_integrity"]
    assert ri["v1_untouched"] is True
    assert ri["v2_stage_unchanged_Staging"] is True
    assert ri["registered_or_transitioned_in_this_task"] is False
    assert ri["mlflow_history_rewritten"] is False


def test_v2_historical_artifacts_untouched():
    ph = json.loads((ROOT / "reports/forecasting/part3_v2_posthoc_holdout_evaluation.json").read_text(encoding="utf-8"))
    assert ph["posthoc_holdout_metrics"]["B_wape"] == TEST_B_WAPE_RECORD
    assert ph["quality_gate"]["passed"] is False
    report = read_report()
    hist = report["baseline_v2_history"]
    assert abs(hist["final_posthoc_test_B_wape"] - 0.628239) <= 1e-6
    assert hist["v2_stage"] == "Staging"


def test_live_registry_unchanged(client):
    versions = client.search_model_versions("name='demand_forecasting_final'")
    assert sorted(int(v.version) for v in versions) == [1, 2]
    v1 = client.get_model_version("demand_forecasting_final", "1")
    v2 = client.get_model_version("demand_forecasting_final", "2")
    assert v1.current_stage == "Staging" and v1.run_id == "7fc871986e9240b1a4d98f7a9c1f893f"
    # V2 stands in Production per the LATER authorized formal gate-policy
    # revision (B_wape reclassified to diagnostic); this experiment itself
    # never transitioned - same run_id, no V3.
    assert v2.current_stage == "Production" and v2.run_id == "5d729a8594dc488bbd18ba0c4e216f5a"
    assert v2.tags.get("posthoc_holdout_evaluated") == "true"


def test_live_experiment_runs_additive_and_tagged(client):
    exp = client.get_experiment_by_name(bc.EXPERIMENT_NAME)
    assert exp is not None
    runs = client.search_runs([exp.experiment_id], order_by=["attributes.start_time"])
    assert len(runs) == 2
    by_name = {r.data.tags.get("mlflow.runName"): r for r in runs}
    assert set(by_name) == {"arm1_v2_control", "arm2_asym_quantile_alpha067"}
    report = read_report()
    for arm_id, run_id in report["run_ids"].items():
        run = client.get_run(run_id)
        assert run.info.status == "FINISHED"
        assert run.data.tags.get("mlflow.runName") == bc.RUN_NAMES[arm_id]
        assert run.data.tags.get("test_used") == "false"
        assert run.data.tags.get("model_selection_stage") == "validation_only"
        assert run.data.tags.get("registered") == "false"
        assert run.data.tags.get("experiment") == "asymmetric_training_2to1"
        if arm_id == "asym_quantile":
            assert "quantile" in run.data.params.get("objective", "")
            assert float(run.data.params.get("alpha")) == pytest.approx(2.0 / 3.0)


def test_no_remediation_noise_in_lifecycle_experiment(client):
    life = client.get_experiment_by_name("part3_final_lifecycle")
    for r in client.search_runs([life.experiment_id], order_by=["attributes.start_time"]):
        if r.data.tags.get("part3_controlled_experiment") == "true":
            pytest.fail(f"controlled-experiment run leaked into lifecycle: {r.info.run_id}")


# --------------------------------------------------------------------------- #
# 9. Report content requirements
# --------------------------------------------------------------------------- #

def test_md_report_contains_all_mandatory_sections():
    md = REPORT_MD.read_text(encoding="utf-8")
    for heading in [
        "## Hypothesis",
        "## Control definition",
        "## Treatment definition",
        "## Exact objective used",
        "### Why this objective represents the 2:1 business asymmetry",
        "## Pre-registered criteria",
        "## Control metrics",
        "## Treatment metrics",
        "## Metric deltas",
        "## Final classification",
        "## Should a candidate be considered for further testing",
    ]:
        assert heading in md, f"missing {heading}"


def test_md_reports_pass_fail_for_every_criterion():
    md = REPORT_MD.read_text(encoding="utf-8")
    report = read_report()
    for c in report["decision"]["checks"]:
        assert c in md


def test_json_embeds_test_protection_statement():
    report = read_report()
    assert "The protected Test split was never opened by this experiment" in report["test_protection_statement"]


def test_report_embeds_hypothesis_and_objective_rationale():
    report = read_report()
    assert report["hypothesis"]
    assert report["objective_used"]["alpha_rationale"] == "tau = under/(under+over) = 2/(2+1) = 2/3"
    assert "3*pinball = asymmetric_loss" in report["objective_used"]["why_represents_2_to_1"]