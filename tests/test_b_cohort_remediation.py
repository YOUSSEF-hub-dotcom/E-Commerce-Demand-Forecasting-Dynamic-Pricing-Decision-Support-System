"""Part-3 controlled B-cohort remediation experiment guards (Train + Validation only).

Encodes the authorized Part-3 controlled-experiment contract established by
``src/models/b_cohort_remediation.py``:

* The experiment is *validation-only selection*: it never reads
  ``data/splits/test.csv``, never predicts on Test, never registers a model and
  never makes a stage transition (V1 / V2 lifecycle history untouched; no V3).
  Note: the LATER formal gate-policy revision
  (``part3_v2_gate_policy_revision``) separately authorized V2's promotion to
  Production under the revised policy -- the experimental tasks themselves
  never transitioned anything.
* The ``v2_control`` arm is an exact replica of the frozen V2 validation
  contract: 93-column schema, ``1+log1p`` matrix weights, ``WINNER_PARAMS``,
  ``n_estimators=600``, ``random_state=1``, no early stopping - and must
  reproduce the frozen V2 validation references (global WAPE 0.3964, B WAPE
  0.5101460932701322, B bias -18.227%).
* Arm 2 (WAPE-aligned weights) is a single pre-registered deterministic,
  train-only formula; Arm 3 (spike features) adds exactly four *strictly
  past-only* context features whose values for row ``t`` depend only on rows
  ``< t``.
* Cohort definitions are the official ``cohort_flags`` / B / B2 constants;
  the decision is produced ONLY by the pre-registered criteria on Validation,
  with the outcome restricted to ``NO_VALID_IMPROVEMENT | CANDIDATE_IMPROVED |
  STOPPED``; the frozen historical post-hoc Test result (B WAPE
  0.628239, gate FAIL) is preserved unchanged.
* MLflow receives three additive runs in the dedicated
  ``part3_b_cohort_remediation_exp`` experiment
  (``arm1_v2_control``, ``arm2_wape_weights``, ``arm3_spike_features``)
  tagged ``test_used=false`` / ``model_selection_stage=validation_only``.

These tests are read-only: JSON/document reads, source inspections, toy-frame
unit checks on the pure feature/weight builders, and read-only MLflow queries.
They never train, register, log runs, or rewrite files.
"""

import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import src.models.b_cohort_remediation as bcm
import src.models.final_forecast as ff

ROOT = Path(__file__).resolve().parents[1]

REPORT_JSON = ROOT / "reports" / "forecasting" / "part3_b_cohort_remediation_experiments.json"
REPORT_MD = ROOT / "reports" / "forecasting" / "part3_b_cohort_remediation_experiments.md"
MODULE = ROOT / "src" / "models" / "b_cohort_remediation.py"
POSTHOC_JSON = ROOT / "reports" / "forecasting" / "part3_v2_posthoc_holdout_evaluation.json"

VAL_GLOBAL_WAPE_REF = 0.3964
VAL_B_WAPE_REF = 0.5101460932701322
VAL_B_BIAS_REF = -18.227
TEST_B_WAPE_RECORD = 0.6282389722943832


def read_report() -> dict:
    return json.loads(REPORT_JSON.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# 1. Test split never accessed
# --------------------------------------------------------------------------- #

def test_module_never_reads_test_split():
    src = MODULE.read_text(encoding="utf-8")
    assert "read_csv" not in src and "read_parquet" not in src
    # The module never references the literal test-split path at all.
    assert "test.csv" not in src
    assert "splits/test" not in src
    assert ".astype(\"float64\")" in src


def test_module_has_no_cli_and_no_force_override():
    src = MODULE.read_text(encoding="utf-8")
    assert "argparse" not in src
    assert "add_argument" not in src
    assert "sys.argv" not in src


# --------------------------------------------------------------------------- #
# 2. V2 frozen schema unchanged for the control arm
# --------------------------------------------------------------------------- #

def test_control_arm_uses_frozen_93_column_schema():
    assert bcm.SCHEMA_V2 == 93
    report = read_report()
    control = report["arms"]["v2_control"]
    assert control["feature_count"] == 93
    assert control["feature_diff_vs_v2"] == []
    assert "1+log1p" in control["weights"]


def test_only_arm3_gains_exactly_the_four_spike_features():
    report = read_report()
    a3 = report["arms"]["spike_features"]
    assert a3["feature_count"] == 93 + len(bcm.SPIKE_FEATURE_COLS)
    assert a3["feature_diff_vs_v2"] == list(bcm.SPIKE_FEATURE_COLS)
    assert len(bcm.SPIKE_FEATURE_COLS) == 4
    assert bcm.SPIKE_FEATURE_COLS == [
        "peak_8w",
        "lag1_ratio_to_8w_peak",
        "weeks_since_peak_8w",
        "burst_count_16w",
    ]


# --------------------------------------------------------------------------- #
# 3. No future leakage in the new spike features (toy-frame mutation invariance)
# --------------------------------------------------------------------------- #

def _toy_frame(weeks, products=("p1", "p2"), seed=7):
    rng = np.random.default_rng(seed)
    rows = []
    for pid in products:
        for w in weeks:
            rows.append({"product_id": pid, "week": w})
    df = pd.DataFrame(rows)
    df["order_demand"] = rng.integers(0, 40, size=len(df)).astype("float64")
    df["rolling_demand_mean_4w"] = df.groupby("product_id")["order_demand"].transform(
        lambda s: s.rolling(4, min_periods=1).mean().shift(1)
    )
    df["demand_lag_1"] = df.groupby("product_id")["order_demand"].transform(
        lambda s: s.shift(1)
    ).fillna(0.0)
    return df.sort_values(["product_id", "week"]).reset_index(drop=True)


def test_spike_features_depend_only_on_strictly_prior_weeks():
    df = _toy_frame([f"w{i:02d}" for i in range(12)])
    f0 = bcm._add_spike_features_impl(df)
    df2 = df.copy()
    last = df2.index[-1]
    df2.loc[last, "order_demand"] = df2.loc[last, "order_demand"] * 1000.0
    df2.loc[last, "demand_lag_1"] = 999.0
    f1 = bcm._add_spike_features_impl(df2)
    # rows < t must be unaffected by mutating row t (and its own feature may change)
    before = f0.iloc[:-1].reset_index(drop=True)
    after = f1.iloc[:-1].reset_index(drop=True)
    pd.testing.assert_frame_equal(
        before[bcm.SPIKE_FEATURE_COLS],
        after[bcm.SPIKE_FEATURE_COLS],
        check_dtype=False,
    )


def test_spike_features_preserve_all_original_columns_and_product_timeline():
    df = _toy_frame([f"w{i:02d}" for i in range(12)])
    out = bcm._add_spike_features_impl(df)
    expected_cols = list(df.columns) + list(bcm.SPIKE_FEATURE_COLS)
    assert list(out.columns) == expected_cols
    for c in bcm.SPIKE_FEATURE_COLS:
        assert not out[c].isna().any()
    pd.testing.assert_frame_equal(out[list(df.columns)], df.reset_index(drop=True))


def test_combined_timeline_features_do_not_leak_future_val_into_train():
    # train w00..w06, val w04..w09 => val overlaps the seam; the future part
    # of val must never change any train-row feature.
    train = _toy_frame([f"w{i:02d}" for i in range(7)]).reset_index(drop=True)
    val = _toy_frame([f"w{i:02d}" for i in range(4, 10)]).reset_index(drop=True)
    tr0, va0 = bcm.add_spike_context_features(train, val)
    val2 = val.copy()
    # Mutate p1's last val week (a middle point of the combined timeline) and
    # p2's very last row: no row strictly BEFORE the mutated position may change.
    p1_last = int(val2.index[(val2["product_id"] == "p1")][-1])
    val2.loc[p1_last, "order_demand"] = val2.loc[p1_last, "order_demand"] * 10_000.0
    val2.loc[val2.index[-1], "order_demand"] = val2.loc[val2.index[-1], "order_demand"] * 10_000.0
    val2.loc[val2.index[-1], "demand_lag_1"] = 999.0
    tr1, va1 = bcm.add_spike_context_features(train, val2)
    pd.testing.assert_frame_equal(tr0[bcm.SPIKE_FEATURE_COLS], tr1[bcm.SPIKE_FEATURE_COLS])
    # val rows at or before the p1 mutation point are unchanged
    keep = int(va0.index[(va0["product_id"] == "p1")][-1])
    pd.testing.assert_frame_equal(
        va0[bcm.SPIKE_FEATURE_COLS].iloc[: keep + 1],
        va1[bcm.SPIKE_FEATURE_COLS].iloc[: keep + 1],
    )
    assert list(tr0.columns) == list(train.columns) + list(bcm.SPIKE_FEATURE_COLS)
    assert list(va0.columns) == list(val.columns) + list(bcm.SPIKE_FEATURE_COLS)
    assert len(tr0) == len(train) and len(va0) == len(val)
    # row order preserved row-for-row
    assert list(tr0["week"]) == list(train["week"])
    assert list(tr0["product_id"]) == list(train["product_id"])
    assert list(va0["week"]) == list(val["week"])


# --------------------------------------------------------------------------- #
# 4. Weighting deterministic and train-only
# --------------------------------------------------------------------------- #

def test_wape_aligned_weights_match_preregistered_formula_and_are_deterministic():
    df = pd.DataFrame({"order_demand": [0.0, 0.0, 1.0, 2.0, 2.0, 10.0, 30.0, 5.0]})
    w1, stats1 = bcm.wape_aligned_weights(df)
    w2, stats2 = bcm.wape_aligned_weights(df)
    np.testing.assert_array_equal(w1, w2)
    assert stats1 == stats2
    p = df["order_demand"].to_numpy()
    pos = p[p > 0]
    p50, p99 = float(np.median(pos)), float(np.percentile(pos, 99))
    cap = max(1.0, p99 / p50)
    expected = np.minimum(np.maximum(1.0, p / p50), cap)
    np.testing.assert_allclose(w1, expected, rtol=1e-12)
    # zeros stay at weight 1; weights bounded in [1, cap]
    np.testing.assert_array_equal(w1[p == 0], 1.0)
    assert w1.min() >= 1.0 and w1.max() <= cap
    assert stats1["source"] == "train-labels + train statistics only"
    assert list(inspect.signature(bcm.wape_aligned_weights).parameters) == ["prepared_train"]


def test_wape_aligned_weights_stats_recorded_in_report():
    report = read_report()
    w = report["weights_arm2"]["stats"]
    assert w["formula"] == "min(max(1, d/p50+), p99+/p50+)"
    assert w["source"] == "train-labels + train statistics only"


# --------------------------------------------------------------------------- #
# 5. Cohort definitions unchanged (official functions + constants)
# --------------------------------------------------------------------------- #

def test_cohort_function_is_the_official_cohort_flags():
    assert bcm.cohort_flags is ff.cohort_flags


def test_cohort_constants_match_the_official_gate():
    assert ff.B_MIN_POS_WEEKS == 6
    assert ff.B_MIN_AVG_WEEKLY == 1.0
    assert ff.B2_MIN_AVG_WEEKLY == 0.5


def test_report_cohort_and_data_contract():
    report = read_report()
    dc = report["data_contract"]
    assert dc["cohorts_from_train"] is True
    assert dc["B_rows_val"] == 589
    assert dc["B_products_val"] == 37
    assert dc["val_weeks"] == 17
    assert dc["train_rows"] == 2_428_492


# --------------------------------------------------------------------------- #
# 6. Validation metrics reproducible (control reproduces the frozen V2 refs)
# --------------------------------------------------------------------------- #

def test_control_reproduces_frozen_v2_validation_references():
    report = read_report()
    rep = report["reproducibility"]
    ctrl = report["arms"]["v2_control"]["metrics"]
    assert rep["control_ok"] is True
    assert rep["control"]["reproduces_global"] is True
    assert rep["control"]["reproduces_B"] is True
    assert abs(rep["control"]["global_wape_recomputed"] - VAL_GLOBAL_WAPE_REF) <= 2e-3
    assert abs(rep["control"]["B_wape_recomputed"] - VAL_B_WAPE_REF) <= 1e-9
    assert abs(ctrl["B_wape"] - VAL_B_WAPE_REF) <= 1e-9
    assert abs(ctrl["B_bias_pct"] - VAL_B_BIAS_REF) <= 0.001
    assert abs(ctrl["wape"] - VAL_GLOBAL_WAPE_REF) <= 2e-3
    assert abs(rep["mase_denominator"] - 0.03804341247443925) <= 1e-9


@pytest.mark.parametrize("arm_id", ["v2_control", "wape_weights", "spike_features"])
def test_all_arms_pass_prediction_integrity(arm_id):
    report = read_report()
    integrity = report["arms"][arm_id]["integrity"]
    assert integrity["pred_integrity"] is True
    assert integrity["n_neg"] == 0
    assert integrity["n_nan_inf"] == 0


def test_all_arms_report_required_metric_suite():
    report = read_report()
    required = {
        "wape", "mae", "rmse", "mase", "bias_pct", "asym21", "pos_wape",
        "B_wape", "B_bias_pct", "B2_wape", "B2_bias_pct", "C_wape", "C_bias_pct",
    }
    for arm_id in ("v2_control", "wape_weights", "spike_features"):
        metrics = report["arms"][arm_id]["metrics"]
        assert required <= set(metrics)
        for k in required:
            assert metrics[k] is not None


# --------------------------------------------------------------------------- #
# 7. Decision made by pre-registered criteria only, on Validation
# --------------------------------------------------------------------------- #

def test_decision_is_from_the_pre_registered_enum():
    report = read_report()
    assert report["decision"]["decision"] in {"NO_VALID_IMPROVEMENT", "CANDIDATE_IMPROVED", "STOPPED"}
    assert report["pre_registered_criteria"] == bcm.DECISION_CRITERIA
    assert report["decision"]["rationale"]


def test_decision_checks_cover_every_preregistered_criterion():
    report = read_report()
    checks = report["decision"]["checks"]
    assert set(checks) == {"wape_weights", "spike_features"}
    expected_keys = {
        "B_wape_improved>=5%",
        "B_bias_reduced>=25%_and_not_worse",
        "global_wape<=+0.01",
        "B2_wape<=+0.03",
        "C_wape<=+0.01",
        "MAE<=x1.05",
        "gross_guards<=x1.10",
    }
    for arm_id in checks:
        assert set(checks[arm_id]) == expected_keys
    qualified = report["decision"].get("qualified")
    assert isinstance(qualified, list)
    if report["decision"]["decision"] == "CANDIDATE_IMPROVED":
        assert report["decision"]["selected_arm"] in {"wape_weights", "spike_features"}
        assert report["decision"]["selected_arm"] in qualified
    else:
        assert report["decision"]["selected_arm"] is None


def test_recorded_decision_matches_observed_metrics():
    report = read_report()
    ctrl = report["arms"]["v2_control"]["metrics"]
    arm3 = report["arms"]["spike_features"]["metrics"]
    cr = bcm.DECISION_CRITERIA
    # For the delivered experiment, Arm 3 satisfies all but the C WAPE tolerance
    # (a pure consequence of the observed numbers - keeps the guard honest).
    assert arm3["B_wape"] <= ctrl["B_wape"] * cr["B_wape_rel_to_control"]
    assert abs(arm3["B_bias_pct"]) <= abs(ctrl["B_bias_pct"]) * cr["B_bias_abs_reduction"]
    assert arm3["wape"] <= ctrl["wape"] + cr["global_wape_abs_tol"]
    assert arm3["B2_wape"] <= ctrl["B2_wape"] + cr["B2_wape_abs_tol"]
    assert arm3["C_wape"] > ctrl["C_wape"] + cr["C_wape_abs_tol"]  # the failing criterion
    assert report["decision"]["decision"] == "NO_VALID_IMPROVEMENT"
    assert report["decision"]["selected_arm"] is None


# --------------------------------------------------------------------------- #
# 8. V1 / V2 lifecycle artifacts untouched (read-only live check + source)
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
    assert "set_model_version_tag" not in src


def test_report_registry_integrity_flags():
    report = read_report()
    ri = report["registry_integrity"]
    assert ri["v1_untouched"] is True
    assert ri["v2_stage_unchanged_Staging"] is True
    assert ri["registered_or_transitioned_in_this_task"] is False
    assert ri["mlflow_history_rewritten"] is False


def test_live_registry_versions_and_stages_unchanged(client):
    versions = client.search_model_versions("name='demand_forecasting_final'")
    assert sorted(int(v.version) for v in versions) == [1, 2]
    v1 = client.get_model_version("demand_forecasting_final", "1")
    v2 = client.get_model_version("demand_forecasting_final", "2")
    assert v1.current_stage == "Staging"
    assert v1.run_id == "7fc871986e9240b1a4d98f7a9c1f893f"
    # V2 stands in Production per the LATER authorized formal gate-policy
    # revision (B_wape reclassified to diagnostic under the revised policy);
    # the experiment cycle itself never transitioned - same run_id, no V3.
    assert v2.current_stage == "Production"
    assert v2.run_id == "5d729a8594dc488bbd18ba0c4e216f5a"
    # the historical post-hoc gate tags survive (proof no V3 / no re-registration)
    assert v2.tags.get("posthoc_holdout_evaluated") == "true"
    assert v2.tags.get("production_gate") == "FAIL"


def test_v2_historical_posthoc_artifacts_untouched():
    ph = json.loads(POSTHOC_JSON.read_text(encoding="utf-8"))
    assert ph["posthoc_holdout_metrics"]["B_wape"] == TEST_B_WAPE_RECORD
    assert ph["quality_gate"]["passed"] is False
    assert ph["quality_gate"]["promoted_stage"] == "Staging"
    assert ph["registry"]["source_run_id"] == "5d729a8594dc488bbd18ba0c4e216f5a"
    report = read_report()
    hist = report["baseline_v2_history"]
    assert abs(hist["final_posthoc_test_wape"] - 0.375373) <= 1e-6
    assert abs(hist["final_posthoc_test_B_wape"] - 0.628239) <= 1e-6
    assert abs(hist["final_posthoc_test_B2_wape"] - 0.556648) <= 1e-6
    assert abs(hist["final_posthoc_test_C_wape"] - 0.365135) <= 1e-6
    assert hist["final_gate"] == "FAIL only B_wape > 0.60"
    assert hist["v2_stage"] == "Staging"


# --------------------------------------------------------------------------- #
# 9. Experiment artifacts reproducible + dedicated MLflow namespace
# --------------------------------------------------------------------------- #

def test_report_embeds_protocol_and_protection_statement():
    report = read_report()
    assert "The protected Test split was never opened by this experiment" in report["test_protection_statement"]
    assert "no experiment surface" not in report["test_protection_statement"]
    assert report["experiment"]["module"] == "src.models.b_cohort_remediation"
    assert report["experiment"]["pipeline"].startswith("log-LGBM frozen architecture")
    assert "n_estimators=600" in report["experiment"]["pipeline"]
    assert report["v2_references"] == bcm.V2_REFS


def test_md_report_covers_all_mandatory_sections():
    md = REPORT_MD.read_text(encoding="utf-8")
    for heading in [
        "## A. Baseline (V2 control reproduction)",
        "## B. Arm 2 - WAPE-aligned sample weights",
        "## C. Arm 3 - Spike-context features",
        "## D. Comparison",
        "## E. Error analysis",
        "## F. Decision",
        "## G. Next step",
    ]:
        assert heading in md, f"missing section {heading}"


def test_md_comparison_table_reports_all_required_metrics():
    md = REPORT_MD.read_text(encoding="utf-8")
    for metric in ["B_wape", "B_bias_pct", "B2_wape", "C_wape", "mae", "mase", "asym21", "pos_wape"]:
        assert f"| {metric} |" in md


def test_live_remediation_experiment_runs_additive_and_tagged(client):
    import mlflow
    from mlflow.tracking import MlflowClient

    exp = client.get_experiment_by_name(bcm.EXPERIMENT_NAME)
    assert exp is not None
    runs = client.search_runs([exp.experiment_id], order_by=["attributes.start_time"])
    assert len(runs) == 3
    by_name = {r.data.tags.get("mlflow.runName"): r for r in runs}
    assert set(by_name) == {"arm1_v2_control", "arm2_wape_weights", "arm3_spike_features"}
    report = read_report()
    for arm_id, run_id in report["run_ids"].items():
        run = MlflowClient().get_run(run_id)
        assert run.info.status == "FINISHED"
        assert run.data.tags.get("mlflow.runName") == bcm.RUN_NAMES[arm_id]
        assert run.data.tags.get("test_used") == "false"
        assert run.data.tags.get("model_selection_stage") == "validation_only"
        assert run.data.tags.get("registered") == "false"
        assert run.data.tags.get("part3_controlled_experiment") == "true"
        assert run.data.params.get("feature_count") == str(report["arms"][arm_id]["feature_count"])


def test_lifecycle_experiment_has_no_remediation_noise(client):
    # No remediation run may leak into the official lifecycle experiment.
    runs = client.search_runs(
        [client.get_experiment_by_name("part3_final_lifecycle").experiment_id],
        order_by=["attributes.start_time"],
    )
    for r in runs:
        if r.data.tags.get("part3_controlled_experiment") == "true":
            pytest.fail(f"controlled-experiment run leaked into lifecycle experiment: {r.info.run_id}")