"""Part-3 V2 post-hoc holdout evaluation lifecycle guards.

Encodes the authorized post-hoc holdout contract established by
``src/models/posthoc_holdout.py``:

* The result is *classified* as ``post_hoc_holdout_evaluation`` and a verbatim
  disclosure (the Test split was already consumed during V1 evaluation) is
  mandated in every artifact -- the result must NEVER be claimed as an
  untouched / independent / unseen / clean final test.
* The module has NO ``--force`` override; an exactly-once guard (dedicated
  marker file + ``posthoc_holdout_evaluated`` MLflow run tag) refuses any
  second execution.
* V1 history (version 1, run ``7fc87198...``, ``part3_final_*`` reports) is
  preserved untouched; ONLY version 2 is updated and it REMAINS version 2
  (no v3, no registration).
* The existing Quality Gate implementation is consumed unchanged (thresholds,
  metric keys, cohort definitions); the gate outcome is recorded honestly.
* The final report / audit reflect the ACTUAL recorded stage (no empty
  ``stage_after`` anywhere).
* Prediction integrity (finite, non-negative, aligned, inverse-transform
  verified) is enforced and reported.

These tests are read-only: JSON/metadata reads and MLflow queries only. They
never create runs, versions, tags, transitions, or rewrite files.
"""

import hashlib
import json
from pathlib import Path

import pytest

import src.models.posthoc_holdout as ph

ROOT = Path(__file__).resolve().parents[1]

DISCLOSURE = (
    "The V2 holdout evaluation is classified as post-hoc because the designated "
    "Test split had previously been consumed during the V1 evaluation and "
    "diagnosis lifecycle. Therefore, this result is not claimed as an "
    "independent untouched final-test estimate."
)


def md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def read_repo_json(rel: str) -> dict:
    return json.loads((ROOT / rel).read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def audit() -> dict:
    return read_repo_json("reports/forecasting/part3_v2_posthoc_holdout_evaluation.json")


@pytest.fixture(scope="module")
def report_md() -> str:
    return (ROOT / "reports" / "forecasting" / "part3_v2_posthoc_holdout_evaluation.md").read_text(
        encoding="utf-8"
    )


@pytest.fixture(scope="module")
def client():
    from src.models.registry import init_mlflow

    return init_mlflow()


# --------------------------------------------------------------------------- #
# 1. Mandatory disclosure + correct labeling
# --------------------------------------------------------------------------- #


def test_mandatory_disclosure_verbatim_in_all_artifacts(report_md, audit):
    assert DISCLOSURE in report_md
    assert audit["disclosure"] == DISCLOSURE
    assert ph.DISCLOSURE == DISCLOSURE


def test_forbidden_independent_test_labels_absent(report_md):
    # The report contains an explicit disclaimer ("never claims an untouched
    # test ..."). Remove that disclaimer sentence, then require that NO
    # forbidden independent-test label is used as a positive claim.
    body = "\n".join(
        line for line in report_md.splitlines() if "never claims" not in line
    ).lower()
    for forbidden in (
        "untouched test",
        "independent final test",
        "unseen test",
        "clean final test",
        "independent generalization estimate",
    ):
        assert forbidden not in body, f"forbidden label present: {forbidden}"


def test_evaluation_labeled_post_hoc(audit):
    assert audit["evaluation_type"] == "post_hoc_holdout_evaluation"
    assert audit["evaluation_group"] == "post_hoc_holdout"
    assert audit["phase"] == "part3-v2-posthoc-holdout"
    assert audit["model"]["version"] == 2
    assert audit["model"]["uri"] == "models:/demand_forecasting_final/2"
    assert audit["model"]["source_run_id"] == "5d729a8594dc488bbd18ba0c4e216f5a"
    assert audit["test_metadata"]["rows"] == 520_391


# --------------------------------------------------------------------------- #
# 2. No --force; module has no override path
# --------------------------------------------------------------------------- #


def test_module_exposes_no_force_flag():
    src = (ROOT / "src" / "models" / "posthoc_holdout.py").read_text(encoding="utf-8")
    assert "argparse" not in src, "post-hoc module must not define a CLI"
    assert "add_argument" not in src
    assert "sys.argv" not in src, "module must not read CLI args (no --force accepted)"
    assert "force_override_exists" in src  # audit still records the fact


def test_marker_guard_prevents_second_execution(audit):
    marker = Path(r"C:\Users\HEDAYA~1\AppData\Local\Temp\opencode\experiments\part3_v2_posthoc_holdout_marker.json")
    assert marker.exists(), "exactly-once marker missing"
    content = json.loads(marker.read_text(encoding="utf-8"))
    assert content["evaluation_type"] == "post_hoc_holdout_evaluation"
    assert content["model_version"] == 2
    already, evidence = ph._already_evaluated()
    assert already is True
    assert "marker file" in evidence
    with pytest.raises(SystemExit) as exc:
        ph.main()
    assert exc.value.code == ph.STOP_EXIT_CODE


def test_protocol_flags_recorded(audit):
    protocol = audit["protocol"]
    assert protocol["exactly_once"] is True
    assert protocol["force_override_exists"] is False
    assert protocol["retraining_performed"] is False
    assert protocol["tuning_after_evaluation"] is False
    assert protocol["thresholds_changed"] is False
    assert protocol["second_model_created"] is False
    assert protocol["v1_history_preserved"] is True


# --------------------------------------------------------------------------- #
# 3. V1 preserved; only V2 updated; still only version 2
# --------------------------------------------------------------------------- #


def test_v1_history_preserved(client):
    v1 = client.get_model_version("demand_forecasting_final", "1")
    assert v1.current_stage == "Staging"
    assert v1.run_id == "7fc871986e9240b1a4d98f7a9c1f893f"
    assert v1.tags.get("test_scored_once") == "true"
    assert v1.tags.get("gate_decision") == "FAIL"
    assert "posthoc_holdout_evaluated" not in v1.tags
    v1_run = client.get_run("7fc871986e9240b1a4d98f7a9c1f893f")
    assert v1_run.data.tags.get("test_scored_once") == "true"
    assert "posthoc_holdout_evaluated" not in v1_run.data.tags
    # historical report facts untouched
    assert read_repo_json("reports/forecasting/part3_final_result.json")["registry"]["stage"] == "Staging"
    assert md5(ROOT / "data" / "splits" / "test.csv") == "d35bfcd546df74b63c9bd25151fde264"


def test_only_v2_updated_and_version_remains_2(client):
    versions = client.search_model_versions("name='demand_forecasting_final'")
    nums = sorted(int(v.version) for v in versions)
    assert nums == [1, 2], "a competing/new version exists (regression in exactly-once protocol)"
    v2 = client.get_model_version("demand_forecasting_final", "2")
    assert v2.run_id == "5d729a8594dc488bbd18ba0c4e216f5a"
    assert v2.tags.get("posthoc_holdout_evaluated") == "true"
    assert v2.tags.get("evaluation_type") == "post_hoc_holdout_evaluation"
    # historical V2 validation metadata preserved on the same version
    assert v2.tags.get("test_scored_once") == "false"
    assert v2.tags.get("evaluation_group") == "v2_validation"
    assert v2.tags.get("wape") == "0.3964"


def test_run_and_version_tags_applied(client):
    run = client.get_run("5d729a8594dc488bbd18ba0c4e216f5a")
    required_run_tags = [
        "posthoc_holdout_evaluated",
        "evaluation_type",
        "evaluation_dataset",
        "lifecycle_stage",
        "production_gate",
        "test_start_date",
        "test_end_date",
        "test_rows",
        "final_wape",
        "final_mae",
        "final_rmse",
        "final_mase",
        "final_bias",
        "final_asymmetric_loss",
        "positive_wape",
    ]
    for key in required_run_tags:
        assert key in run.data.tags, f"missing run tag {key}"
    assert run.data.tags["posthoc_holdout_evaluated"] == "true"
    assert run.data.tags["evaluation_dataset"] == "test"
    assert run.data.tags["test_rows"] == "520391"
    v2 = client.get_model_version("demand_forecasting_final", "2")
    for key in required_run_tags:
        assert key in v2.tags, f"missing version tag {key}"


def test_posthoc_artifacts_logged_to_v2_run(client):
    paths = [a.path for a in client.list_artifacts("5d729a8594dc488bbd18ba0c4e216f5a")]
    assert "posthoc_holdout_audit.json" in paths
    assert "posthoc_holdout_report.md" in paths


# --------------------------------------------------------------------------- #
# 4. Quality Gate implementation unchanged + honest outcome
# --------------------------------------------------------------------------- #


def test_gate_implementation_unchanged():
    from src.models.quality_gate import _NUMERIC_CRITERIA, _TECHNICAL_CRITERIA, evaluate_gate

    assert evaluate_gate is ph.evaluate_gate  # untouched implementation consumed
    bounds = {c["key"]: c["bound"] for c in _NUMERIC_CRITERIA}
    assert bounds["B_wape"] == 0.6
    assert bounds["wape"] == 0.41
    assert bounds["B_bias_pct"] == -20.0
    assert len(_NUMERIC_CRITERIA) == 12
    assert len(_TECHNICAL_CRITERIA) == 7


def test_gate_outcome_honest(audit):
    gate = audit["quality_gate"]
    assert gate["passed"] is False
    assert gate["promoted_stage"] == "Staging"
    assert len(gate["outcomes"]) == 19  # 12 numeric + 7 technical
    assert all(o["group"] == "post_hoc_holdout" for o in gate["outcomes"])
    b_wape = next(o for o in gate["outcomes"] if o["key"] == "B_wape")
    assert b_wape["passed"] is False
    assert b_wape["observed"] > 0.6
    assert all(o["passed"] for o in gate["outcomes"] if o["key"] != "B_wape")


def test_final_metrics_reported_unchanged_definitions(audit):
    metrics = audit["posthoc_holdout_metrics"]
    for key in (
        "wape", "mae", "rmse", "mase", "bias_pct", "asym21", "pos_wape",
        "B_wape", "B_bias_pct", "B2_wape", "B2_bias_pct", "C_wape", "C_bias_pct",
    ):
        assert key in metrics, f"missing metric {key}"


# --------------------------------------------------------------------------- #
# 5. Prediction integrity + final stage recorded
# --------------------------------------------------------------------------- #


def test_prediction_integrity_enforced(audit):
    integrity = audit["integrity"]
    assert integrity["pred_integrity"] is True
    assert integrity["rows_scored"] == 520_391
    assert integrity["n_neg"] == 0
    assert integrity["n_nan_inf"] == 0
    assert integrity["length_match"] is True
    assert integrity["row_product_week_alignment"] is True
    assert integrity["schema_columns"] == 93
    assert integrity["schema_order_ok"] is True
    assert integrity["signature_ok"] is True
    assert integrity["artifact_schema_ok"] is True
    assert integrity["inverse_transform_verified"] is True
    assert integrity["split_reproduction"]["prepared_rows"] == 520_391
    assert integrity["test_csv_md5_pinned"] == "d35bfcd546df74b63c9bd25151fde264"


def test_final_report_contains_actual_stage(audit, report_md):
    # Report ordering bug guard: no artifact may carry an empty stage_after.
    assert "Production -> Staging" in report_md
    assert "transitioned: `True`" in report_md
    assert audit["model"]["stage_after"] == "Staging"
    assert audit["registry"]["stage_after"] == "Staging"
    assert audit["registry"]["stage_before"] == "Production"
    assert audit["registry"]["transitioned"] is True
    assert audit["registry"]["gate_decision"] == "FAIL"
    assert audit["registry"]["version"] == 2