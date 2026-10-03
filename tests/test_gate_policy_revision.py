"""Part-3 V2 quality-gate policy revision guards.

Encodes the authorized formal quality-gate policy revision contract
(``src/models/gate_policy_revision.py``):

* Policy versioning: ``quality_gate_policy_v1_legacy`` (historical default,
  B_wape <= 0.60 is a hard blocker; 19 outcomes) vs
  ``quality_gate_policy_v2_revised`` (B_wape reported as a diagnostic /
  monitoring metric, non-blocking; 11 numeric + 7 technical = 18 blocking
  outcomes). ``evaluate_gate`` defaults to the LEGACY policy so every
  pre-existing caller keeps the exact legacy behavior.
* The revised gate keeps ``B_bias_pct >= -20`` (and every other criterion)
  as a HARD blocker; only ``B_wape`` is reclassified (owner decision after
  the provenance audit -- NOT tuned against any Test outcome, hence
  ``test_used_for_policy_revision=false``).
* No Test access: metrics are replayed from the persisted, already-public
  post-hoc evaluation report; the legacy gate replay must be deterministic and
  byte-identical to the persisted post-hoc outcomes (19 outcomes, single
  failure ``B_wape``).
* No ``--force``: an exactly-once guard (dedicated marker file +
  ``gate_policy_revision_applied`` MLflow run tag) refuses re-execution.
* Additive tags only: ``production_gate=FAIL``, ``gate_decision=FAIL``,
  ``promoted_stage=Staging``, ``lifecycle_stage=Staging`` and all historical
  v2 tags are preserved verbatim; the revision decision is recorded under NEW
  tag keys. Only version 2's stage may be transitioned (Staging -> Production
  iff every revised hard criterion passes). No V3, no registration, V1
  untouched.

These tests are read-only: JSON/metadata reads and MLflow queries only. They
never create runs, versions, tags, transitions, or rewrite files.
"""

import json
import re
from pathlib import Path

import pytest

from src.models.quality_gate import (
    POLICY_LEGACY,
    POLICY_REVISED,
    evaluate_gate,
)
import src.models.gate_policy_revision as gpr

ROOT = Path(__file__).resolve().parents[1]

REVISION_MARKER = Path(
    r"C:\Users\HEDAYA~1\AppData\Local\Temp\opencode\experiments\part3_v2_gate_policy_revision_marker.json"
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def read_repo_json(rel: str) -> dict:
    return json.loads((ROOT / rel).read_text(encoding="utf-8"))


def ok_metrics() -> dict:
    return dict(read_repo_json("reports/forecasting/part3_v2_posthoc_holdout_evaluation.json")[
        "posthoc_holdout_metrics"
    ])


def ok_technical() -> dict:
    payload = read_repo_json("reports/forecasting/part3_v2_posthoc_holdout_evaluation.json")
    return {
        o["key"]: bool(o["passed"])
        for o in payload["quality_gate"]["outcomes"]
        if o["op"] == "bool"
    }


def revision_applied() -> bool:
    already, _ = gpr._already_revised()
    return already


# --------------------------------------------------------------------------- #
# 1. Policy versioning in the quality gate
# --------------------------------------------------------------------------- #
def test_policy_constants_defined():
    assert POLICY_LEGACY == "quality_gate_policy_v1_legacy"
    assert POLICY_REVISED == "quality_gate_policy_v2_revised"
    assert gpr.DIAGNOSTIC_METRICS == {"B_wape": "diagnostic"}


def test_evaluate_gate_defaults_to_legacy_policy():
    metrics = ok_metrics()
    technical = ok_technical()
    default = evaluate_gate(metrics, technical)
    legacy = evaluate_gate(metrics, technical, policy=POLICY_LEGACY)
    assert default.to_dict() == legacy.to_dict()
    assert default.policy == POLICY_LEGACY
    assert len(default.outcomes) == 19  # 12 numeric + 7 technical


def test_legacy_policy_lists_b_wape_as_blocker():
    metrics = ok_metrics()
    technical = ok_technical()
    legacy = evaluate_gate(metrics, technical, policy=POLICY_LEGACY)
    keys = [o.key for o in legacy.outcomes]
    assert "B_wape" in keys
    assert len([o for o in legacy.outcomes if o.key == "B_wape"]) == 1
    # duplicated-name guard: B_wape must not also appear as a diagnostic row
    assert len(legacy.diagnostics) == 0


def test_revised_policy_moves_b_wape_to_diagnostics():
    metrics = ok_metrics()
    technical = ok_technical()
    revised = evaluate_gate(metrics, technical, policy=POLICY_REVISED)
    assert revised.policy == POLICY_REVISED
    blocking_keys = [o.key for o in revised.outcomes]
    assert "B_wape" not in blocking_keys
    assert "B_bias_pct" in blocking_keys  # hard business constraint survives
    assert len(revised.outcomes) == 18  # 11 numeric + 7 technical
    assert [d.key for d in revised.diagnostics] == ["B_wape"]
    assert revised.diagnostics[0].bound == "diagnostic"
    assert revised.diagnostics[0].passed is True  # non-blocking by construction


def test_revised_policy_still_blocks_b_bias_defect():
    metrics = ok_metrics()
    technical = ok_technical()
    metrics["B_bias_pct"] = -31.0  # old -30% stockout defect regime
    revised = evaluate_gate(metrics, technical, policy=POLICY_REVISED)
    assert revised.passed is False
    assert revised.promoted_stage == "Staging"
    assert "B_bias_pct" in [o.key for o in revised.failed_criteria()]


def test_revised_policy_still_blocks_technical_failures():
    metrics = ok_metrics()
    technical = ok_technical()
    technical["schema_ok"] = False
    revised = evaluate_gate(metrics, technical, policy=POLICY_REVISED)
    assert revised.passed is False
    assert revised.promoted_stage == "Staging"
    assert "schema_ok" in [o.key for o in revised.failed_criteria()]


def test_unknown_policy_rejected():
    with pytest.raises(ValueError):
        evaluate_gate(ok_metrics(), ok_technical(), policy="not-a-policy")


def test_gate_documentation_covers_both_policies():
    from src.models.quality_gate import gate_documentation_markdown

    md = gate_documentation_markdown()
    assert "## Policy versioning" in md
    assert POLICY_LEGACY in md
    assert POLICY_REVISED in md
    assert "diagnostic" in md


# --------------------------------------------------------------------------- #
# 2. Persisted-source integrity (metrics + technical flags)
# --------------------------------------------------------------------------- #
def test_persisted_posthoc_source_present():
    assert gpr.POSTHOC_AUDIT.exists()
    payload = read_repo_json("reports/forecasting/part3_v2_posthoc_holdout_evaluation.json")
    for key in ("posthoc_holdout_metrics", "quality_gate"):
        assert key in payload


def test_metrics_and_technical_extraction():
    persisted = gpr._load_persisted_evaluation()
    metrics = gpr._extract_metrics(persisted)
    technical = gpr._extract_technical(persisted)
    assert metrics["B_wape"] == pytest.approx(0.628239, abs=1e-6)
    assert len(technical) == 7
    assert all(technical.values())  # persisted post-hoc technicals all PASS


def test_module_has_no_test_split_access():
    src = (ROOT / "src" / "models" / "gate_policy_revision.py").read_text(encoding="utf-8")
    start = src.find('"""')
    end = src.find('"""', start + 3)
    body = src[end + 3:]  # module docstring excluded (it names the protected file to ban)
    for token in ("data/splits", "test.csv", "design_matrix_rows", "full_metric_suite",
                  "score_", "build_prepared", "load_model", "pyfunc.load_model"):
        assert token not in body, f"forbidden Test/scoring touchpoint present: {token}"
    for imp in ("data_prep", "final_forecast", "forecasting", "finalize_lifecycle",
                "posthoc_holdout"):
        assert re.search(rf"import\s+{imp}\b|from\s+{imp}\b", body) is None, (
            f"module must not import scoring/split machinery: {imp}"
        )


# --------------------------------------------------------------------------- #
# 3. Deterministic replay + revised decision on persisted metrics (pure logic)
# --------------------------------------------------------------------------- #
def test_legacy_replay_is_deterministic_and_identical():
    persisted = gpr._load_persisted_evaluation()
    metrics = gpr._extract_metrics(persisted)
    technical = gpr._extract_technical(persisted)
    legacy = gpr._replay_legacy(metrics, technical)
    assert legacy["passed"] is False
    assert legacy["failed_criteria"] == ["B_wape"]
    assert legacy["deterministic"] is True
    assert legacy["outcome_count"] == 19
    check = gpr._verify_replay_matches_persisted(persisted, legacy)
    assert check["identical"] is True
    assert check["mismatches"] == []


def test_evaluate_decision_revised_gate_applied():
    decision = gpr.evaluate()
    assert decision["policy_decision"] == "REVISED_GATE_APPLIED"
    assert decision["v2_decision"] == "PRODUCTION"
    assert decision["policy"] == POLICY_REVISED
    assert decision["legacy_gate"]["passed"] is False
    assert decision["legacy_gate"]["failed_criteria"] == ["B_wape"]
    assert decision["legacy_replay"]["identical_to_persisted"] is True
    assert decision["revised_gate"]["passed"] is True
    assert decision["revised_gate"]["promoted_stage"] == "Production"
    assert decision["revised_gate"]["blocking_outcome_count"] == 18
    assert all(o["passed"] for o in decision["revised_gate"]["blocking_criteria"])
    diag = decision["revised_gate"]["diagnostic_metrics"]
    assert [d["key"] for d in diag] == ["B_wape"]
    assert diag[0]["observed"] == pytest.approx(0.628239, abs=1e-6)
    assert diag[0]["role"] == "diagnostic"
    assert decision["test_used_for_policy_revision"] is False
    assert decision["b_wape_role"] == "diagnostic"


# --------------------------------------------------------------------------- #
# 4. Module source-level guards (no --force, additive-only tags)
# --------------------------------------------------------------------------- #
def test_module_exposes_no_force_flag():
    src = (ROOT / "src" / "models" / "gate_policy_revision.py").read_text(encoding="utf-8")
    assert "argparse" not in src
    assert "add_argument" not in src
    assert "sys.argv" not in src, "module must not read CLI args (no --force accepted)"


def test_revision_tags_are_additive_and_never_rewrite_history():
    decision = gpr.evaluate()
    tags = gpr.build_revision_tags(decision)
    for required in (
        "gate_policy_version",
        "legacy_gate_decision",
        "revised_gate_decision",
        "b_wape_role",
        "evaluation_type",
        "test_used_for_policy_revision",
    ):
        assert required in tags
    assert tags["gate_policy_version"] == POLICY_REVISED
    assert tags["legacy_gate_decision"] == "FAIL"
    assert tags["revised_gate_decision"] == "PASS"
    assert tags["b_wape_role"] == "diagnostic"
    assert tags["evaluation_type"] == "post_hoc_holdout_evaluation"
    assert tags["test_used_for_policy_revision"] == "false"
    # historical keys must never be written by the revision
    for forbidden in ("production_gate", "gate_decision", "promoted_stage",
                      "lifecycle_stage", "evaluation_group", "test_scored_once",
                      "final_wape", "posthoc_holdout_evaluated"):
        assert forbidden not in tags


# --------------------------------------------------------------------------- #
# 5. Post-approval state (skips until the module has been executed once)
# --------------------------------------------------------------------------- #
def test_second_execution_refused_when_marker_present():
    if not revision_applied():
        pytest.skip("revision not yet applied in this workspace")
    with pytest.raises(SystemExit) as exc:
        gpr.main()
    assert exc.value.code == gpr.STOP_EXIT_CODE


def test_marker_records_policy_and_version():
    if not REVISION_MARKER.exists():
        pytest.skip("revision marker missing")
    content = json.loads(REVISION_MARKER.read_text(encoding="utf-8"))
    assert content["policy_applied"] == POLICY_REVISED
    assert content["policy_legacy"] == POLICY_LEGACY
    assert content["model_version"] == 2
    assert content["source_run_id"] == "5d729a8594dc488bbd18ba0c4e216f5a"


@pytest.fixture(scope="module")
def client():
    from src.models.registry import init_mlflow

    return init_mlflow()


@pytest.fixture(scope="module")
def revision_audit() -> dict:
    if not revision_applied():
        pytest.skip("revision not yet applied")
    return read_repo_json("reports/forecasting/part3_v2_gate_policy_revision.json")


def test_live_registry_versions_stages(client, revision_audit):
    versions = client.search_model_versions("name='demand_forecasting_final'")
    assert sorted(int(v.version) for v in versions) == [1, 2]
    v1 = client.get_model_version("demand_forecasting_final", "1")
    v2 = client.get_model_version("demand_forecasting_final", "2")
    assert v1.current_stage == "Staging"
    assert v1.run_id == "7fc871986e9240b1a4d98f7a9c1f893f"
    assert v2.run_id == "5d729a8594dc488bbd18ba0c4e216f5a"
    assert v2.current_stage == "Production"


def test_run_and_version_revision_tags_applied(client, revision_audit):
    run = client.get_run("5d729a8594dc488bbd18ba0c4e216f5a")
    v2 = client.get_model_version("demand_forecasting_final", "2")
    required = {
        "gate_policy_version": POLICY_REVISED,
        "legacy_gate_decision": "FAIL",
        "revised_gate_decision": "PASS",
        "b_wape_role": "diagnostic",
        "evaluation_type": "post_hoc_holdout_evaluation",
        "test_used_for_policy_revision": "false",
    }
    for key, value in required.items():
        assert run.data.tags.get(key) == value, f"run tag {key}"
        assert v2.tags.get(key) == value, f"version tag {key}"


def test_historical_posthoc_tags_preserved(client, revision_audit):
    run = client.get_run("5d729a8594dc488bbd18ba0c4e216f5a")
    v2 = client.get_model_version("demand_forecasting_final", "2")
    run_keys = {
        "production_gate": "FAIL",
        "lifecycle_stage": "Staging",
        "posthoc_holdout_evaluated": "true",
        "evaluation_group": "v2_validation",
        "test_scored_once": "false",
    }
    version_keys = {
        **run_keys,
        "gate_decision": "FAIL",
        "promoted_stage": "Staging",
        "wape": "0.3964",
    }
    for key, value in run_keys.items():
        assert run.data.tags.get(key) == value, f"run historical tag {key} unchanged"
    for key, value in version_keys.items():
        assert v2.tags.get(key) == value, f"version historical tag {key} unchanged"


def test_revision_audit_content(revision_audit):
    a = revision_audit
    assert a["phase"] == "part3-v2-gate-policy-revision"
    assert a["policy_decision"] == "REVISED_GATE_APPLIED"
    assert a["v2_decision"] == "PRODUCTION"
    assert a["policy"]["name"] == POLICY_REVISED
    assert a["policy"]["test_used_for_policy_revision"] is False
    assert a["legacy_gate"]["failed_criteria"] == ["B_wape"]
    assert a["legacy_replay"]["identical_to_persisted"] is True
    assert a["revised_gate"]["passed"] is True
    assert a["revised_gate"]["blocking_outcome_count"] == 18
    assert len(a["blocking_criteria"]) == 18
    assert a["diagnostic_metrics"] == [
        {
            "key": "B_wape",
            "description": "B cohort WAPE (sustained high-volume, the business-critical segment)",
            "observed": 0.628239,
            "bound": "diagnostic",
            "role": "diagnostic",
        }
    ]
    assert a["registry"]["version"] == 2
    assert a["registry"]["stage_after"] == "Production"
    assert a["registry"]["transitioned"] is True
    assert a["registry"]["historical_tags_preserved"] is True
    integ = a["integrity"]
    assert integ["no_test_access"] is True
    assert integ["metrics_replayed_from_persisted_report"] is True
    assert integ["no_retraining"] is True
    assert integ["no_threshold_value_change"] is True
    assert integ["no_v3_no_registration"] is True
    assert integ["v1_history_preserved"] is True
    proto = a["protocol"]
    assert proto["exactly_once"] is True
    assert proto["force_override_exists"] is False
    assert proto["retraining_performed"] is False
    assert proto["thresholds_changed"] is False
    assert proto["second_model_created"] is False
    assert proto["test_used_for_policy_revision"] is False