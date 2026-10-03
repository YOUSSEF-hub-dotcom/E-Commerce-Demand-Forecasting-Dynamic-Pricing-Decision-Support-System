"""Lifecycle-route guard tests (Part-3 automation gap closure).

Proves the guarded promotion contract on an isolated MLflow store:

* deterministic PASS -> Production / FAIL -> Staging routing,
* a PASS claim REQUIRES blocking-criteria evidence (no bare passed=True),
* a contradictory PASS claim (any failed blocking outcome) is refused,
* provenance: the decision's run_id must match the registered version.run_id,
* policy: the decision must match the version's recorded policy tag on clash,
* guarded promotion records the active policy additively on first verified run,
* check_only never transitions and never writes tags,
* guarded Production promotions archive the previously-Production version,
* FAIL decisions route to Staging and never touch Production.
"""

import pytest

import numpy as np
import pandas as pd

mlflow = pytest.importorskip("mlflow")
from mlflow.tracking import MlflowClient

import src.models.registry as registry
import src.models.lifecycle_router as router
from src.models.lifecycle_router import (
    LifecycleRouteError,
    STAGE_PRODUCTION,
    STAGE_STAGING,
    route_stage,
    verify_decision_record,
)


@pytest.fixture
def isolated_store(tmp_path, monkeypatch):
    uri = (tmp_path / "mlruns").as_uri()
    monkeypatch.setattr(registry, "TRACKING_URI", uri)
    monkeypatch.setattr(registry, "FINAL_EXPERIMENT", "test_lifecycle")
    monkeypatch.setattr(registry, "MODEL_NAME", "test_demand_forecaster")
    monkeypatch.setattr(registry, "ARTIFACTS_DIR", tmp_path)
    mlflow.set_tracking_uri(uri)
    client = registry.init_mlflow()
    yield uri, client


class _ConstModel:
    def predict(self, X):
        return np.zeros(X.shape[0])


def _cols():
    return [f"f{i:02d}" for i in range(93)]


def _register(client, passed):
    cols = _cols()
    run_id = registry.log_final_run(
        model=_ConstModel(),
        feature_columns=cols,
        freeze_config={"feature_version": "phase3-exp2-v1", "params": {"lr": 1.0}},
        val_metrics={},
        test_metrics={"wape": 0.38},
    )
    reg = registry.register_and_promote(run_id, passed=passed)
    return reg["version"], run_id


def _pass_decision(policy=router.VALID_POLICIES[1]):
    return {
        "policy": policy,
        "passed": True,
        "promoted_stage": STAGE_PRODUCTION,
        "outcomes": [
            {"key": "wape", "op": "le", "bound": 0.41, "observed": 0.38, "passed": True},
            {"key": "pred_integrity", "op": "bool", "bound": True, "observed": True, "passed": True},
        ],
    }


def _fail_decision(policy=router.VALID_POLICIES[1]):
    return {
        "policy": policy,
        "passed": False,
        "promoted_stage": STAGE_STAGING,
        "outcomes": [
            {"key": "B_wape", "op": "le", "bound": 0.60, "observed": 0.63, "passed": False},
        ],
    }


# ---- routing -----------------------------------------------------------------

def test_route_stage_pass_to_production():
    assert route_stage(True) == STAGE_PRODUCTION


def test_route_stage_fail_to_staging():
    assert route_stage(False) == STAGE_STAGING


# ---- decision-record contract ------------------------------------------------

def test_decision_pass_requires_criteria_evidence():
    with pytest.raises(LifecycleRouteError, match="without blocking-criteria evidence"):
        verify_decision_record({"policy": "quality_gate_policy_v1_legacy", "passed": True})


def test_decision_contradictory_pass_refused():
    bad = _pass_decision()
    bad["outcomes"][0]["passed"] = False
    with pytest.raises(LifecycleRouteError, match="contradicts blocking outcomes"):
        verify_decision_record(bad)


def test_decision_unknown_policy_refused():
    with pytest.raises(LifecycleRouteError, match="unknown or missing policy"):
        verify_decision_record({"policy": "made_up_policy", "passed": True, "outcomes": []})


def test_decision_stage_inconsistent_refused():
    d = _pass_decision()
    d["promoted_stage"] = STAGE_STAGING
    with pytest.raises(LifecycleRouteError, match="inconsistent"):
        verify_decision_record(d)


def test_decision_valid_pass_ok():
    out = verify_decision_record(_pass_decision())
    assert out["passed"] is True
    assert out["promoted_stage"] == STAGE_PRODUCTION
    assert out["blocking_count"] == 2


def test_decision_valid_fail_ok():
    out = verify_decision_record(_fail_decision())
    assert out["passed"] is False
    assert out["promoted_stage"] == STAGE_STAGING


# ---- guarded promotion -------------------------------------------------------

def test_guarded_promote_pass_transitions_staging_to_production(isolated_store):
    uri, client = isolated_store
    version, run_id = _register(client, passed=False)
    result = router.guarded_promote(
        model_name=registry.MODEL_NAME, version=version, run_id=run_id,
        decision=_pass_decision(),
    )
    assert result["passed"] is True
    assert result["source_stage"] == STAGE_STAGING
    assert result["target_stage"] == STAGE_PRODUCTION
    assert result["transitioned"] is True
    mv = client.search_model_versions(f"name='{registry.MODEL_NAME}'")[0]
    assert mv.current_stage == STAGE_PRODUCTION
    assert mv.tags.get(router.POLICY_TAG) == router.VALID_POLICIES[1]


def test_guarded_promote_fail_stays_staging(isolated_store):
    uri, client = isolated_store
    version, run_id = _register(client, passed=False)
    result = router.guarded_promote(
        model_name=registry.MODEL_NAME, version=version, run_id=run_id,
        decision=_fail_decision(),
    )
    assert result["target_stage"] == STAGE_STAGING
    assert result["transitioned"] is False
    assert client.search_model_versions(f"name='{registry.MODEL_NAME}'")[0].current_stage == STAGE_STAGING


def test_guarded_promote_passes_archives_previous_production(isolated_store):
    uri, client = isolated_store
    v1, run1 = _register(client, passed=True)
    v2, run2 = _register(client, passed=False)
    router.guarded_promote(
        model_name=registry.MODEL_NAME, version=v2, run_id=run2, decision=_pass_decision()
    )
    stages = {str(v.version): v.current_stage for v in client.search_model_versions(f"name='{registry.MODEL_NAME}'")}
    assert stages[str(v2)] == STAGE_PRODUCTION
    assert stages[str(v1)] == "Archived"


def test_guarded_promote_provenance_mismatch_refused(isolated_store):
    uri, client = isolated_store
    version, run_id = _register(client, passed=True)
    with pytest.raises(LifecycleRouteError, match="provenance mismatch"):
        router.guarded_promote(
            model_name=registry.MODEL_NAME, version=version, run_id="deadbeef" * 4,
            decision=_pass_decision(),
        )
    assert client.search_model_versions(f"name='{registry.MODEL_NAME}'")[0].current_stage == STAGE_PRODUCTION


def test_guarded_promote_pass_without_evidence_refused(isolated_store):
    uri, client = isolated_store
    version, run_id = _register(client, passed=False)
    with pytest.raises(LifecycleRouteError, match="without blocking-criteria evidence"):
        router.guarded_promote(
            model_name=registry.MODEL_NAME, version=version, run_id=run_id,
            decision={"policy": "quality_gate_policy_v1_legacy", "passed": True},
        )
    assert client.search_model_versions(f"name='{registry.MODEL_NAME}'")[0].current_stage == STAGE_STAGING


def test_guarded_promote_policy_clash_refused(isolated_store):
    uri, client = isolated_store
    version, run_id = _register(client, passed=False)
    client.set_model_version_tag(registry.MODEL_NAME, str(version), router.POLICY_TAG, "quality_gate_policy_v1_legacy")
    with pytest.raises(LifecycleRouteError, match="policy clash"):
        router.guarded_promote(
            model_name=registry.MODEL_NAME, version=version, run_id=run_id,
            decision=_pass_decision(policy="quality_gate_policy_v2_revised"),
        )


def test_guarded_promote_idempotent_on_already_in_target(isolated_store):
    uri, client = isolated_store
    version, run_id = _register(client, passed=True)
    result = router.guarded_promote(
        model_name=registry.MODEL_NAME, version=version, run_id=run_id, decision=_pass_decision()
    )
    assert result["transitioned"] is False
    assert result["source_stage"] == result["target_stage"] == STAGE_PRODUCTION


def test_check_only_never_transitions_nor_writes(isolated_store):
    uri, client = isolated_store
    version, run_id = _register(client, passed=False)
    result = router.guarded_promote(
        model_name=registry.MODEL_NAME, version=version, run_id=run_id,
        decision=_pass_decision(), check_only=True,
    )
    assert result["check_only"] is True
    assert result["would_transition"] is True
    assert result["transitioned"] is False
    mv = client.search_model_versions(f"name='{registry.MODEL_NAME}'")[0]
    assert mv.current_stage == STAGE_STAGING
    assert router.POLICY_TAG not in (mv.tags or {})