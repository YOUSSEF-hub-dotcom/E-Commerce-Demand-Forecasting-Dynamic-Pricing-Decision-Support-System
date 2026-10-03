"""Guarded promotion contract for the Part-3 lifecycle (automation infrastructure).

Closing the automation gap identified in ``reports/forecasting/part3_automation_audit``:
``register_and_promote`` trusted a caller-supplied ``passed`` boolean, so a human or a
buggy script could structurally promote a version that never produced a valid
quality-gate PASS record. This module provides the enforcement layer:

* **Deterministic routing** - ``route_stage`` maps a decision record to a stage
  (Production iff gate PASS, otherwise Staging): the exact PASS/FAIL contract
  already exercised by ``test_quality_gate`` / ``test_registry``.
* **Decision-record contract** - ``verify_decision_record`` requires a machine-readable
  gate record (``policy``, ``passed``, ``promoted_stage`` and, for a PASS claim, the
  actual blocking outcomes) and rejects bare ``passed=True`` claims with no criteria.
* **Provenance enforcement** - ``guarded_promote`` refuses any stage change whose
  submitted ``run_id`` does not match the registered model version's ``run_id``, and
  refuses to apply a policy other than the one already recorded on the version
  (recording it additively on first verified promotion).
* **Never weakens policy** - threshold values, the legacy/revised policy set, and
  historical runs/artifacts are untouched. Nothing here reads the protected test split.

This router is the standard entry point for NEW lifecycle automations. It does not
rewrite the historical V1/V2 decisions; those remain as recorded.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mlflow.tracking import MlflowClient

from src.utils.logging_setup import get_logger

LOGGER = get_logger(__name__)

# The frozen policy set (must stay identical to quality_gate.py).
VALID_POLICIES = ("quality_gate_policy_v1_legacy", "quality_gate_policy_v2_revised")

# Tag key that records the active policy on a model version.
POLICY_TAG = "gate_policy_version"

STAGE_PRODUCTION = "Production"
STAGE_STAGING = "Staging"


class LifecycleRouteError(Exception):
    """Raised when a promotion request violates the guarded contract."""


@dataclass(frozen=True)
class RoutePlan:
    """Normalized, verified promotion request."""

    policy: str
    passed: bool
    source_stage: str
    target_stage: str


def route_stage(passed: bool) -> str:
    """Deterministic PASS/FAIL routing: PASS -> Production, FAIL -> Staging."""
    return STAGE_PRODUCTION if passed else STAGE_STAGING


def verify_decision_record(decision: dict[str, Any]) -> dict[str, Any]:
    """Validate a machine-readable gate decision record.

    Accepted shapes are ``QualityGateReport.to_dict()`` (keys ``policy`` / ``passed`` /
    ``promoted_stage`` / ``outcomes`` / ``diagnostics``) and the gate-policy-revision
    fragments (``blocking_criteria`` instead of ``outcomes``). A ``PASS`` claim MUST be
    backed by a non-empty blocking-criteria list with every blocking outcome passing;
    anything less is an unverifiable promotion request.

    Returns the normalized record. Raises ``LifecycleRouteError`` on violation.
    """
    if not isinstance(decision, dict):
        raise LifecycleRouteError("lifecycle_router: decision record must be a dict")

    policy = decision.get("policy")
    if policy not in VALID_POLICIES:
        raise LifecycleRouteError(
            f"lifecycle_router: unknown or missing policy {policy!r}; "
            f"valid policies: {sorted(VALID_POLICIES)}"
        )

    passed_raw = decision.get("passed")
    if not isinstance(passed_raw, (bool, int)):
        raise LifecycleRouteError("lifecycle_router: decision['passed'] must be a boolean")
    passed = bool(passed_raw)

    promoted_stage = decision.get("promoted_stage", "")
    if promoted_stage not in ("", STAGE_PRODUCTION, STAGE_STAGING):
        raise LifecycleRouteError(
            f"lifecycle_router: invalid promoted_stage {promoted_stage!r}"
        )

    blocking = decision.get("outcomes", decision.get("blocking_criteria"))
    blocking = list(blocking) if blocking is not None else []

    if passed:
        if not blocking:
            raise LifecycleRouteError(
                "lifecycle_router: PASS claim without blocking-criteria evidence is "
                "not promotable (refusing unverifiable promotion)"
            )
        failed = [o for o in blocking if not bool(o.get("passed"))]
        if passed and failed:
            raise LifecycleRouteError(
                "lifecycle_router: PASS claim contradicts blocking outcomes "
                f"({[o.get('key') for o in failed]}); refusing production promotion"
            )
        expected = STAGE_PRODUCTION
    else:
        expected = STAGE_STAGING

    if promoted_stage and promoted_stage != expected:
        raise LifecycleRouteError(
            f"lifecycle_router: promoted_stage {promoted_stage!r} inconsistent with "
            f"passed={passed} (expected {expected!r})"
        )

    return {
        "policy": policy,
        "passed": passed,
        "promoted_stage": expected,
        "blocking_count": len(blocking),
    }


def _fetch_version(client: MlflowClient, model_name: str, version: str | int) -> Any:
    rows = [
        v
        for v in client.search_model_versions(f"name='{model_name}'")
        if str(v.version) == str(version)
    ]
    if not rows:
        raise LifecycleRouteError(
            f"lifecycle_router: no version {version} of '{model_name}' in the registry"
        )
    return rows[0]


def guarded_promote(
    *,
    model_name: str,
    version: str | int,
    run_id: str,
    decision: dict[str, Any],
    check_only: bool = False,
) -> dict[str, Any]:
    """Verify a promotion request against the registered version, then act (or only check).

    Contract enforced before any ``transition_model_version_stage`` call:

    1. decision record is valid (``verify_decision_record``);
    2. provenance: ``run_id`` must equal the registered version's ``run_id``;
    3. policy: the decision policy must match the policy already recorded on the
       version (``gate_policy_version`` tag). On first verified promotion the tag is
       recorded additively; a conflicting pre-recorded policy is refused;
    4. stage routing stays deterministic: PASS -> Production / FAIL -> Staging.

    With ``check_only=True`` nothing is transitioned and no tag is written; the plan is
    returned with ``would_transition`` describing the action a real call would take.
    """
    client = MlflowClient()
    mv = _fetch_version(client, model_name, version)

    verified = verify_decision_record(decision)
    policy, passed = verified["policy"], verified["passed"]

    version_run_id = mv.run_id
    if version_run_id != run_id:
        raise LifecycleRouteError(
            f"lifecycle_router: provenance mismatch - decision/run {run_id} does not "
            f"match version {version} run {version_run_id}"
        )

    existing_policy = dict(mv.tags or {}).get(POLICY_TAG)
    if existing_policy is not None and existing_policy != policy:
        raise LifecycleRouteError(
            f"lifecycle_router: policy clash - version records {existing_policy!r} but "
            f"request applies {policy!r}"
        )

    before = mv.current_stage
    target = route_stage(passed)
    plan = RoutePlan(
        policy=policy,
        passed=passed,
        source_stage=before,
        target_stage=target,
    )
    result = {
        "model_name": model_name,
        "version": str(version),
        "run_id": run_id,
        "policy": policy,
        "passed": passed,
        "source_stage": before,
        "target_stage": target,
        "would_transition": before != target,
    }

    if check_only:
        result["check_only"] = True
        result["transitioned"] = False
        LOGGER.info(
            "lifecycle_router (check-only): %s version %s -> %s.",
            model_name, version, target,
        )
        return result

    if existing_policy is None:
        client.set_model_version_tag(model_name, str(version), POLICY_TAG, policy)
        LOGGER.info("lifecycle_router: recorded policy tag %s on version %s.", policy, version)

    transitioned = False
    if before != target:
        client.transition_model_version_stage(
            name=model_name,
            version=str(version),
            stage=target,
            archive_existing_versions=(target == STAGE_PRODUCTION),
        )
        transitioned = True
        LOGGER.info(
            "lifecycle_router: version %s of '%s' %s -> %s.",
            version, model_name, before, target,
        )
    else:
        LOGGER.info(
            "lifecycle_router: version %s of '%s' already in %s; no transition.",
            version, model_name, target,
        )

    result["transitioned"] = transitioned
    result["stage"] = target
    return result