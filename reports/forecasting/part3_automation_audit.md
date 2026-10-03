# Part 3 — Quality-Gate & Promotion Lifecycle Automation Audit

**Scope:** The Part 3 production lifecycle (gate execution, PASS/FAIL routing, promotion). Pure audit; no historical decision, threshold, policy, run, or artifact was modified.
**Date:** 2026-09-24
**Status:** **PARTIALLY AUTOMATED** — the gate decision and PASS/FAIL promotion routing are fully automated *and regression-tested*; every lifecycle *trigger* (training, scoring, candidate selection, gate invocation) is a MANUAL console command with **no CI/CD, scheduler, or orchestration config present** in the repository.

---

## A1 — Gate execution

| Question | Verdict | Evidence |
|---|---|---|
| Is the quality gate auto-invoked after scoring? | **WITHIN a module: AUTOMATED.** The *trigger that starts each scoring module* is **MANUAL** (`python -m src.models.<module>`) — no scheduler/CI. | `finalize_lifecycle.py:337,384`; `finalize_v2.py:211,249`; `posthoc_holdout.py:376`; `gate_policy_revision.py:201-202,263`. |
| Is the gate deterministic? | **AUTOMATED (deterministic pure function).** Same metrics+technical → same outcome. | `quality_gate.py:evaluate_gate`; `tests/test_gate_policy_revision.py` (byte-identical legacy replay vs persisted, ×2 determinism run). |
| Are gate criteria versioned? | **AUTOMATED.** Two frozen policies: `quality_gate_policy_v1_legacy` (default, 12 numeric + 7 technical = 19 outcomes), `quality_gate_policy_v2_revised` (B_wape → diagnostics; 11 numeric + 7 technical = 18 blocking). Unknown policy → `ValueError`. | `quality_gate.py` (`POLICY_LEGACY`, `POLICY_REVISED`, `DIAGNOSTIC_METRICS`); `tests/test_quality_gate.py`. |
| Is the active policy recorded? | **AUTOMATED.** `gate_policy_version` tag on v2 run+version; `quality_gate_report.json` artifacts capture the decision dict. | Live registry tags (`gate_policy_version=quality_gate_policy_v2_revised`); `registry.py` logging. |
| Are thresholds frozen & documented? | **AUTOMATED.** Thresholds are module constants; `gate_documentation_markdown()` renders them per policy. | `quality_gate.py`; `tests/test_quality_gate.py` markdown-content asserts. |

## A2 — Promotion

| Question | Verdict | Evidence |
|---|---|---|
| Does gate PASS automatically → Production? | **AUTOMATED (when a module runs).** `register_and_promote(..., passed=True)` → `transition_model_version_stage(Production, archive_existing_versions=True)`. | `registry.py:249-255`; `tests/test_registry.py::test_register_and_promote_production`, `test_promote_production_archives_previous`. |
| Does gate FAIL automatically → non-Production (Staging)? | **AUTOMATED (when a module runs).** `passed=False` → `transition_model_version_stage(Staging, archive_existing_versions=False)`. | `registry.py:256-262`; `tests/test_registry.py::test_failed_candidate_stays_in_staging`. |
| Is promotion blocked when a blocking criterion fails? | **AUTOMATED.** `passed = gate.passed`; any failed blocking criterion forces `passed=False` → Staging, never Production. | `finalize_lifecycle.py:394`; `finalize_v2.py:257`; `tests/test_quality_gate.py` (wape-fail and B_bias-fail → Staging). |
| Is the promoted run/version identity preserved? | **AUTOMATED.** Version carries `run_id` tag; provenance pinned. | `registry.py` version tags; live registry (v1 run `7fc87198`, v2 run `5d729a85`). |
| Are historical runs/results never rewritten? | **AUTOMATED + tested.** Lifecycle updates are additive tags; only `gate_decision`/`promoted_stage` track current standing (documented). Gate-policy revision added purely additive tags. | `posthoc_holdout.py:update_lifecycle`; `gate_policy_revision.py:update_lifecycle`; `tests/test_gate_policy_revision.py`. |
| **GAP:** Is a human *structurally prevented* from promoting a version with no valid PASS decision? | **NOT PRESENT.** `register_and_promote` trusts a caller-supplied `passed` bool; nothing verifies it against a persisted gate record. **Closed in this work** via `lifecycle_router.py` (guarded promotion contract) + `tests/test_lifecycle_router.py`. | See new module + tests. |

## A3 — Human-intervention inventory

**AUTOMATED (code + regression tests)**
- Gate evaluation (deterministic, versioned policies).
- PASS/FAIL → Production/Staging routing and stage transition (incl. archiving previous Production).
- Run logging, artifact logging, version tag provenance, audit-trail JSON.
- PyFunc load + predict smoke verification before the gate decision is finalized.
- Once-guards (marker file + MLflow tag scan) preventing protected-test re-scoring.
- Policy replay + byte-identical verification (`gate_policy_revision.py`).
- Route enforcement now provided by `lifecycle_router.py`.

**MANUAL (still requires a human)**
- Running each phase: `python -m src.models.{optuna_tuning,finalize_lifecycle,finalize_v2,posthoc_holdout,gate_policy_revision}`.
- Candidate selection: target run/scoring decisions are module constants (e.g., posthoc `SOURCE_RUN_ID=5d729a85`, revision `MODEL_VERSION=2`).
- Triggering retraining/re-evaluation on new data.
- The policy-revision call itself (should remain an owner decision).
- No CI/CD, scheduler, DAG, or pre-commit automation exists anywhere in the repository (verified: no `.github/`, `Makefile`, `pyproject.toml`, `setup.cfg`, `pytest.ini`, cron/DAG config; no version control configured).

**PARTIALLY AUTOMATED (shaded zone, now closed)**
- Once a lifecycle module executes, gate→promotion is deterministic and enforced — but nothing before this work verified a human-provided promotion call against the persisted decision record.

## A4 — Test automation proof

| Requirement | Test evidence |
|---|---|
| PASS → Production | `test_registry.py::test_register_and_promote_production`, `test_promote_production_archives_previous`; `test_gate_policy_revision.py` live v2→Production. |
| FAIL → Staging | `test_registry.py::test_failed_candidate_stays_in_staging`; `test_quality_gate.py` (wape-fail / B_bias-fail → Staging). |
| No promotion when a blocking criterion fails | `test_quality_gate.py` (each numeric + technical bound, hard practice); `test_posthoc_holdout.py` (honest FAIL outcome, stage recorded). |
| Policy version preserved | `test_quality_gate.py` (legacy default 19 / revised 18+diagnostics; unknown policy rejected); `test_gate_policy_revision.py` (policy version tags). |
| Historical decisions reproducible | `test_gate_policy_revision.py` (deterministic replay byte-identical to persisted post-hoc FAIL). |
| Model version / run id preserved | `test_registry.py` (version.run_id), `test_posthoc_holdout.py` (still version 2, no v3). |
| No manual threshold modification possible | `test_gate_policy_revision.py` (unchanged thresholds, B_bias still hard), `tests/test_quality_gate.py` (threshold constants in markdown). |
| Guarded promotion contract (NEW) | `tests/test_lifecycle_router.py`: Production refused without a PASS record; provenance mismatch refused; stage override without policy refused; FAIL decision routes to Staging; persisted decision dict required. |

## Verdict & implemented gap-closure

**PARTIALLY AUTOMATED.** End-to-end lifecycle automation (retraining+scheduling+CI) is NOT demonstrable from the repository and is NOT claimed. The one code-level automation gap that could be closed without altering policy, thresholds, or history — an enforcement contract making unverifiable/manual promotion structurally impossible — is closed by:

- `src/models/lifecycle_router.py` — guarded promotion contract (decision-record provenance, policy-version requirement, run_id match, FAIL→Staging/never-Production enforcement; `--check` read-only mode).
- `tests/test_lifecycle_router.py` — contract regression tests.

Nothing in this audit modified: gate thresholds, active policies, historical MLflow runs/versions/tags, protected-test artifacts, or the Production decision.