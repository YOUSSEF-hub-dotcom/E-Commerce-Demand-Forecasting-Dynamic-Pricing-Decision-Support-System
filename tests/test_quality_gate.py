"""Part-3 quality-gate tests (Phase G).

Covers the automated multi-condition gate: PASS -> Production, FAIL ->
Staging, edge behaviors (NaN observed), integrity flags, and the documented
threshold policy.
"""

import numpy as np
import pytest

from src.models.quality_gate import evaluate_gate, gate_documentation_markdown


def ok_metrics() -> dict:
    """Deterministic metrics that satisfy every gate threshold."""
    return {
        "wape": 0.3980,
        "mae": 0.0235,
        "rmse": 0.2700,
        "mase": 0.6180,
        "bias_pct": -8.0,
        "asym21": 0.0370,
        "pos_wape": 0.3750,
        "B_wape": 0.5300,
        "B_bias_pct": -18.0,
        "B2_wape": 0.5600,
        "B2_bias_pct": -18.0,
        "C_wape": 0.3850,
    }


def ok_technical() -> dict:
    return {k: True for k in [
        "pred_integrity", "schema_ok", "model_load_ok", "pyfunc_predict_ok",
        "signature_ok", "artifacts_ok", "tests_ok",
    ]}


def test_gate_pass_promotes_to_production():
    report = evaluate_gate(ok_metrics(), ok_technical())
    assert report.passed is True
    assert report.promoted_stage == "Production"


def test_gate_fail_global_wape_goes_to_staging():
    metrics = ok_metrics()
    metrics["wape"] = 0.4500  # well beyond the 0.41 global budget
    report = evaluate_gate(metrics, ok_technical())
    assert report.passed is False
    assert report.promoted_stage == "Staging"
    assert "wape" in [o.key for o in report.failed_criteria()]


def test_gate_fail_b_bias_goes_to_staging():
    metrics = ok_metrics()
    metrics["B_bias_pct"] = -31.0  # old -30% defect regime
    report = evaluate_gate(metrics, ok_technical())
    assert report.passed is False
    assert report.promoted_stage == "Staging"
    assert "B_bias_pct" in [o.key for o in report.failed_criteria()]


def test_gate_fail_integrity_blocks_production():
    technical = ok_technical()
    technical["schema_ok"] = False  # frozen 93-col contract broken
    report = evaluate_gate(ok_metrics(), technical)
    assert report.passed is False
    assert report.promoted_stage == "Staging"
    assert "schema_ok" in [o.key for o in report.failed_criteria()]


def test_gate_fail_pyfunc_predict_blocks_production():
    technical = ok_technical()
    technical["pyfunc_predict_ok"] = False
    report = evaluate_gate(ok_metrics(), technical)
    assert report.passed is False
    assert "pyfunc_predict_ok" in [o.key for o in report.failed_criteria()]


def test_gate_nan_observed_is_a_failure():
    metrics = ok_metrics()
    metrics["B_bias_pct"] = float("nan")  # degenerate cohort -> cannot certify
    report = evaluate_gate(metrics, ok_technical())
    assert report.passed is False
    assert "B_bias_pct" in [o.key for o in report.failed_criteria()]


def test_gate_requires_all_technical_flags():
    technical = ok_technical()
    technical.pop("tests_ok")  # missing flag defaults to False
    report = evaluate_gate(ok_metrics(), technical)
    assert report.passed is False
    assert "tests_ok" in [o.key for o in report.failed_criteria()]


def test_gate_report_machine_readable():
    report = evaluate_gate(ok_metrics(), ok_technical())
    payload = report.to_dict()
    assert payload["passed"] is True
    assert payload["promoted_stage"] == "Production"
    assert len(payload["outcomes"]) == 19  # 12 numeric + 7 technical


def test_gate_thresholds_documented():
    """Every criterion must carry a justification and reference."""
    report = evaluate_gate(ok_metrics(), ok_technical())
    for outcome in report.outcomes:
        assert outcome.justification, f"{outcome.key} has no justification"
        if outcome.group == "final_test" and outcome.op != "bool":
            assert outcome.reference, f"{outcome.key} has no reference"


def test_gate_documentation_markdown_is_rendered():
    md = gate_documentation_markdown()
    assert "# Quality Gate Thresholds" in md
    assert "WAPE" in md
    assert "B_bias_pct" in md