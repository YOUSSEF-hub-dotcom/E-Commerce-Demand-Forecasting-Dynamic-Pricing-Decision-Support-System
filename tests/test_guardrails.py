"""Part-4 guardrail tests: explicit bounds, cost gating, smoothing, no invented margins."""

import pytest

from src.optimization.guardrails import (
    GuardrailConfig,
    apply_guardrails,
)


def test_margin_floor_requires_cost():
    # min_margin requested but no cost -> rule inactive and flagged, never applied
    out = apply_guardrails(8.0, 10.0, cost=None, cfg=GuardrailConfig(min_margin_pct=0.10))
    assert out.margin_floor_inactive_no_cost is True
    assert out.margin_floor_applied is False
    assert out.recommended_price == pytest.approx(8.0)
    assert any("INACTIVE" in n for n in out.notes)


def test_margin_floor_with_cost():
    # cost=5, min_margin=20% -> floor = 6: candidate 5.5 must be clamped up
    out = apply_guardrails(5.5, 10.0, cost=5.0, cfg=GuardrailConfig(min_margin_pct=0.20))
    assert out.margin_floor_applied is True
    assert out.recommended_price == pytest.approx(6.0)


def test_ceiling_caps_increase():
    # current=10, max_increase=10% -> ceiling 11
    out = apply_guardrails(13.0, 10.0, cfg=GuardrailConfig(max_increase_pct=0.10))
    assert out.ceiling_applied is True
    assert out.recommended_price == pytest.approx(11.0)


def test_explicit_absolute_bounds():
    cfg = GuardrailConfig(price_floor_abs=6.0, price_ceiling_abs=12.0, max_increase_pct=0.0)
    out = apply_guardrails(4.0, 10.0, cfg=cfg)
    assert out.recommended_price == pytest.approx(6.0)
    out2 = apply_guardrails(15.0, 10.0, cfg=cfg)
    assert out2.recommended_price == pytest.approx(12.0)


def test_smoothing_blends_with_current():
    cfg = GuardrailConfig(smoothing_alpha=0.5)
    out = apply_guardrails(12.0, 10.0, cfg=cfg)
    assert out.smoothed is True
    assert out.recommended_price == pytest.approx(11.0)


def test_no_guardrails_default_is_transparent():
    out = apply_guardrails(12.0, 10.0, cfg=GuardrailConfig())
    # default max_increase 20% caps at 12
    assert out.recommended_price == pytest.approx(12.0)
    assert out.margin_floor_inactive_no_cost is False


def test_invalid_config_rejected():
    with pytest.raises(ValueError, match="smoothing_alpha"):
        GuardrailConfig(smoothing_alpha=1.5)
    with pytest.raises(ValueError, match="max_increase_pct"):
        GuardrailConfig(max_increase_pct=-0.1)