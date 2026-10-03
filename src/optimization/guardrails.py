"""Part-4 business guardrails enforcer (Phase 4.5).

Enforces operational constraints on top of the optimizer's recommended price:

* **Margin floor** — ``cost * (1 + min_margin_pct)``. Active **only when a real cost
  value is present**; without cost the rule is *inactive and flagged*, never invented.
* **Price ceiling** — ``current * (1 + max_increase_pct)`` (default change cap 20%,
  matching the Blueprint scenario range) and/or an explicit absolute ceiling.
* **Absolute price floor** — explicit lower bound when provided.
* **Smoothing** — exponential blend with the current price to prevent sudden spikes:
  ``final = alpha * bounded_optimum + (1 - alpha) * current``.

Every constraint is configurable (``GuardrailConfig``); defaults are transparent and
tested, and nothing approximates an unobserved cost.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields

from src.utils.logging_setup import get_logger

LOGGER = get_logger(__name__)


@dataclass(frozen=True)
class GuardrailConfig:
    """Explicitly-configurable pricing guardrails.

    ``min_margin_pct``:   minimum margin above cost, as a fraction (e.g. 0.10 = 10%).
                          Only binds when a cost value is supplied.
    ``max_increase_pct``: max allowed increase over the current price (fraction; 0.20 = 20%).
    ``price_floor_abs``:  explicit absolute price floor (None = no explicit floor).
    ``price_ceiling_abs``: explicit absolute price ceiling (None = none).
    ``smoothing_alpha``:  blend weight of the candidate price vs the current price
                          (1.0 = no smoothing, output the candidate; 0.5 = 50/50).
    """

    min_margin_pct: float = 0.0
    max_increase_pct: float = 0.20
    price_floor_abs: float | None = None
    price_ceiling_abs: float | None = None
    smoothing_alpha: float = 1.0

    def __post_init__(self) -> None:
        for f in fields(self):
            v = getattr(self, f.name)
            if isinstance(v, float) and not (v >= 0.0):
                raise ValueError(f"guardrails: {f.name} must be >= 0")
        if not (0.0 <= self.smoothing_alpha <= 1.0):
            raise ValueError("guardrails: smoothing_alpha must be within [0, 1]")


@dataclass(frozen=True)
class GuardrailOutcome:
    candidate_price: float
    recommended_price: float
    margin_floor_applied: bool
    margin_floor_inactive_no_cost: bool
    ceiling_applied: bool
    floor_applied: bool
    smoothed: bool
    notes: tuple[str, ...] = field(default_factory=tuple)


def _clamp(value: float, lo: float | None, hi: float | None) -> float:
    if lo is not None:
        value = max(value, lo)
    if hi is not None:
        value = min(value, hi)
    return value


def apply_guardrails(
    candidate_price: float,
    current_price: float,
    *,
    cost: float | None = None,
    cfg: GuardrailConfig | None = None,
) -> GuardrailOutcome:
    """Clamp and smooth a candidate price; returns the final recommended price."""
    cfg = cfg or GuardrailConfig()
    if not (current_price > 0):
        raise ValueError("guardrails: current_price must be > 0")
    if not (candidate_price > 0):
        raise ValueError("guardrails: candidate_price must be > 0")

    notes: list[str] = []
    margin_floor_applied = False
    margin_floor_inactive_no_cost = False

    applied = candidate_price

    floor = cfg.price_floor_abs
    if cost is not None:
        margin_floor = cost * (1.0 + cfg.min_margin_pct)
        floor = max(floor, margin_floor) if floor is not None else margin_floor
        margin_floor_applied = True
    else:
        margin_floor_inactive_no_cost = cfg.min_margin_pct > 0
        if margin_floor_inactive_no_cost:
            notes.append("margin floor INACTIVE: no cost supplied (not approximated)")

    ceiling = cfg.price_ceiling_abs
    if cfg.max_increase_pct > 0:
        inc_ceiling = current_price * (1.0 + cfg.max_increase_pct)
        ceiling = min(ceiling, inc_ceiling) if ceiling is not None else inc_ceiling

    before_floor, before_ceil = applied, applied
    applied = _clamp(applied, floor, ceiling)
    floor_applied = bool(floor is not None and before_floor < floor)
    ceiling_applied = bool(ceiling is not None and before_ceil > ceiling)

    alpha = cfg.smoothing_alpha
    smoothed = bool(alpha < 1.0)
    final = alpha * applied + (1.0 - alpha) * current_price
    final = round(final, 2)

    return GuardrailOutcome(
        candidate_price=candidate_price,
        recommended_price=final,
        margin_floor_applied=margin_floor_applied,
        margin_floor_inactive_no_cost=margin_floor_inactive_no_cost,
        ceiling_applied=ceiling_applied,
        floor_applied=floor_applied,
        smoothed=smoothed,
        notes=tuple(notes),
    )