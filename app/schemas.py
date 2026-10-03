"""External API input/output contracts.

The API is business-level only: it exposes a product ID, an ISO week and an
optional unit cost. The internal 93-feature engineering schema, the
composition of the feature set, LightGBM parameters and the rolling/lag
implementation details are never exposed here.
"""

from __future__ import annotations

import math
from datetime import date
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


def _clean_product_id(product_id: str) -> str:
    cleaned = product_id.strip()
    return cleaned


# ---------------------------------------------------------------------------
# Requests
# ---------------------------------------------------------------------------


class ProductParams(BaseModel):
    """Shared product/week inputs."""

    product_id: str = Field(..., min_length=1, max_length=64)
    week: Optional[date] = None

    model_config = ConfigDict(extra="forbid")

    @field_validator("product_id")
    @classmethod
    def _strip_product_id(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("product_id must not be blank")
        return cleaned


class PredictionRequest(ProductParams):
    """Predict demand for a product in a week (default: reference week)."""


class PricingRequest(ProductParams):
    """Recommend a price for a product in a week (default: reference week).

    ``cost`` is the per-unit landed cost, optional. When absent the service
    cannot compute profit and marks the recommendation as revenue-optimal
    only.
    """

    cost: Optional[float] = Field(None, gt=0)

    @field_validator("cost")
    @classmethod
    def _finite_positive_cost(cls, value: Optional[float]) -> Optional[float]:
        if value is not None and not math.isfinite(value):
            raise ValueError("cost must be a finite number")
        return value


class ExplanationRequest(ProductParams):
    """Explain a demand prediction for a product in a week."""


# ---------------------------------------------------------------------------
# Model / inference metadata embedded in every response
# ---------------------------------------------------------------------------


class ModelMeta(BaseModel):
    name: str
    version: int
    stage: str
    status: str
    run_id: str
    source: str


class InferenceMeta(BaseModel):
    request_id: str
    inference_logged: bool


# ---------------------------------------------------------------------------
# Responses
# ---------------------------------------------------------------------------


class PredictionResponse(BaseModel):
    product_id: str
    week: str
    predicted_demand: float
    model: ModelMeta
    inference: InferenceMeta


class ScenarioRow(BaseModel):
    multiplier: float
    price: float
    pred_demand: float
    revenue: float
    profit: Optional[float] = None


class GuardrailReport(BaseModel):
    margin_floor_applied: bool
    margin_floor_inactive_no_cost: bool
    ceiling_applied: bool
    floor_applied: bool
    smoothed: bool
    notes: list[str] = []


class PricingResponse(BaseModel):
    product_id: str
    week: str
    current_price: Optional[float] = None
    baseline_demand: Optional[float] = None
    category: Optional[str] = None
    elasticity_status: Optional[str] = None
    elasticity_source: Optional[str] = None
    elasticity: Optional[float] = None
    reason: str
    profit_computable: bool
    objective: Optional[str] = None
    optimal_price: Optional[float] = None
    optimal_multiplier: Optional[float] = None
    optimal_revenue: Optional[float] = None
    optimal_profit: Optional[float] = None
    recommended_price: Optional[float] = None
    extrapolation_required: Optional[bool] = None
    guardrails: Optional[GuardrailReport] = None
    scenarios: list[ScenarioRow] = []
    model: ModelMeta
    inference: InferenceMeta


class Contributor(BaseModel):
    feature: str
    feature_code: str
    contribution_log1p: float


class ExplanationAttribution(BaseModel):
    feature: str
    feature_code: str
    contribution_log1p: float
    contribution_demand_units: float


class ExplainResponse(BaseModel):
    product_id: str
    week: str
    predicted_demand_units: float
    base_log1p: float
    expected_value_log1p: float
    sum_shap_log1p: float
    additivity_residual_demand_units: float
    attribution_of: str
    disclaimer: str
    reconstruction_block: str
    top_positive: list[ExplanationAttribution]
    top_negative: list[ExplanationAttribution]
    feature_count: int
    model: ModelMeta
    inference: InferenceMeta


# ---------------------------------------------------------------------------
# Lifecycle + errors
# ---------------------------------------------------------------------------


class HealthResponse(BaseModel):
    status: Literal["ok"]
    app_env: str
    version: str
    time_utc: str


class ReadinessComponent(BaseModel):
    status: Literal["ok", "not_ready", "unknown"]
    details: Optional[str] = None


class ReadinessResponse(BaseModel):
    status: Literal["ready", "not_ready"]
    model: str
    model_version: Optional[int] = None
    components: dict[str, ReadinessComponent]
    time_utc: str


class ErrorResponse(BaseModel):
    detail: str
    code: str
    request_id: Optional[str] = None
    extra: dict[str, Any] = {}