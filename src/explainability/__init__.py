"""Part-5 SHAP explainability package.

Explains the existing Production demand model (``models:/demand_forecasting_final/2``)
read-only. ``shap_analysis`` provides:

* canonical Production-model loading + schema verification;
* deterministic reference-week feature frames (Train/Validation only);
* Tree SHAP global/local analysis;
* artifact + report generation.
"""

from src.explainability.shap_analysis import (
    LAST_VAL_WEEK,
    MODEL_NAME,
    MODEL_URI,
    TEST_START_WEEK,
    analyze,
    feature_schema,
    model_info_from_registry,
    reference_frame,
    resolve_estimator,
    run_analysis,
)

__all__ = [
    "MODEL_NAME",
    "MODEL_URI",
    "TEST_START_WEEK",
    "LAST_VAL_WEEK",
    "model_info_from_registry",
    "feature_schema",
    "resolve_estimator",
    "reference_frame",
    "analyze",
    "run_analysis",
]