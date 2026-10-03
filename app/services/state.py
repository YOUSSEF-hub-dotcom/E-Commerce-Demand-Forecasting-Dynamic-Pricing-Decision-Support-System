"""Application runtime state: lazy, write-once loads behind a lock.

Owns every heavy runtime object (model, elasticity, feature frames, SHAP
explainer) and records per-component readiness that drives ``/ready`` and the
``MODEL_NOT_READY`` gate on business endpoints.

All feature engineering is delegated to the verified Part-3/Part-4/Part-5
modules (``src.*``) — nothing here re-implements lag/rolling/category logic.
"""

from __future__ import annotations

import logging
import threading
from datetime import date
from typing import Optional

import numpy as np
import pandas as pd

from app.config import Settings
from app.errors import (
    FEATURE_BUILD_FAILED,
    INVALID_WEEK,
    MODEL_NOT_READY,
    PRODUCT_NOT_FOUND,
    TEST_BOUNDARY_VIOLATION,
    WEEK_OUT_OF_SUPPORTED_RANGE,
    AppError,
)
from app.services import model_service

logger = logging.getLogger("deploy.state")

PRODUCT_COL = "product_id"
WEEK_COL = "week"
PRICE_COL = "avg_price"
CATEGORY_COL = "product_category_name"

READINESS_GATE = (
    "model_loaded",
    "feature_state",
    "elasticity_state",
    "shap_runtime",
    "warmed_up",
)


class AppState:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._lock = threading.RLock()

        # model
        self.model: Optional[object] = None
        self.estimator: Optional[object] = None
        self.estimator_source: Optional[str] = None
        self.schema: list[str] = []
        self.model_info: Optional[dict] = None

        # feature / market state
        self.matrix: Optional[pd.DataFrame] = None
        self.catalog: frozenset[str] = frozenset()
        self.matrix_weeks: frozenset[str] = frozenset()
        self.category_of: dict[str, str] = {}
        self.reference_week: str = settings.reference_week
        self._week_frames: dict[str, pd.DataFrame] = {}
        self._week_price_maps: dict[str, dict[str, float]] = {}

        # elasticity
        self.elasticity_meta: dict = {}
        self._elasticity_map: dict[str, float] = {}
        self._elasticity_status: dict[str, str] = {}
        self._elasticity_source: dict[str, str] = {}

        # explainability
        self.explainer: Optional[object] = None
        self.explainer_expected_log1p: Optional[float] = None

        # lifecycle
        self.warmup: Optional[dict] = None
        self.loaded: bool = False
        self._components: dict[str, tuple[str, str | None]] = {}

    # ------------------------------------------------------------------ ops

    def mark_component(self, name: str, status: str, details: str | None = None) -> None:
        self._components[name] = (status, details)

    def _component_status(self, name: str) -> str:
        return self._components.get(name, ("unknown", None))[0]

    def is_ready(self) -> bool:
        return all(self._component_status(n) == "ok" for n in READINESS_GATE)

    def component_snapshot(self) -> dict:
        out = {}
        for name in READINESS_GATE:
            status, details = self._components.get(name, ("unknown", None))
            out[name] = {"status": status, "details": details}
        db_status, db_details = self._components.get("database", ("unknown", None))
        out["database"] = {"status": db_status, "details": db_details}
        return out

    # --------------------------------------------------------------- loading

    def load_all(self) -> None:
        """Run every startup load step; record per-component readiness."""
        steps = (
            ("model_loaded", self._load_model),
            ("feature_state", self._load_feature_state),
            ("elasticity_state", self._load_elasticity_state),
            ("shap_runtime", self._load_shap),
            ("warmed_up", self._warmup),
        )
        for name, fn in steps:
            if self._component_status(name) == "ok":
                continue
            try:
                fn()
                self.mark_component(name, "ok")
            except Exception as exc:
                logger.exception("startup step %s failed", name)
                self.mark_component(name, "not_ready", str(exc))
        self.loaded = self.is_ready()

    def _load_model(self) -> None:
        payload = model_service.load_production(self.settings)
        self.model = payload["model"]
        self.estimator = payload["estimator"]
        self.estimator_source = payload["estimator_source"]
        self.schema = payload["schema"]
        self.model_info = payload["info"]

    def _load_feature_state(self) -> None:
        from src.optimization import pricing_data
        from src.optimization.demand_baseline import frozen_features_for_week

        self.matrix = pricing_data.load_matrix()
        self.catalog = frozenset(self.matrix[PRODUCT_COL].astype(str).unique())
        self.matrix_weeks = frozenset(self.matrix[WEEK_COL].astype(str).unique())
        self.category_of = (
            self.matrix.assign(_p=self.matrix[PRODUCT_COL].astype(str))
            .drop_duplicates("_p")[["_p", CATEGORY_COL]]
            .set_index("_p")[CATEGORY_COL]
            .astype(str)
            .to_dict()
        )
        if self.reference_week not in self.matrix_weeks:
            raise ValueError(f"reference week {self.reference_week} not in matrix weeks")
        from src.optimization.demand_baseline import schema_from_frame

        frame = frozen_features_for_week(None, self.reference_week, model=self.model)
        frame_schema = schema_from_frame(frame)
        if frame_schema != self.schema:
            # Same 93 features but a different order: the model's signature order is
            # authoritative for scoring; the frame's order is fine as long as the
            # feature sets match exactly.
            if set(frame_schema) != set(self.schema):
                raise ValueError(
                    "feature schema of the reference frame does not match the model schema"
                )
        with self._lock:
            self._week_frames[self.reference_week] = frame

    def _load_elasticity_state(self) -> None:
        from src.optimization import elasticity as elasticity_mod
        from src.optimization import pricing_data

        assert self.matrix is not None
        periods = pricing_data.split_periods(self.matrix)
        train, val = periods["train"], periods["val"]
        report = elasticity_mod.estimate(train, val)
        eff = report["effective"]
        keep = [
            "method",
            "identification_bounds",
            "sample",
            "pooled_product_fe",
            "pooled_product_week_fe",
            "val_diagnostics",
            "status_counts",
        ]
        self.elasticity_meta = {k: report[k] for k in keep if k in report}
        self._elasticity_map = {}
        self._elasticity_status = {}
        self._elasticity_source = {}
        for _, r in eff.iterrows():
            pid = str(r[PRODUCT_COL])
            self._elasticity_status[pid] = str(r["status"])
            self._elasticity_source[pid] = str(r["elasticity_source"])
            if pd.notna(r.get("elasticity")):
                self._elasticity_map[pid] = float(r["elasticity"])
        logger.info(
            "elasticity ready: %d products with effective elasticity",
            len(self._elasticity_map),
        )

    def _load_shap(self) -> None:
        import shap

        assert self.estimator is not None
        frame = self._week_frames[self.reference_week]
        n = len(frame)
        rng = np.random.RandomState(self.settings.shap_seed)
        size = min(self.settings.shap_background_size, n)
        bg_idx = rng.choice(n, size=size, replace=False)
        bg = frame.iloc[bg_idx][self.schema].astype("float64")
        self.explainer = shap.TreeExplainer(
            self.estimator, data=bg, feature_names=self.schema
        )
        self.explainer_expected_log1p = float(
            np.asarray(self.explainer.expected_value, dtype=float).ravel()[0]
        )

    def _warmup(self) -> None:
        frame = self._week_frames[self.reference_week]
        row = frame.iloc[[0]][self.schema].astype("float64")
        pred = predict_units(self.model, row)
        if not np.isfinite(pred):
            raise ValueError("warm-up prediction is not finite")
        self.warmup = {
            "product_id": str(frame.iloc[0][PRODUCT_COL]),
            "week": self.reference_week,
            "prediction": float(pred),
        }
        logger.info("warm-up prediction complete: %s -> %.4f", self.warmup["product_id"], pred)

    # ------------------------------------------------------------- validation

    def require_ready(self) -> None:
        if not self.is_ready():
            done = [n for n in READINESS_GATE if self._component_status(n) == "ok"]
            missing = [n for n in READINESS_GATE if n not in done]
            raise AppError(
                503,
                MODEL_NOT_READY,
                "model not ready: missing " + ", ".join(missing) + " (see /ready)",
            )

    def validate_product(self, product_id: str) -> str:
        if product_id not in self.catalog:
            raise AppError(
                422,
                PRODUCT_NOT_FOUND,
                f"product {product_id!r} is not in the product catalog",
            )
        return product_id

    def validate_week(self, week: date | str | None) -> str:
        if week is None:
            label = self.reference_week
        elif isinstance(week, date):
            label = week.isoformat()
        else:
            label = str(week)
        try:
            d = date.fromisoformat(label)
        except ValueError as exc:
            raise AppError(422, INVALID_WEEK, f"week {label!r} is not a valid ISO date") from exc
        if d.weekday() != 0:
            raise AppError(422, INVALID_WEEK, "week must be a Monday (week-start)")
        if label >= "2018-05-21":
            raise AppError(
                422,
                TEST_BOUNDARY_VIOLATION,
                f"week {label} is on/after the protected test boundary (2018-05-21); "
                "the service serves history weeks only",
            )
        if label not in self.matrix_weeks:
            raise AppError(
                422,
                WEEK_OUT_OF_SUPPORTED_RANGE,
                f"week {label} is not within the supported demand weeks",
            )
        return label

    # ------------------------------------------------------- cached heavy data

    def frame_for_week(self, week_label: str) -> pd.DataFrame:
        with self._lock:
            if week_label in self._week_frames:
                return self._week_frames[week_label]
            try:
                from src.optimization.demand_baseline import frozen_features_for_week

                frame = frozen_features_for_week(None, week_label, model=self.model)
            except ValueError as exc:
                raise AppError(503, FEATURE_BUILD_FAILED, f"feature build failed: {exc}") from exc
            self._week_frames[week_label] = frame
            return frame

    def price_at(self, week_label: str, product_id: str) -> Optional[float]:
        """Observed average price for the product in ``week_label`` (>0 only)."""
        with self._lock:
            if week_label not in self._week_price_maps:
                assert self.matrix is not None
                sub = self.matrix[self.matrix[WEEK_COL].astype(str) == week_label]
                sub = sub[sub[PRICE_COL].fillna(0) > 0]
                self._week_price_maps[week_label] = dict(
                    zip(sub[PRODUCT_COL].astype(str), sub[PRICE_COL].astype(float))
                )
            return self._week_price_maps[week_label].get(product_id)

    def elasticity(self, product_id: str) -> Optional[float]:
        return self._elasticity_map.get(product_id)

    def elasticity_status_of(self, product_id: str) -> Optional[str]:
        return self._elasticity_status.get(product_id)

    def elasticity_source_of(self, product_id: str) -> Optional[str]:
        return self._elasticity_source.get(product_id)


def predict_units(model, frame: pd.DataFrame) -> float:
    """Business-scale prediction for a single feature row (wrapper output = units)."""
    pred = model.predict(frame)
    if isinstance(pred, pd.DataFrame):
        value = pred["pred_demand"].to_numpy(dtype=float)[0]
    else:
        value = float(np.asarray(pred, dtype=float).ravel()[0])
    return value