"""Inference-log persistence tests (SQLite store, mirrors PostgreSQL schema)."""

from __future__ import annotations

import pytest
from sqlalchemy import select

from app.db.database import InferenceStore
from app.db.models import InferenceRecord
from app.db.repository import build_error_record, build_success_record, persist
from app.errors import PersistenceError


def test_create_schema_ping_insert_roundtrip(tmp_path):
    url = f"sqlite:///{tmp_path / 'inference.db'}"
    store = InferenceStore(url)
    store.create_schema()
    store.ping()

    rec = build_success_record(
        request_id="r1",
        endpoint="/predict",
        status_code=200,
        latency_ms=12.3,
        model_name="demand_forecasting_final",
        model_version=2,
        product_id="P1",
        week="2018-05-14",
        prediction=5.23,
    )
    store.insert(rec)

    assert store.engine is not None
    with store.engine.connect() as conn:
        rows = conn.execute(select(InferenceRecord)).all()
    assert len(rows) == 1
    row = rows[0]
    assert row.product_id == "P1"
    assert abs(row.latency_ms - 12.3) < 1e-6
    assert row.model_version == 2


def test_insert_error_record(tmp_path):
    url = f"sqlite:///{tmp_path / 'inference.db'}"
    store = InferenceStore(url)
    store.create_schema()
    rec = build_error_record(
        request_id="r2",
        endpoint="/predict",
        status_code=422,
        latency_ms=1.1,
        error_type="PRODUCT_NOT_FOUND",
        error_message="no such product",
        product_id="NOPE",
    )
    store.insert(rec)
    assert store.engine is not None
    with store.engine.connect() as conn:
        rows = conn.execute(select(InferenceRecord)).all()
    assert len(rows) == 1
    assert rows[0].error_type == "PRODUCT_NOT_FOUND"


def test_unreachable_store_raises_persistence_error():
    store = InferenceStore("sqlite:////definitely/missing/db.sqlite")
    with pytest.raises(PersistenceError):
        store.insert(build_success_record(
            request_id="x", endpoint="/predict", status_code=200,
            latency_ms=1.0,
        ))


def test_persist_helper_returns_false_on_failure():
    store = InferenceStore("sqlite:////definitely/missing/db.sqlite")
    ok = persist(store, build_success_record(
        request_id="x", endpoint="/predict", status_code=200, latency_ms=1.0,
    ))
    assert ok is False


def test_json_column_stores_pricing_result(tmp_path):
    url = f"sqlite:///{tmp_path / 'inference.db'}"
    store = InferenceStore(url)
    store.create_schema()
    rec = build_success_record(
        request_id="r3",
        endpoint="/pricing",
        status_code=200,
        latency_ms=5.0,
        product_id="P1",
        week="2018-05-14",
        pricing_result={"reason": "optimized", "objective": "profit"},
        elasticity_status="Reliable",
        elasticity_source="own",
        recommended_price=9.99,
        profit_computable=True,
    )
    store.insert(rec)
    assert store.engine is not None
    with store.engine.connect() as conn:
        rows = conn.execute(select(InferenceRecord)).all()
    row = rows[0]
    assert row.pricing_result["reason"] == "optimized"
    assert row.profit_computable is True
    assert row.recommended_price == 9.99