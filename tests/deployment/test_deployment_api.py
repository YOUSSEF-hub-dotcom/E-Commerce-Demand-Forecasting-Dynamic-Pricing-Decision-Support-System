"""Endpoint behaviour tests against a stubbed (fully-loaded) application state."""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import create_app
from tests.deployment.conftest import FakeState

MONDAY_OK = "2018-03-12"
NON_MONDAY = "2018-03-13"
TEST_WEEK = "2018-05-21"
UNSUPPORTED = "2018-01-01"


def make_client(settings, state=None, load_on_startup=True):
    return TestClient(
        create_app(settings, state=state, load_on_startup=load_on_startup)
    )


def test_health(settings_tmp_db, fake_state):
    with make_client(settings_tmp_db, fake_state) as client:
        r = client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert "time_utc" in body


def test_ready_ok(settings_tmp_db, fake_state):
    with make_client(settings_tmp_db, fake_state) as client:
        r = client.get("/ready")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ready"
        assert body["model"] == "demand_forecasting_final"
        assert body["components"]["model_loaded"]["status"] == "ok"


def test_ready_not_when_unloaded(settings_tmp_db):
    state = FakeState()
    state.mark_component("model_loaded", "not_ready", "boom")
    with make_client(settings_tmp_db, state) as client:
        r = client.get("/ready")
        assert r.status_code == 503
        assert r.json()["status"] == "not_ready"


def test_predict_success(settings_tmp_db, fake_state):
    with make_client(settings_tmp_db, fake_state) as client:
        r = client.post("/predict", json={"product_id": "P1", "week": MONDAY_OK})
        assert r.status_code == 200
        body = r.json()
        assert body["product_id"] == "P1"
        assert body["week"] == MONDAY_OK
        assert body["predicted_demand"] == 5.23
        assert body["model"]["version"] == 2
        assert body["inference"]["inference_logged"] is True
        assert "request_id" in body["inference"]


def test_predict_default_week(settings_tmp_db, fake_state):
    with make_client(settings_tmp_db, fake_state) as client:
        r = client.post("/predict", json={"product_id": "P1"})
        assert r.status_code == 200
        assert r.json()["week"] == "2018-05-14"


def test_predict_unknown_product_422(settings_tmp_db, fake_state):
    with make_client(settings_tmp_db, fake_state) as client:
        r = client.post("/predict", json={"product_id": "NOPE"})
        assert r.status_code == 422
        assert r.json()["code"] == "PRODUCT_NOT_FOUND"


def test_predict_non_monday_422(settings_tmp_db, fake_state):
    with make_client(settings_tmp_db, fake_state) as client:
        r = client.post("/predict", json={"product_id": "P1", "week": NON_MONDAY})
        assert r.status_code == 422
        assert r.json()["code"] == "INVALID_WEEK"


def test_predict_test_boundary_422(settings_tmp_db, fake_state):
    with make_client(settings_tmp_db, fake_state) as client:
        r = client.post("/predict", json={"product_id": "P1", "week": TEST_WEEK})
        assert r.status_code == 422
        assert r.json()["code"] == "TEST_BOUNDARY_VIOLATION"


def test_predict_unsupported_week_422(settings_tmp_db, fake_state):
    with make_client(settings_tmp_db, fake_state) as client:
        r = client.post("/predict", json={"product_id": "P1", "week": UNSUPPORTED})
        assert r.status_code == 422
        assert r.json()["code"] == "WEEK_OUT_OF_SUPPORTED_RANGE"


def test_pricing_optimized_with_cost(settings_tmp_db, fake_state):
    with make_client(settings_tmp_db, fake_state) as client:
        r = client.post(
            "/pricing",
            json={"product_id": "P1", "week": MONDAY_OK, "cost": 8.0},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["reason"] == "optimized"
        assert body["profit_computable"] is True
        assert body["objective"] == "profit"
        assert body["current_price"] == 10.0
        assert body["recommended_price"] is not None
        assert 8.0 <= body["recommended_price"] <= 12.0
        assert len(body["scenarios"]) == 9
        assert body["model"]["version"] == 2


def test_pricing_profit_not_computable_without_cost(settings_tmp_db, fake_state):
    with make_client(settings_tmp_db, fake_state) as client:
        r = client.post("/pricing", json={"product_id": "P1", "week": MONDAY_OK})
        assert r.status_code == 200
        body = r.json()
        assert body["reason"] == "PROFIT_NOT_COMPUTABLE (no cost supplied)"
        assert body["profit_computable"] is False
        assert body["objective"] == "revenue"


def test_pricing_no_elasticity(settings_tmp_db, fake_state):
    with make_client(settings_tmp_db, fake_state) as client:
        r = client.post("/pricing", json={"product_id": "P3"})
        assert r.status_code == 200
        body = r.json()
        assert body["reason"] == "NO_ELASTICITY"
        assert body["profit_computable"] is False
        assert body["elasticity"] is None


def test_pricing_no_price_observation(settings_tmp_db, fake_state):
    with make_client(settings_tmp_db, fake_state) as client:
        # P2 has no price map entry for the reference week.
        r = client.post("/pricing", json={"product_id": "P2"})
        assert r.status_code == 200
        body = r.json()
        assert body["reason"] == "NO_PRICE_OBSERVATION"
        assert body["current_price"] is None


def test_pricing_invalid_cost(settings_tmp_db, fake_state):
    with make_client(settings_tmp_db, fake_state) as client:
        r = client.post(
            "/pricing", json={"product_id": "P1", "cost": 0.0}
        )
        assert r.status_code == 422
        assert r.json()["code"] == "VALIDATION_ERROR"


def test_explain_success(settings_tmp_db, fake_state):
    with make_client(settings_tmp_db, fake_state) as client:
        r = client.post("/explain", json={"product_id": "P1"})
        assert r.status_code == 200
        body = r.json()
        assert body["feature_count"] == 3
        assert body["top_positive"]
        assert body["top_negative"]
        assert abs(body["additivity_residual_demand_units"]) < 1e-6
        assert "expm1" in body["reconstruction_block"]
        assert "causality" in body["disclaimer"]
        # Business labels, not raw schema exposure.
        assert body["top_positive"][0]["feature"]
        assert all("feature_code" in c for c in body["top_positive"])


def test_unknown_endpoint_404(settings_tmp_db, fake_state):
    with make_client(settings_tmp_db, fake_state) as client:
        r = client.get("/does-not-exist")
        assert r.status_code == 404


def test_openapi_does_not_expose_feature_schema(settings_tmp_db, fake_state):
    with make_client(settings_tmp_db, fake_state) as client:
        spec = client.get("/openapi.json").json()
        text = str(spec)
        assert "demand_lag_1" not in text
        assert "rolling_demand" not in text
        assert "cat_" not in text
        paths = spec["paths"]
        assert set(paths) == {
            "/health",
            "/ready",
            "/predict",
            "/pricing",
            "/explain",
        }