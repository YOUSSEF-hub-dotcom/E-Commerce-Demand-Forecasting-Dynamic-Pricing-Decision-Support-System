"""Error-handling and middleware behaviour tests."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.errors import MODEL_NOT_READY, AppError, business_rule
from app.main import create_app
from tests.deployment.conftest import FakeState


def make_client(settings, state):
    return TestClient(create_app(settings, state=state, load_on_startup=False))


def test_not_ready_gate_returns_503(settings_tmp_db):
    state = FakeState()
    state.is_ready = lambda: False
    with make_client(settings_tmp_db, state) as client:
        r = client.post("/predict", json={"product_id": "P1"})
        assert r.status_code == 503
        assert r.json()["code"] == MODEL_NOT_READY


def test_validation_error_struct(settings_tmp_db, fake_state):
    with make_client(settings_tmp_db, fake_state) as client:
        r = client.post("/predict", json={"product_id": ""})
        assert r.status_code == 422
        body = r.json()
        assert body["code"] == "VALIDATION_ERROR"
        assert "fields" in body["extra"]


def test_internal_error_never_leaks_stacktrace(settings_tmp_db):
    import asyncio

    from starlette.requests import Request as StarletteRequest

    from app.exception_handlers import unhandled_error_handler

    class Boom(Exception):
        pass

    # Starlette's ServerErrorMiddleware serves the registered Exception handler;
    # TestClient's transport re-raises/empties server errors, so assert the
    # handler contract directly (production path exercised in the docker smoke).
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/predict",
        "headers": [],
        "query_string": b"",
        "server": ("test", 80),
        "client": ("127.0.0.1", 50000),
        "scheme": "http",
        "state": {},
    }
    request = StarletteRequest(scope)
    response = asyncio.run(unhandled_error_handler(request, Boom("secret traceback")))
    assert response.status_code == 500
    body = response.body.decode()
    assert '"code":"INTERNAL_ERROR"' in body
    assert "secret traceback" not in body
    assert "Traceback" not in body


def test_request_id_echo(settings_tmp_db, fake_state):
    with make_client(settings_tmp_db, fake_state) as client:
        r = client.post(
            "/predict",
            json={"product_id": "P1"},
            headers={"X-Request-ID": "my-trace-id"},
        )
        assert r.json()["inference"]["request_id"] == "my-trace-id"
        assert r.headers["X-Request-ID"] == "my-trace-id"


def test_payload_limit_rejects_oversize(settings_tmp_db, fake_state):
    settings = Settings(
        database_url=settings_tmp_db.database_url, max_payload_bytes=32
    )
    with make_client(settings, fake_state) as client:
        r = client.post(
            "/predict", json={"_": "x" * 1000}
        )
        assert r.status_code == 413
        assert r.json()["code"] == "PAYLOAD_TOO_LARGE"


def test_rate_limit_default_off(settings_tmp_db, fake_state):
    with make_client(settings_tmp_db, fake_state) as client:
        for _ in range(5):
            r = client.post("/predict", json={"product_id": "P1"})
            assert r.status_code == 200


def test_rate_limit_when_enabled(settings_tmp_db):
    from fastapi.testclient import TestClient as TC

    settings = Settings(
        database_url=settings_tmp_db.database_url,
        rate_limit_enabled=True,
        rate_limit_max_requests=3,
    )
    state = FakeState()
    with TC(create_app(settings, state=state, load_on_startup=False)) as client:
        for _ in range(3):
            assert client.post("/predict", json={"product_id": "P1"}).status_code == 200
        r = client.post("/predict", json={"product_id": "P1"})
        assert r.status_code == 429
        assert r.json()["code"] == "RATE_LIMITED"


def test_persistence_failure_does_not_fail_response(settings_tmp_db, fake_state):
    """When the inference DB is unreachable the endpoint still succeeds."""
    settings = Settings(database_url="sqlite:////definitely/missing/path/db.sqlite")
    with make_client(settings, fake_state) as client:
        r = client.post("/predict", json={"product_id": "P1"})
        assert r.status_code == 200
        assert r.json()["inference"]["inference_logged"] is False


def test_app_error_to_response_model_shape(settings_tmp_db, fake_state):
    exc = business_rule(422, "INVALID_WEEK", "not a monday", reason="x")
    assert isinstance(exc, AppError)
    assert exc.code == "INVALID_WEEK"