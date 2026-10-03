"""End-to-end integration against the REAL Production model and dataset.

Skipped by default (heavy: model + matrix + reference frame + elasticity + SHAP
warm-up take several minutes). Run explicitly with:

    $env:RUN_DEPLOYMENT_INTEGRATION=1; python -m pytest tests/deployment/test_deployment_integration.py -q -s

This is the closest locally-run proxy for the docker smoke test.
"""

from __future__ import annotations

import os
import time

import pytest

RUN_KEY = "RUN_DEPLOYMENT_INTEGRATION"

pytestmark = pytest.mark.skipif(
    os.environ.get(RUN_KEY) != "1",
    reason=f"set {RUN_KEY}=1 to run the heavy real-model integration test",
)


@pytest.fixture(scope="module")
def real_environment(tmp_path_factory):
    from app.config import Settings
    from app.services.state import AppState

    settings = Settings(
        database_url=f"sqlite:///{tmp_path_factory.mktemp('i') / 'inference.db'}",
        mlflow_mlruns_dir="mlruns",
    )
    state = AppState(settings)
    t0 = time.perf_counter()
    state.load_all()
    elapsed = time.perf_counter() - t0
    assert state.is_ready(), f"state not ready: {state.component_snapshot()}"
    yield {"state": state, "settings": settings, "startup_seconds": round(elapsed, 1)}


@pytest.fixture(scope="module")
def client(real_environment):
    from fastapi.testclient import TestClient

    from app.main import create_app

    env = real_environment
    with TestClient(
        create_app(env["settings"], state=env["state"], load_on_startup=False)
    ) as c:
        yield c


def _pick_product(state):
    return sorted(state.catalog)[0]


def test_startup_elapsed_and_ready(real_environment):
    assert real_environment["startup_seconds"] < 900
    state = real_environment["state"]
    snap = state.component_snapshot()
    for k, v in snap.items():
        if k == "database":
            continue
        assert v["status"] == "ok", f"{k}: {v}"


def test_health_and_ready(client):
    assert client.get("/health").status_code == 200
    r = client.get("/ready")
    assert r.status_code == 200
    assert r.json()["status"] == "ready"


def test_real_predict(client, real_environment):
    pid = _pick_product(real_environment["state"])
    r = client.post("/predict", json={"product_id": pid})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["product_id"] == pid
    assert body["week"] == "2018-05-14"
    assert body["predicted_demand"] >= 0.0
    assert body["inference"]["inference_logged"] is True
    assert body["model"]["version"] == 2


def test_real_predict_other_week(client, real_environment):
    pid = _pick_product(real_environment["state"])
    r = client.post("/predict", json={"product_id": pid, "week": "2018-04-09"})
    assert r.status_code == 200, r.text
    assert r.json()["week"] == "2018-04-09"


def test_real_validation_errors(client):
    pid = client and "x"
    r1 = client.post("/predict", json={"product_id": "NO_SUCH_PRODUCT_ZZZ"})
    assert r1.status_code == 422 and r1.json()["code"] == "PRODUCT_NOT_FOUND"
    r2 = client.post("/predict", json={"product_id": "x", "week": "2018-05-21"})
    assert r2.status_code == 422 and r2.json()["code"] == "TEST_BOUNDARY_VIOLATION"
    r3 = client.post("/predict", json={"product_id": "x", "week": "2018-05-15"})
    assert r3.status_code == 422 and r3.json()["code"] == "INVALID_WEEK"


def test_real_pricing_with_cost(client, real_environment):
    state = real_environment["state"]
    pid = sorted(state.catalog)[0]
    r = client.post("/pricing", json={"product_id": pid, "cost": 8.0})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["reason"] in ("optimized", "PROFIT_NOT_COMPUTABLE (no cost supplied)", "NO_ELASTICITY")
    assert body["week"] == "2018-05-14"
    assert body["profit_computable"] in (True, False)
    assert body["inference"]["inference_logged"] is True


def test_real_explain(client, real_environment):
    pid = _pick_product(real_environment["state"])
    r = client.post("/explain", json={"product_id": pid})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["feature_count"] == 93
    assert body["predicted_demand_units"] >= 0.0
    assert abs(body["additivity_residual_demand_units"]) < 1e-6
    assert body["top_positive"]
    assert body["top_negative"]
    assert body["model"]["version"] == 2


def test_real_inference_rows_written(client, real_environment):
    # At least one success was logged during the previous real tests.
    from sqlalchemy import func, select

    from app.db.models import InferenceRecord

    state = real_environment["state"]
    store_url = real_environment["settings"].database_url
    from app.db.database import InferenceStore

    store = InferenceStore(store_url)
    store.create_schema()
    assert store.engine is not None
    with store.engine.connect() as conn:
        total = conn.execute(select(func.count()).select_from(InferenceRecord)).scalar()
    assert total >= 5