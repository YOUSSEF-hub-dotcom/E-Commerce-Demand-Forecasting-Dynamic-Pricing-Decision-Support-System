"""Shared fixtures for the deployment implementation tests."""

from __future__ import annotations

import pandas as pd
import numpy as np
import pytest

from app.config import Settings
from app.services.state import READINESS_GATE, AppState


class StubWrapper:
    """Mimics the registered PyFunc wrapper predict (returns units)."""

    def predict(self, frame):
        return pd.DataFrame({"pred_demand": [5.23] * len(frame)})


class StubEstimator:
    """Mimics the underlying LGBM estimator (returns log1p values)."""

    def predict(self, frame):
        return np.full(len(frame), np.log1p(5.23))


class StubExplanation:
    def __init__(self, values):
        self.values = values


class StubExplainer:
    def __init__(self, values_1d):
        self._v = values_1d
        self.expected_value = np.array([0.5])

    def __call__(self, X, check_additivity=False):
        n = len(X)
        return StubExplanation(np.tile(self._v, (n, 1)))


def build_fake_state() -> AppState:
    """An AppState fully 'ready' but with stub model/explainer and a tiny frame."""
    state = AppState(Settings())
    state.model_info = {
        "name": "demand_forecasting_final",
        "version": 2,
        "stage": "Production",
        "status": "READY",
        "run_id": "run-test",
        "source": "models:/m-test",
    }
    state.schema = ["demand_lag_1", "price_prev", "cat_unknown"]
    state.model = StubWrapper()
    state.estimator = StubEstimator()
    state.explainer = StubExplainer(values_1d=[0.2, -0.1, 0.3])
    state.explainer_expected_log1p = 0.5

    state.reference_week = "2018-05-14"
    state.catalog = frozenset({"P1", "P2", "P3"})
    state.matrix_weeks = frozenset(
        {
            "2018-02-05", "2018-02-12", "2018-02-19", "2018-02-26",
            "2018-03-05", "2018-03-12", "2018-03-19", "2018-03-26",
            "2018-04-02", "2018-04-09", "2018-04-16", "2018-04-23",
            "2018-04-30", "2018-05-07", "2018-05-14",
        }
    )
    state.category_of = {"P1": "bed_bath_table", "P2": "bed_bath_table", "P3": "computers"}
    state._elasticity_map = {"P1": -1.2, "P2": -1.2}
    state._elasticity_status = {"P1": "Reliable", "P2": "Weak"}
    state._elasticity_source = {"P1": "own", "P2": "category_pooled"}
    state._price_maps = {"2018-05-14": {"P1": 10.0, "P3": 5.0}}

    rows = {
        "product_id": ["P1", "P2", "P3"],
        "week": ["2018-05-14"] * 3,
        "demand_lag_1": [4.0, 2.0, 1.0],
        "price_prev": [10.0, 8.0, 5.0],
        "cat_unknown": [0.0, 0.0, 1.0],
    }
    state._week_frames = {"2018-05-14": pd.DataFrame(rows)}
    for name in READINESS_GATE:
        state.mark_component(name, "ok")
    state.mark_component("database", "ok")
    state.loaded = True
    return state


class FakeState(AppState):
    """AppState subclass fully loaded with stubs, only disk/matrix methods overridden."""

    def __init__(self):
        src = build_fake_state()
        self.__dict__.update(src.__dict__)  # copy heavy-loaded attributes

    def frame_for_week(self, week_label):
        if week_label not in self._week_frames:
            # Cheap synthetic frame for any validated (pre-test) week.
            frame = self._week_frames[self.reference_week].copy()
            frame["week"] = week_label
            self._week_frames[week_label] = frame
        return self._week_frames[week_label]

    def price_at(self, week_label, product_id):
        maps = self._price_maps.get(week_label)
        if maps is None:
            maps = self._price_maps.get(self.reference_week, {})
        return maps.get(product_id)


@pytest.fixture
def fake_state():
    return FakeState()


@pytest.fixture
def settings_tmp_db(tmp_path):
    return Settings(database_url=f"sqlite:///{tmp_path / 'inference.db'}")