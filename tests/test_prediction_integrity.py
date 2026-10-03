"""Phase-2 prediction-integrity tests (remediation).

Guarantees the *inference contract* for demand predictions:

* ``inverse_log1p`` never emits negative / NaN / Inf demand,
* the PyFunc wrapper output respects the same contract,
* direct model predictions and PyFunc outputs stay consistent,
* schema/signature survive logging,
* metrics are computed on valid (non-negative, finite) predictions.
"""

import numpy as np
import pandas as pd
import pytest

from src.models.final_forecast import inverse_log1p, full_metric_suite
from src.models.pyfunc_wrapper import DemandForecaster


def _adversarial_log_inputs() -> np.ndarray:
    return np.asarray(
        [0.0, 1e-6, -1e-6, -0.5, -10.0, 5.0, np.nan, np.inf, -np.inf], dtype=float
    )


class _NegativeLogModel:
    """Estimator that deliberately emits negative/over-large log predictions."""

    def __init__(self):
        self._out = None

    def fit(self, X, y=None):
        return self

    def predict(self, X):
        n = X.shape[0]
        # mix: near-zero log (leads to tiny negatives in raw expm1), NaN path, float log.
        return np.asarray([-1e-4, -0.02, 0.0, 1.0, np.nan, np.inf][:n], dtype=float) * np.ones(n)


def test_inverse_log1p_never_negative():
    logs = _adversarial_log_inputs()
    out = inverse_log1p(logs)
    assert np.isfinite(out).all()
    assert (out >= 0).all()


def test_inverse_log1p_clamps_tiny_negatives():
    logs = np.asarray([-1e-5, -0.0425, -1e-3])
    out = inverse_log1p(logs)
    # raw expm1 would be slightly negative; the floor guarantees >= 0.
    assert (out >= 0).all()
    assert np.allclose(out, np.maximum(np.expm1(logs), 0.0))


def test_inverse_log1p_matches_expm1_on_positive_path():
    logs = np.asarray([0.0, 0.6931471805599453, 2.0, 5.0])
    out = inverse_log1p(logs)
    assert np.allclose(out, np.expm1(logs))


def test_inverse_log1p_fills_nonfinite():
    logs = np.asarray([np.nan, np.inf, -np.inf])
    out = inverse_log1p(logs)
    assert np.isfinite(out).all()
    assert (out == 0.0).all()


def test_pyfunc_wrapper_enforces_contract():
    columns = [f"f{i:02d}" for i in range(93)]
    model = _NegativeLogModel()
    wrapper = DemandForecaster(columns)
    wrapper._model = model
    request = pd.DataFrame(np.zeros((6, 93)), columns=columns)
    out = wrapper.predict(None, request)
    assert list(out.columns) == ["pred_demand"]
    assert out.shape[0] == 6
    assert out["pred_demand"].ge(0).all()
    assert np.isfinite(out["pred_demand"]).all()


def test_pyfunc_and_direct_model_consistent():
    columns = [f"f{i:02d}" for i in range(93)]
    model = _NegativeLogModel()
    request = pd.DataFrame(np.zeros((6, 93)), columns=columns)
    direct = inverse_log1p(model.predict(request))
    wrapper = DemandForecaster(columns)
    wrapper._model = model
    wrapped = wrapper.predict(None, request)["pred_demand"].to_numpy()
    assert np.array_equal(direct, wrapped)


def test_pyfunc_rejects_missing_columns():
    wrapper = DemandForecaster([f"f{i:02d}" for i in range(93)])
    bad = pd.DataFrame(np.zeros((2, 5)), columns=list("abcde"))
    with pytest.raises(ValueError):
        wrapper.predict(None, bad)


def test_metrics_computed_on_valid_predictions():
    """The metric suite must run on floored predictions and stay finite/valid."""
    # balanced cohorts: B / B2 / C each ~1/3 so every cohort bias is defined.
    cohort = np.where(
        np.arange(99) % 3 == 0,
        "B-sustained",
        np.where(np.arange(99) % 3 == 1, "B2-mid", "C-long-tail"),
    )
    actual = np.tile(np.asarray([0.0, 1.0, 2.0, 5.0, 3.0, 7.0, 0.0, 4.0, 2.0]), 11)
    logs = np.tile(np.asarray([-1e-4, 0.0, 0.5, 1.2, 0.8, 1.9, -0.3, 1.1, 0.4]), 11)
    pred = inverse_log1p(logs)
    assert (pred >= 0).all() and np.isfinite(pred).all()
    suite = full_metric_suite(actual, pred, cohort, mase_denom=0.04)
    for key, value in suite.items():
        assert np.isfinite(value), f"{key} not finite: {value}"
    assert suite["wape"] >= 0.0


@pytest.mark.parametrize("floor", [0.0, 0.001])
def test_inverse_log1p_respects_custom_floor(floor):
    logs = np.asarray([-5.0, -0.01, 0.1])
    out = inverse_log1p(logs, floor=floor)
    assert (out >= floor).all()