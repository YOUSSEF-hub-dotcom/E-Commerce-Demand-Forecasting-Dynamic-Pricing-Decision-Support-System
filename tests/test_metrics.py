"""Unit tests for the Phase-2 metric suite (WAPE, MAE, RMSE, Asymmetric Loss).

Validates numeric correctness, the zero-actual guard, asymmetric weighting,
shape validation, and the aggregate ``describe_metrics`` helper.
"""

import numpy as np
import pytest

from src.utils.metrics import (
    asymmetric_loss,
    describe_metrics,
    mae,
    rmse,
    wape,
)


class TestWape:
    """Weighted Absolute Percentage Error."""

    def test_perfect_prediction_is_zero(self):
        actual = [10.0, 20.0, 30.0]
        predicted = [10.0, 20.0, 30.0]
        assert wape(actual, predicted) == pytest.approx(0.0)

    def test_known_error_value(self):
        # error = |10-9| + |20-18| + |30-27| = 1 + 2 + 3 = 6 ; total_actual = 60
        actual = [10.0, 20.0, 30.0]
        predicted = [9.0, 18.0, 27.0]
        assert wape(actual, predicted) == pytest.approx(6.0 / 60.0)

    def test_total_miss_is_one(self):
        assert wape([1.0, 1.0, 1.0], [0.0, 0.0, 0.0]) == pytest.approx(1.0)

    def test_zero_actual_demands_returns_zero(self):
        # No demand means any all-zero prediction is trivially perfect; must not
        # raise ZeroDivisionError.
        assert wape([0.0, 0.0, 0.0], [0.0, 0.0, 0.0]) == pytest.approx(0.0)

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError):
            wape([1.0, 2.0], [1.0])


class TestMaeRmse:
    """MAE and RMSE numerical checks."""

    def test_mae_known_value(self):
        assert mae([0.0, 0.0, 0.0], [1.0, 2.0, 3.0]) == pytest.approx(2.0)

    def test_rmse_known_value(self):
        # errors = 1, 2, 3 -> MSE = 14/3
        assert rmse([0.0, 0.0, 0.0], [1.0, 2.0, 3.0]) == pytest.approx(np.sqrt(14.0 / 3.0))

    def test_rmse_penalizes_tails_more_than_mae(self):
        actual = [0.0, 0.0, 0.0, 0.0, 0.0]
        smooth = [1.0, 1.0, 1.0, 1.0, 1.0]  # MAE 1.0, RMSE 1.0
        spiky = [0.0, 5.0, 0.0, 0.0, 0.0]   # MAE 1.0, RMSE sqrt(5) ~ 2.24
        # Same MAE, but spiky must yield higher RMSE.
        assert mae(actual, smooth) == pytest.approx(mae(actual, spiky))
        assert rmse(actual, spiky) > rmse(actual, smooth)

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError):
            mae([1.0], [1.0, 2.0])
        with pytest.raises(ValueError):
            rmse([1.0, 2.0], [1.0])


class TestAsymmetricLoss:
    """Cost-of-error loss with asymmetric under/over weighting."""

    def test_underforecast_weighed_more(self):
        # Under-forecast: actual 10, predicted 5 (stockout).
        under = asymmetric_loss([10.0], [5.0], under_weight=2.0, over_weight=1.0)
        over = asymmetric_loss([5.0], [10.0], under_weight=2.0, over_weight=1.0)
        assert under > over

    def test_equals_under_weight_times_mae(self):
        assert asymmetric_loss([10.0], [5.0], 2.0, 1.0) == pytest.approx(2.0 * mae([10.0], [5.0]))

    def test_symmetric_when_weights_equal(self):
        a = [1.0, 2.0, 3.0]
        p = [3.0, 2.0, 1.0]
        assert asymmetric_loss(a, p, 1.0, 1.0) == pytest.approx(mae(a, p))

    def test_nonpositive_weights_raise(self):
        with pytest.raises(ValueError):
            asymmetric_loss([1.0], [1.0], under_weight=0.0, over_weight=1.0)
        with pytest.raises(ValueError):
            asymmetric_loss([1.0], [1.0], under_weight=1.0, over_weight=-1.0)


class TestDescribeMetrics:
    """Aggregate metrics helper."""

    def test_returns_all_keys(self):
        result = describe_metrics([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
        assert set(result.keys()) == {"wape", "mae", "rmse", "asymmetric_loss_2_1"}
        assert all(isinstance(v, float) for v in result.values())