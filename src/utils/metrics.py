"""Forecast evaluation metrics for the Demand Forecasting Engine.

Implements the Phase-2 metric suite defined in PROJECT_BLUEPRINT.md Part 2 /
Project Map evaluation framework:

* ``wape``              - WAPE, WAPE = |actual - predicted|_sum / actual_sum.
                          Scale-relative and robust to the heavy right tail of
                          the order_demand target. PRIMARY metric.
* ``mae``               - mean absolute error (interpretable, units demanded).
* ``rmse``              - root mean squared error (penalizes heavy tails).
* ``asymmetric_loss``   - cost-of-error loss: under-forecasts (stockouts) are
                          penalized more heavily than over-forecasts (holding
                          cost), per the asymmetric Cost-of-Error framework.

All functions accept array-likes and are validated to share shape and length;
they raise ``ValueError`` on mismatch so silent averaging bugs cannot occur.
"""

import numpy as np

from src.utils.logging_setup import get_logger

LOGGER = get_logger(__name__)


def _validate_1d(actual, predicted):
    """Cast inputs to equal-length 1D float arrays, validating their shapes.

    Parameters
    ----------
    actual : array-like
        Ground-truth observations.
    predicted : array-like
        Model predictions.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Flattened float arrays of identical shape.

    Raises
    ------
    ValueError
        If the arrays do not share the same total length.
    """
    a = np.asarray(actual, dtype=float).ravel()
    p = np.asarray(predicted, dtype=float).ravel()
    if a.shape != p.shape:
        raise ValueError(
            f"actual and predicted must match in shape, got {np.shape(actual)} vs {np.shape(predicted)}"
        )
    return a, p


def wape(actual, predicted) -> float:
    """Weighted Absolute Percentage Error.

    Defined as ``sum(|actual - predicted|) / sum(actual)``. Returns 0.0 when
    total actual demand is zero (a no-demand period is trivially perfectly
    predicted by any zero prediction), which also guards the zero-division.

    Parameters
    ----------
    actual : array-like
        Ground-truth demand values.
    predicted : array-like
        Predicted demand values.

    Returns
    -------
    float
        WAPE in [0, inf). Lower is better.

    Raises
    ------
    ValueError
        If input arrays have mismatched shapes.
    """
    a, p = _validate_1d(actual, predicted)
    denominator = a.sum()
    if denominator == 0.0:
        LOGGER.warning("wape: total actual demand is 0; returning 0.0 (perfect agreement).")
        return 0.0
    return float(np.abs(a - p).sum() / denominator)


def mae(actual, predicted) -> float:
    """Mean Absolute Error.

    Parameters
    ----------
    actual : array-like
        Ground-truth demand values.
    predicted : array-like
        Predicted demand values.

    Returns
    -------
    float
        MAE in units of demand. Lower is better.
    """
    a, p = _validate_1d(actual, predicted)
    return float(np.mean(np.abs(a - p)))


def rmse(actual, predicted) -> float:
    """Root Mean Squared Error.

    Parameters
    ----------
    actual : array-like
        Ground-truth demand values.
    predicted : array-like
        Predicted demand values.

    Returns
    -------
    float
        RMSE in units of demand. Lower is better; heavier tail penalty than MAE.
    """
    a, p = _validate_1d(actual, predicted)
    return float(np.sqrt(np.mean((a - p) ** 2)))


def asymmetric_loss(
    actual, predicted, under_weight: float = 2.0, over_weight: float = 1.0
) -> float:
    """Asymmetric Cost-of-Error loss.

    Penalizes under-forecasts (``actual > predicted``, stockouts / lost sales)
    with ``under_weight`` and over-forecasts (surplus, holding cost) with
    ``over_weight``. The default ratio 2:1 encodes the business rule that a
    stockout is twice as costly as holding surplus inventory.

    Parameters
    ----------
    actual : array-like
        Ground-truth demand values.
    predicted : array-like
        Predicted demand values.
    under_weight : float, optional
        Multiplier applied to under-forecasts (default 2.0).
    over_weight : float, optional
        Multiplier applied to over-forecasts (default 1.0).

    Returns
    -------
    float
        Mean weighted absolute error.

    Raises
    ------
    ValueError
        If either weight is non-positive or inputs are shape-mismatched.
    """
    if under_weight <= 0 or over_weight <= 0:
        raise ValueError("asymmetric_loss: weights must be strictly positive.")
    a, p = _validate_1d(actual, predicted)
    error = a - p  # positive error => under-forecast
    weights = np.where(error > 0.0, under_weight, over_weight)
    return float(np.mean(weights * np.abs(error)))


def describe_metrics(actual, predicted) -> dict:
    """Compute the full Phase-2 metric suite in one call.

    Convenience aggregator used by notebooks and evaluation scripts.

    Parameters
    ----------
    actual : array-like
        Ground-truth demand values.
    predicted : array-like
        Predicted demand values.

    Returns
    -------
    dict
        Mapping of metric name to value for WAPE, MAE, RMSE and the
        asymmetric loss at the default 2:1 ratio.
    """
    return {
        "wape": wape(actual, predicted),
        "mae": mae(actual, predicted),
        "rmse": rmse(actual, predicted),
        "asymmetric_loss_2_1": asymmetric_loss(actual, predicted),
    }