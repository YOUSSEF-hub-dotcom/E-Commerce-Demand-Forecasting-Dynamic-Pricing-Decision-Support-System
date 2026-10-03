"""Unit tests for the Phase-3 feature engineering stack.

Validates lag semantics (weeks 1/2/4), rolling mean/std mathematics on a known
series, calendar derivation, the chronological window-restriction helper, and
the no-lookahead invariance of the Phase-3 feature set.
"""

import numpy as np
import pandas as pd
import pytest

from src.features.feature_engineering import (
    PHASE3_LAGS,
    add_calendar_features,
    build_phase3_feature_matrix,
    restrict_to_window,
)

# Synthetic weekly demand series for product pA.
DEMAND = [0.0, 5.0, 2.0, 10.0, 0.0, 8.0]
WEEKS = pd.period_range("2017-01-01", periods=6, freq="W")


def make_matrix() -> pd.DataFrame:
    """Dense demand matrix fixture for a single product."""
    return pd.DataFrame(
        {
            "product_id": ["pA"] * 6,
            "week": WEEKS,
            "order_demand": np.array(DEMAND, dtype="int64"),
            "avg_price": [np.nan, 10.0, 10.0, 11.0, 11.0, 12.0],
            "avg_freight": [np.nan, 2.0, 2.0, 3.0, 3.0, 2.0],
            "product_category_name": ["cat_x"] * 6,
        }
    )


class TestPhase3Lags:
    """Lag-1/2/4 semantics on a known series."""

    def test_exact_lag_values(self):
        feats = build_phase3_feature_matrix(make_matrix())
        assert list(feats["demand_lag_1"]) == [0.0, 0.0, 5.0, 2.0, 10.0, 0.0]
        assert list(feats["demand_lag_2"]) == [0.0, 0.0, 0.0, 5.0, 2.0, 10.0]
        # Pure warm-up needs at least 4 prior observations -> all 0 here.
        assert list(feats["demand_lag_4"]) == [0.0, 0.0, 0.0, 0.0, 0.0, 5.0]

    def test_lag_set_matches_phase3(self):
        feats = build_phase3_feature_matrix(make_matrix())
        lag_cols = [c for c in feats.columns if c.startswith("demand_lag_")]
        assert lag_cols == [f"demand_lag_{k}" for k in PHASE3_LAGS]


class TestPhase3Rolling:
    """Rolling mean/std computed strictly on past (shift-by-1) demand."""

    def test_rolling_mean_known(self):
        feats = build_phase3_feature_matrix(make_matrix())
        # Shifted series = [NaN, 0, 5, 2, 10, 0]; 4-wide mean of [0,5,2,10] = 4.25.
        assert feats["rolling_demand_mean_4w"].tolist() == pytest.approx(
            [0.0, 0.0, 0.0, 0.0, 4.25, 4.25]
        )

    def test_rolling_std_known(self):
        feats = build_phase3_feature_matrix(make_matrix())
        expected_std = float(np.std([0.0, 5.0, 2.0, 10.0], ddof=1))
        assert feats["rolling_demand_std_4w"].iloc[4] == pytest.approx(expected_std)
        # Last row same window (different order) -> identical std.
        assert feats["rolling_demand_std_4w"].iloc[5] == pytest.approx(expected_std)

    def test_rolling_uses_no_current_week(self):
        feats = build_phase3_feature_matrix(make_matrix())
        # Week 2 (index1, demand 5): rolling window must NOT include demand=5.
        assert feats["rolling_demand_mean_4w"].iloc[1] == 0.0


class TestCalendar:
    """Stateless calendar/seasonal derivations."""

    def test_known_week_attributes(self):
        df = pd.DataFrame({"week": pd.period_range("2017-01-01", periods=3, freq="W")})
        cal = add_calendar_features(df)
        # Week 2016-12-26/2017-01-01 ends on Sunday 2017-01-01 -> ISO year 2016, month 12.
        assert cal["week_of_year"].iloc[0] == 52
        assert cal["year"].iloc[0] == 2016
        assert cal["month"].iloc[0] == 12
        assert cal["quarter"].iloc[0] == 4
        # Following weeks land in Jan 2017.
        assert cal["year"].iloc[1] == 2017
        assert cal["month"].iloc[1] == 1
        assert (cal["quarter"].iloc[1:] == 1).all()


class TestRestrictToWindow:
    """Chronological cutoff helper (fold-boundary feature recomputation)."""

    def test_truncates_inclusive(self):
        matrix = build_phase3_feature_matrix(make_matrix())
        cutoff = WEEKS[3]
        truncated = restrict_to_window(matrix, cutoff)
        assert len(truncated) == 4
        assert truncated["week"].max() == cutoff

    def test_rejects_non_period_frame(self):
        df = pd.DataFrame({"week": pd.to_datetime(["2017-01-01"])})
        with pytest.raises(ValueError):
            restrict_to_window(df, "2017-01-15")


class TestPhase3NoLookahead:
    """Features at week t are invariant to the facts observed at week t."""

    def test_nulling_current_week_facts_does_not_change_features(self):
        baseline = build_phase3_feature_matrix(make_matrix())
        engineered = [
            "week_of_year", "month", "quarter", "year",
            "demand_lag_1", "demand_lag_2", "demand_lag_4",
            "rolling_demand_mean_4w", "rolling_demand_std_4w",
            "price_prev", "price_chg_1w", "freight_prev", "category_share_prev",
        ]
        for idx in [1, 3]:
            perturbed = make_matrix().copy()
            perturbed.loc[idx, ["order_demand", "avg_price", "avg_freight"]] = np.nan
            rebuilt = build_phase3_feature_matrix(perturbed)
            for col in engineered:
                assert np.isclose(baseline[col].iloc[idx], rebuilt[col].iloc[idx], equal_nan=True), col