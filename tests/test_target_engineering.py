"""Unit tests for demand target engineering (weekly aggregation + dense matrix).

Covers aggregation correctness, business-status policy, dense zero-filling,
week monotonicity, and category handling on synthetic and real inputs.
"""

import numpy as np
import pandas as pd
import pytest

from src.data.data_processing import apply_status_policy
from src.data.target_engineering import build_demand_matrix, build_weekly_demand


def make_fact() -> pd.DataFrame:
    """Synthetic fact table fixture with two products across a few weeks."""
    orders = pd.DataFrame(
        {
            "order_id": [f"o{i}" for i in range(8)],
            "order_item_id": [1] * 8,
            "product_id": ["pA"] * 4 + ["pB"] * 4,
            "price": [10.0, 10.0, 12.0, 12.0, 5.0, 5.0, 7.0, 7.0],
            "freight_value": [2.0, 2.0, 3.0, 3.0, 1.0, 1.0, 1.5, 1.5],
            "order_status": ["delivered"] * 7 + ["delivered"],
            "order_purchase_timestamp": pd.to_datetime(
                [
                    "2017-01-02", "2017-01-02", "2017-01-09", "2017-01-09",
                    "2017-01-02", "2017-01-02", "2017-01-16", "2017-01-16",
                ]
            ),
        }
    )
    products = pd.DataFrame(
        {
            "product_id": ["pA", "pB"],
            "product_category_name": ["cat_x", "cat_y"],
        }
    )
    return orders.merge(products, on="product_id", how="left")


class TestBuildWeeklyDemand:
    """Sparse product-week aggregation."""

    def test_counts_and_price_means(self):
        demand = build_weekly_demand(make_fact())

        pa = demand[demand["product_id"] == "pA"]
        # pA: week1 (2 units @10) and week2 (2 units @12).
        assert pa["order_demand"].tolist() == [2, 2]
        assert pa["avg_price"].tolist() == [10.0, 12.0]
        assert pa["avg_freight"].tolist() == [2.0, 3.0]

        pb = demand[demand["product_id"] == "pB"]
        assert pb["order_demand"].tolist() == [2, 2]
        assert pb["avg_price"].tolist() == [5.0, 7.0]

    def test_week_column_is_period(self):
        demand = build_weekly_demand(make_fact())
        assert pd.api.types.is_datetime64_any_dtype(demand["week"]) is False
        assert isinstance(demand["week"].dtype, pd.PeriodDtype)

    def test_sorted_by_product_then_week(self):
        demand = build_weekly_demand(make_fact())
        assert demand.equals(demand.sort_values(["product_id", "week"]).reset_index(drop=True))

    def test_category_lookup_preserved(self):
        demand = build_weekly_demand(make_fact())
        assert set(demand["product_category_name"]) == {"cat_x", "cat_y"}

    def test_missing_timestamp_column_raises(self):
        bad = make_fact().drop(columns=["order_purchase_timestamp"])
        with pytest.raises(ValueError):
            build_weekly_demand(bad)


class TestStatusPolicy:
    """Economically-valid order filtering."""

    def test_canceled_and_unavailable_excluded(self):
        fact = make_fact()
        fact.loc[0, "order_status"] = "canceled"
        fact.loc[1, "order_status"] = "unavailable"
        filtered = apply_status_policy(fact)
        assert len(filtered) == len(fact) - 2
        assert "canceled" not in set(filtered["order_status"])
        assert "unavailable" not in set(filtered["order_status"])

    def test_valid_in_transit_orders_kept(self):
        fact = make_fact()
        fact.loc[0, "order_status"] = "shipped"
        fact.loc[1, "order_status"] = "invoiced"
        fact.loc[2, "order_status"] = "processing"
        filtered = apply_status_policy(fact)
        # All three are genuine demand and must remain.
        assert len(filtered) == len(fact)

    def test_missing_status_column_raises(self):
        with pytest.raises(ValueError):
            apply_status_policy(make_fact().drop(columns=["order_status"]))


class TestBuildDemandMatrix:
    """Dense product x week zero-filled matrix."""

    def test_zero_filled_between_first_and_last_week(self):
        demand = build_weekly_demand(make_fact())
        matrix = build_demand_matrix(demand)

        pa = matrix[matrix["product_id"] == "pA"]
        # 3 distinct weeks between 2017-01-01 and 2017-01-16 (W-SUN grid).
        assert pa["week"].nunique() == 3
        assert pa["order_demand"].sum() == 4
        assert pa["order_demand"].dtype == np.int64

    def test_no_sales_weeks_have_nan_price(self):
        demand = build_weekly_demand(make_fact())
        matrix = build_demand_matrix(demand)
        gap = matrix[(matrix["product_id"] == "pA") & (matrix["order_demand"] == 0)]
        assert len(gap) >= 1
        assert gap["avg_price"].isna().all()

    def test_demand_column_is_zero_filled_not_nan(self):
        demand = build_weekly_demand(make_fact())
        matrix = build_demand_matrix(demand)
        assert matrix["order_demand"].isna().sum() == 0

    def test_deterministic_row_order(self):
        demand = build_weekly_demand(make_fact())
        m1 = build_demand_matrix(demand)
        m2 = build_demand_matrix(demand)
        pd.testing.assert_frame_equal(m1, m2)