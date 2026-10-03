"""Part-4 pricing-data tests: period boundaries, cost input validation, leakage guard."""

import pandas as pd
import pytest

from src.optimization import pricing_data as pd_mod


def test_assign_period_boundaries():
    assert pd_mod.assign_period("2018-01-22") == "train"   # inclusive train max
    assert pd_mod.assign_period("2018-01-15") == "train"
    assert pd_mod.assign_period("2018-01-29") == "val"     # val start
    assert pd_mod.assign_period("2018-05-14") == "val"
    assert pd_mod.assign_period("2018-05-21") == "test"    # test start


def test_week_start_label_normalization():
    assert pd_mod.week_start_label("2018-01-29/2018-02-04") == "2018-01-29"
    assert pd_mod.week_start_label("2018-01-29") == "2018-01-29"


def test_load_matrix_assigns_periods(tmp_path):
    csv = tmp_path / "matrix.csv"
    pd.DataFrame(
        {
            "product_id": ["p1", "p1", "p1"],
            "week": ["2017-01-02/2017-01-08", "2018-02-05/2018-02-11", "2018-06-04/2018-06-10"],
            "order_demand": [1.0, 2.0, 3.0],
            "avg_price": [10.0, 11.0, 12.0],
            "avg_freight": [5.0, 6.0, 7.0],
            "product_category_name": ["c1", "c1", "c1"],
        }
    ).to_csv(csv, index=False)
    df = pd_mod.load_matrix(csv)
    assert list(df["part4_period"]) == ["train", "val", "test"]


def test_require_periods_raises_on_forbidden_period():
    df = pd.DataFrame({"part4_period": ["train", "test"]})
    with pytest.raises(AssertionError, match="forbidden period"):
        pd_mod.require_periods(df, allowed=("train",))
    pd_mod.require_periods(df[df["part4_period"] == "train"], allowed=("train",))


def test_load_costs_validation(tmp_path):
    good = tmp_path / "costs.csv"
    pd.DataFrame({"product_id": ["p1", "p2"], "cost": [5.5, 9.0]}).to_csv(good, index=False)
    assert pd_mod.load_costs(good) == {"p1": 5.5, "p2": 9.0}

    bad_cols = tmp_path / "bad_cols.csv"
    pd.DataFrame({"x": [1]}).to_csv(bad_cols, index=False)
    with pytest.raises(ValueError, match="must contain"):
        pd_mod.load_costs(bad_cols)

    neg = tmp_path / "neg.csv"
    pd.DataFrame({"product_id": ["p1"], "cost": [-1.0]}).to_csv(neg, index=False)
    with pytest.raises(ValueError, match="strictly positive"):
        pd_mod.load_costs(neg)

    dup = tmp_path / "dup.csv"
    pd.DataFrame({"product_id": ["p1", "p1"], "cost": [1.0, 2.0]}).to_csv(dup, index=False)
    with pytest.raises(ValueError, match="duplicate"):
        pd_mod.load_costs(dup)