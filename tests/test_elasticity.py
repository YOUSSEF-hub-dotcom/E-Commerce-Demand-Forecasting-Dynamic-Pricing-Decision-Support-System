"""Part-4 elasticity tests: log-log identification, classification, pooling."""

import numpy as np
import pandas as pd
import pytest

from src.optimization import elasticity as el

TRAIN_WEEKS = ["2016-09-05", "2016-09-12", "2016-09-19", "2016-09-26", "2016-10-03"]
VAL_WEEKS = ["2018-01-29", "2018-02-05"]


def _panel():
    rows = []
    # p1: elastic (own estimate expected Reliable)
    for w, p in zip(TRAIN_WEEKS, [8, 9, 10, 11, 12]):
        rows.append({"product_id": "p1", "week": w, "order_demand": 500 * (p ** -1.5),
                     "avg_price": p, "product_category_name": "catA", "part4_period": "train"})
    # p2: elastic with noise
    rng = np.random.default_rng(7)
    for w, p in zip(TRAIN_WEEKS, [10, 11, 12, 13, 14]):
        rows.append({"product_id": "p2", "week": w, "order_demand": 400 * (p ** -1.2) * (1 + 0.01 * rng.normal()),
                     "avg_price": p, "product_category_name": "catA", "part4_period": "train"})
    # p3: constant price within catA -> own Unsupported, cat pool fallback
    for w in TRAIN_WEEKS:
        rows.append({"product_id": "p3", "week": w, "order_demand": 100.0,
                     "avg_price": 15.0, "product_category_name": "catA", "part4_period": "train"})
    # p4: constant price, category with a single sale -> no usable pool
    rows.append({"product_id": "p4", "week": TRAIN_WEEKS[0], "order_demand": 40.0,
                 "avg_price": 20.0, "product_category_name": "catB", "part4_period": "train"})
    for w, p in zip(VAL_WEEKS, [9, 10]):
        rows.append({"product_id": "p1", "week": w, "order_demand": 500 * (p ** -1.5),
                     "avg_price": p, "product_category_name": "catA", "part4_period": "val"})
        rows.append({"product_id": "p2", "week": w, "order_demand": 400 * (p ** -1.2),
                     "avg_price": p, "product_category_name": "catA", "part4_period": "val"})
    df = pd.DataFrame(rows)
    for w, p in zip([10, 11], [13.0, 13.5]):
        df.loc[len(df)] = ["p4", w, 30.0, p, "catB", "val"]
    return df


def test_estimate_identification_and_classification():
    df = _panel()
    train = df[df["part4_period"] == "train"]
    val = df[df["part4_period"] == "val"]
    rep = el.estimate(train, val)

    pp = rep["per_product"].set_index("product_id")
    assert pp.loc["p1", "status"] == el.STATUS_RELIABLE
    assert pp.loc["p1", "b"] < 0
    assert pp.loc["p2", "status"] in (el.STATUS_RELIABLE, el.STATUS_WEAK)
    assert pp.loc["p3", "status"] == el.STATUS_UNSUPPORTED
    assert pp.loc["p4", "status"] == el.STATUS_UNSUPPORTED

    eff = rep["effective"].set_index("product_id")
    assert eff.loc["p1", "elasticity"] < 0
    assert eff.loc["p1", "elasticity_source"] == "own"
    assert eff.loc["p3", "elasticity"] == pytest.approx(rep["per_category"]["catA"]["b"])
    assert eff.loc["p3", "elasticity_source"] == "category_pooled"
    assert np.isnan(eff.loc["p4", "elasticity"])  # no own, no usable category pool
    assert eff.loc["p4", "elasticity_source"] == "none"

    assert rep["pooled_product_fe"]["b"] < 0
    assert rep["val_diagnostics"]["train_val_sign_match"] is True
    assert rep["status_counts"][el.STATUS_UNSUPPORTED] == 2


def test_classify_own_thresholds():
    row = pd.Series({"b": -1.5, "t": -4.0, "span": 1.2, "n_distinct": 4, "n_weeks": 6})
    assert el.classify_own(row) == el.STATUS_RELIABLE
    row["t"] = -0.5
    assert el.classify_own(row) == el.STATUS_WEAK
    row2 = row.copy()
    row2["n_distinct"] = 1
    assert el.classify_own(row2) == el.STATUS_UNSUPPORTED
    row3 = row.copy()
    row3["b"] = 0.5  # positive -> not elastic
    assert el.classify_own(row3) == el.STATUS_WEAK
    row4 = row.copy()
    row4["b"] = -5.0  # implausible magnitude
    assert el.classify_own(row4) == el.STATUS_WEAK


def test_pooled_ols_basic():
    df = _panel()
    train = df[df["part4_period"] == "train"]
    res = el.pooled_ols(train, fe_keys=("product_id",))
    assert res["b"] < 0
    assert res["n"] > 0
    assert isinstance(res["fe"], list) and res["fe"] == ["product_id"]