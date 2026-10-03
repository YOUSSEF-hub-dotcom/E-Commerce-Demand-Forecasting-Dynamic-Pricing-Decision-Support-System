"""Part-4 pricing-engine integration tests + anti-leakage guarantees."""

import joblib
import numpy as np
import pandas as pd
import pytest

import src.optimization.pricing_engine as engine
from src.optimization import pricing_data

TRAIN_WEEKS = ["2016-09-05", "2016-09-12", "2016-09-19", "2016-09-26", "2016-10-03"]
VAL_WEEKS = ["2018-01-29", "2018-02-05"]
REF_WEEK = "2018-02-05"
TEST_WEEK = "2018-05-21"
FEATURE_COLS = [f"f{i:02d}" for i in range(93)]


def _matrix_frame():
    rows = []
    for pid, prices in {
        "p1": [8, 9, 10, 11, 12],
        "p2": [10, 11, 12, 13, 14],
    }.items():
        for w, p in zip(TRAIN_WEEKS, prices):
            rows.append({"product_id": pid, "week": f"{w}/xx", "order_demand": 500 * (p ** -1.5),
                         "avg_price": p, "avg_freight": 5.0, "product_category_name": "catA"})
        for w, p in zip(VAL_WEEKS, [prices[-1] + 1, prices[-1] + 2]):
            rows.append({"product_id": pid, "week": f"{w}/xx", "order_demand": 500 * (p ** -1.5),
                         "avg_price": p, "avg_freight": 5.0, "product_category_name": "catA"})
        rows.append({"product_id": pid, "week": f"{TEST_WEEK}/xx", "order_demand": 100.0,
                     "avg_price": prices[-1], "avg_freight": 5.0, "product_category_name": "catA"})
    for w in TRAIN_WEEKS:  # p3 constant price in catA
        rows.append({"product_id": "p3", "week": f"{w}/xx", "order_demand": 100.0,
                     "avg_price": 15.0, "avg_freight": 5.0, "product_category_name": "catA"})
    for w in VAL_WEEKS:
        rows.append({"product_id": "p3", "week": f"{w}/xx", "order_demand": 100.0,
                     "avg_price": 15.0, "avg_freight": 5.0, "product_category_name": "catA"})
    rows.append({"product_id": "p3", "week": f"{TEST_WEEK}/xx", "order_demand": 100.0,
                 "avg_price": 15.0, "avg_freight": 5.0, "product_category_name": "catA"})
    # p4 constant price, catB single train sale -> no usable pool
    rows.append({"product_id": "p4", "week": f"{TRAIN_WEEKS[0]}/xx", "order_demand": 40.0,
                 "avg_price": 20.0, "avg_freight": 5.0, "product_category_name": "catB"})
    for w, p in zip(VAL_WEEKS, [13.0, 13.5]):
        rows.append({"product_id": "p4", "week": f"{w}/xx", "order_demand": 30.0,
                     "avg_price": p, "avg_freight": 5.0, "product_category_name": "catB"})
    return pd.DataFrame(rows)


def _prepared_val_frame():
    identity = ["product_id", "week", "order_demand", "demand_log1p", "sample_weight"]
    rows = []
    for pid in ("p1", "p2", "p3", "p4"):
        for w in VAL_WEEKS:
            row = {"product_id": pid, "week": w, "order_demand": 10.0, "demand_log1p": np.log1p(10.0),
                   "sample_weight": 1.0}
            row.update({c: 0.0 for c in FEATURE_COLS})
            rows.append(row)
    df = pd.DataFrame(rows)
    return df[identity + FEATURE_COLS]


@pytest.fixture(name="engine_env", scope="module")
def _engine_env(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("pricing_engine")
    matrix_csv = tmp / "matrix.csv"
    _matrix_frame().to_csv(matrix_csv, index=False)
    prepared_path = tmp / "prepared.joblib"
    joblib.dump({"val": _prepared_val_frame()}, prepared_path)
    return tmp, matrix_csv, prepared_path


class _StubModel:
    def __init__(self, val=12.0):
        self._val = val

    def predict(self, frame):
        return pd.DataFrame({"pred_demand": np.full(len(frame), self._val)})


def _patch(monkeypatch, engine_env, tmp_path):
    tmp, matrix_csv, prepared_path = engine_env
    monkeypatch.setattr(pricing_data, "MATRIX_PATH", matrix_csv)
    monkeypatch.setattr(engine, "ARTIFACT_DIR", tmp / "art")
    monkeypatch.setattr(engine, "load_production_model", lambda uri="x": _StubModel())
    return prepared_path


def test_engine_profit_not_computable_without_cost(monkeypatch, engine_env, tmp_path):
    prepared_path = _patch(monkeypatch, engine_env, tmp_path)
    report = engine.run_engine(prepared=prepared_path, reference_week=REF_WEEK, out_dir=tmp_path)
    s = report["summary"]
    assert report["test_split_accessed"] is False
    assert report["profit_computed"] is False
    assert s["products_scored"] == 4
    assert s["products_profit_computable"] == 0
    assert s["products_with_effective_elasticity"] == 3  # p1, p2, p3 (cat pool)

    csv_path = engine.ARTIFACT_DIR / "recommended_prices.csv"
    out = pd.read_csv(csv_path)
    by_id = out.set_index("product_id")
    assert by_id.loc["p1", "objective"] == "revenue"
    assert by_id.loc["p1", "reason"] == "PROFIT_NOT_COMPUTABLE (no cost supplied)"
    assert pd.notna(by_id.loc["p1", "recommended_price"])
    assert by_id.loc["p3", "elasticity_source"] == "category_pooled"
    assert by_id.loc["p4", "reason"] == "NO_ELASTICITY"
    assert pd.isna(by_id.loc["p4", "recommended_price"])


def test_engine_profit_computable_with_cost(monkeypatch, engine_env, tmp_path):
    tmp, matrix_csv, prepared_path = engine_env
    cost_csv = tmp / "costs.csv"
    pd.DataFrame({"product_id": ["p1", "p2", "p3", "p4"], "cost": [5.0, 6.0, 7.0, 8.0]}).to_csv(cost_csv, index=False)
    prepared_path = _patch(monkeypatch, engine_env, tmp_path)
    report = engine.run_engine(prepared=prepared_path, reference_week=REF_WEEK,
                               cost_path=cost_csv, out_dir=tmp_path)
    assert report["profit_computed"] is True
    s = report["summary"]
    assert s["products_profit_computable"] == 3
    out = pd.read_csv(engine.ARTIFACT_DIR / "recommended_prices.csv")
    assert (out.loc[out["profit_computable"], "objective"] == "profit").all()
    assert (out.loc[out["profit_computable"], "reason"] == "optimized").all()


def test_engine_rejects_test_period_reference_week(monkeypatch, engine_env, tmp_path):
    prepared_path = _patch(monkeypatch, engine_env, tmp_path)
    with pytest.raises(AssertionError, match="TEST"):
        engine.run_engine(prepared=prepared_path, reference_week=TEST_WEEK, out_dir=tmp_path)


def test_module_never_reads_test_split_source():
    import ast
    import inspect

    import src.optimization.pricing_data as pd_mod
    import src.optimization.elasticity as el_mod
    import src.optimization.simulation as sim_mod
    import src.optimization.optimizer as opt_mod
    import src.optimization.guardrails as grd_mod
    import src.optimization.demand_baseline as db_mod
    import src.optimization.pricing_engine as eng_mod

    for mod in (pd_mod, el_mod, sim_mod, opt_mod, grd_mod, db_mod, eng_mod):
        src = inspect.getsource(mod)
        tree = ast.parse(src)
        # Drop the module docstring (it legitimately mentions the protected path
        # in prose: "never reads data/splits/test.csv").
        if (
            tree.body
            and isinstance(tree.body[0], ast.Expr)
            and isinstance(tree.body[0].value, ast.Constant)
            and isinstance(tree.body[0].value.value, str)
        ):
            lines = src.splitlines()
            doc_end = tree.body[0].end_lineno
            body_lines = "\n".join(lines[doc_end:])
        else:
            body_lines = src
        assert "test.csv" not in body_lines
        assert "splits/test" not in body_lines.replace("data/splits/{train,val,test}.csv", "")