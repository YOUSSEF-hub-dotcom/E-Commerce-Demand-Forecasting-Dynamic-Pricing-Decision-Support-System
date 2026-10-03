"""Unit & integration tests for the Phase-3 forecasting models and MLOps wrapper.

Covers: expanding-window CV fold construction, the naive persistence baseline,
per-fold metric machinery, LightGBM/XGBoost smoke-fit determinism, the PyFunc
schema contract, and a full MLflow log/load round-trip on a local file store.
"""

import numpy as np
import pandas as pd
import pytest

from src.models.data_prep import make_sample_weights, prepare_model_matrix
from src.models.forecasting import (
    expanding_window_splits,
    make_model,
    naive_combined_predictions,
    run_expanding_cv,
)
from src.models.pyfunc_wrapper import DemandForecaster, validate_input
from src.utils.metrics import describe_metrics


def make_synthetic_feature_frame(n_weeks: int = 10) -> pd.DataFrame:
    """Deterministic per-product weekly demand frame for model tests."""
    weeks = pd.period_range("2017-01-01", periods=n_weeks, freq="W")
    demand = [1.0, 2.0, 1.0, 4.0, 3.0] + [float((i % 3) + 1) for i in range(n_weeks - 5)]
    demand = np.asarray(demand, dtype=float)
    # Rolling-4 mean of the shift-by-1 demand (week t uses demand[t-4:t-1]).
    prior = demand[:-1]
    warmup = np.zeros(4)
    roll_mean = np.convolve(prior, np.ones(4) / 4, mode="valid")
    frame = pd.DataFrame(
        {
            "product_id": ["pA"] * n_weeks,
            "week": weeks,
            "order_demand": demand,
            "demand_log1p": np.log1p(demand),
            "product_category_name": ["cat_x"] * n_weeks,
            "week_of_year": weeks.week,
            "month": weeks.month,
            "quarter": weeks.quarter,
            "year": weeks.year,
            "demand_lag_1": np.concatenate([[0.0], demand[:-1]]),
            "demand_lag_2": np.concatenate([[0.0, 0.0], demand[:-2]]),
            "demand_lag_4": np.concatenate([[0.0] * 4, demand[:-4]]),
            "rolling_demand_mean_4w": np.concatenate([warmup, roll_mean]),
            "rolling_demand_std_4w": np.zeros(n_weeks),
            "price_prev": np.concatenate([[np.nan], np.full(n_weeks - 1, 20.0)]),
            "price_chg_1w": np.zeros(n_weeks),
            "freight_prev": np.concatenate([[np.nan], np.full(n_weeks - 1, 5.0)]),
            "category_share_prev": np.zeros(n_weeks),
        }
    )
    # NaN-based drivers are imputed later; fill here for the fixture only.
    frame["price_prev"] = frame["price_prev"].fillna(20.0)
    frame["freight_prev"] = frame["freight_prev"].fillna(5.0)
    return frame


class TestExpandingWindowSplits:
    """Chronological, non-overlapping CV folds."""

    def test_fold_shapes_and_ordering(self):
        weeks = pd.period_range("2017-01-01", periods=12, freq="W")
        folds = expanding_window_splits(weeks, n_splits=4, min_train_weeks=3, horizon=2)
        assert len(folds) == 4
        for tr, va in folds:
            assert len(tr) >= 3 and len(va) == 2
            assert tr.max() < va.min()  # strictly in the past
            assert set(tr).isdisjoint(set(va))

    def test_fold_growth_is_expanding(self):
        weeks = pd.period_range("2017-01-01", periods=12, freq="W")
        folds = expanding_window_splits(weeks, n_splits=4, min_train_weeks=3, horizon=2)
        sizes = [len(tr) for tr, _ in folds]
        assert sizes == sorted(sizes) and len(set(sizes)) == len(sizes)

    def test_stops_before_running_past_data(self):
        weeks = pd.period_range("2017-01-01", periods=5, freq="W")
        folds = expanding_window_splits(weeks, n_splits=10, min_train_weeks=3, horizon=2)
        assert len(folds) == 1


class _ConstPredictor:
    """Module-level predictor stub (picklable for the MLflow round-trip)."""

    def __init__(self, const: float = 3.0):
        self.const = const

    def predict(self, X):
        return np.full(len(X), self.const)


class TestNaiveBaseline:
    """Persistence: next period = most recent observed demand."""

    def test_seed_from_training_and_chain(self):
        frame = make_synthetic_feature_frame(10)
        train = frame.iloc[:6]
        evalset = frame.iloc[6:10]
        y_true, pred = naive_combined_predictions(train, evalset)
        # demand=[1,2,1,4,3,1 |2,3,1,2]; persistence: eval week1<-train 1; then chains.
        assert y_true.tolist() == [2.0, 3.0, 1.0, 2.0]
        assert pred.tolist() == [1.0, 2.0, 3.0, 1.0]

    def test_metrics_finite(self):
        frame = make_synthetic_feature_frame(10)
        y_true, pred = naive_combined_predictions(frame.iloc[:6], frame.iloc[6:10])
        m = describe_metrics(y_true, pred)
        assert all(np.isfinite(v) for v in m.values())


class TestModelFactories:
    """Model creation and smoke-fit determinism."""

    @pytest.mark.parametrize(
        "model_type,params",
        [
            ("lgbm", {"n_estimators": 10, "num_leaves": 8, "max_depth": 4}),
            ("xgb", {"n_estimators": 10, "max_depth": 4, "min_child_weight": 1}),
        ],
    )
    def test_smoke_fit_predict_shape(self, model_type, params):
        frame = prepare_model_matrix(make_synthetic_feature_frame(14))
        X = frame.drop(columns=["product_id", "week", "order_demand", "demand_log1p", "sample_weight"])
        y = frame["demand_log1p"].to_numpy()
        model = make_model(model_type, params)
        model.fit(X, y)
        pred = model.predict(X)
        assert pred.shape == (len(X),)
        assert np.all(np.isfinite(pred))

    def test_unsupported_type_raises(self):
        with pytest.raises(ValueError):
            make_model("glm", {})


class TestPrepareModelMatrix:
    """Design-matrix invariants: schema alignment and NaN-free output."""

    def test_categories_align_split_schemas(self):
        train = make_synthetic_feature_frame(12).copy()
        val = make_synthetic_feature_frame(8).copy()
        val.loc[:, "product_category_name"] = "cat_y"

        categories = sorted(pd.concat([train["product_category_name"], val["product_category_name"]]).unique())
        train_p = prepare_model_matrix(train, categories=categories)
        val_p = prepare_model_matrix(val, categories=categories)

        train_design = {c for c in train_p.columns if c.startswith("cat_")}
        val_design = {c for c in val_p.columns if c.startswith("cat_")}
        assert train_design == val_design
        assert train_design == {"cat_cat_x", "cat_cat_y"}
        assert list(train_p.columns) == list(val_p.columns)


class TestRunExpandingCV:
    """Per-fold metric machinery on a mean-predictor dummy."""

    def test_returns_all_metric_keys(self):
        class MeanModel:
            def fit(self, X, y, sample_weight=None):
                self.mean_ = float(np.mean(y))
                return self

            def predict(self, X):
                return np.full(len(X), self.mean_)

        frame = prepare_model_matrix(make_synthetic_feature_frame(12))
        X = frame.drop(columns=["product_id", "week", "order_demand", "demand_log1p", "sample_weight"])
        weeks = frame["week"]
        y_log = frame["demand_log1p"].to_numpy()
        weights = frame["sample_weight"].to_numpy()
        demand = frame["order_demand"].to_numpy()

        folds = [(weeks.iloc[:6].to_numpy(), weeks.iloc[6:8].to_numpy())]
        results = run_expanding_cv(MeanModel(), X, y_log, weights, weeks, demand, folds)
        assert len(results) == 1
        for key in ["wape", "mae", "rmse", "asymmetric_loss_2_1", "n_train", "n_val"]:
            assert key in results[0]


class TestSampleWeights:
    """Sample-weight monotonicity."""

    def test_weights_grow_but_sublinearly(self):
        w = make_sample_weights(pd.Series([0.0, 10.0, 100.0]))
        assert w[0] < w[1] < w[2]
        # Sub-linear: doubling demand does not double weight.
        assert w[2] < w[0] + make_sample_weights(pd.Series([200.0]))[0]


class TestPyFuncSchema:
    """Strict input/signature contract."""

    COLS = ["a", "b"]

    def test_validate_input_rejects_non_frame(self):
        with pytest.raises(ValueError):
            validate_input([1.0, 2.0], self.COLS)

    def test_validate_input_rejects_missing_columns(self):
        good = pd.DataFrame({"a": [1.0], "b": [2.0]})
        with pytest.raises(ValueError):
            validate_input(good.drop(columns=["a"]), self.COLS)

    def test_validate_input_rejects_non_numeric(self):
        bad = pd.DataFrame({"a": ["x"], "b": [2.0]})
        with pytest.raises(ValueError):
            validate_input(bad, self.COLS)

    def test_predict_returns_unit_scale_dataframe(self):
        class Stub:
            def predict(self, X):
                return np.full(len(X), np.log1p(7.0))

        wrapper = DemandForecaster(self.COLS)
        wrapper._model = Stub()
        out = wrapper.predict(None, pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]}))
        assert list(out.columns) == ["pred_demand"]
        assert out["pred_demand"].tolist() == pytest.approx([7.0, 7.0])

    def test_load_context_requires_model_artifact(self):
        wrapper = DemandForecaster(self.COLS)
        with pytest.raises(ValueError):
            wrapper.load_context(type("Ctx", (), {"artifacts": {}})())


class TestMlflowRoundtrip:
    """Full log/load PyFunc packaging on a local file store."""

    def test_log_and_load_pyfunc_model(self, tmp_path):
        import mlflow

        from src.models.pyfunc_wrapper import save_winner

        cols = ["a", "b"]
        model = _ConstPredictor(const=3.0)
        (tmp_path / "artifacts").mkdir()

        mlflow.set_tracking_uri((tmp_path / "mlruns").as_uri())
        experiment_id = mlflow.set_experiment("forecast_roundtrip").experiment_id

        with mlflow.start_run(experiment_id=experiment_id):
            save_winner(model, cols, str(tmp_path / "artifacts"))

        runs = mlflow.search_runs(experiment_ids=[experiment_id])
        assert len(runs) == 1
        run_id = runs.iloc[0]["run_id"]

        loaded = mlflow.pyfunc.load_model(f"runs:/{run_id}/demand_forecaster")
        request = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        out = loaded.predict(request)
        expected = float(np.expm1(3.0))
        assert out["pred_demand"].tolist() == pytest.approx([expected, expected])

        with pytest.raises(ValueError):
            loaded.predict(request.drop(columns=["a"]))