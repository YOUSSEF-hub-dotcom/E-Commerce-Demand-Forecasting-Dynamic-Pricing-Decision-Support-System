"""Part-5 SHAP explainability tests.

Covers: Production model contract (version/stage/run/schema), data protection
(reference week < Test split; module never reads ``data/splits/test.csv``),
schema fidelity (no identity columns, feature ordering preserved), SHAP output
(dimensions, finiteness, additivity, determinism, reproducibility), artifact
existence, and interpretation consistency.
"""

import ast
import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import src.explainability.shap_analysis as sa
from src.models.final_forecast import IDENTITY_COLS

EXPECTED_RUN_ID = "5d729a8594dc488bbd18ba0c4e216f5a"
TOL = 1e-6


def _synthetic_frame(schema, n_rows=400, seed=7) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    frame = pd.DataFrame(rng.rand(n_rows, len(schema)), columns=schema)
    frame["product_id"] = [f"p{i % 40}" for i in range(n_rows)]
    frame["week"] = sa.LAST_VAL_WEEK
    return frame[["product_id", "week"] + list(schema)]


def _tiny_estimator(schema):
    import lightgbm

    rng = np.random.RandomState(1234)
    n = 200
    X = pd.DataFrame(rng.rand(n, len(schema)), columns=schema).astype("float64")
    y = 2.0 * X[schema[0]].to_numpy() - 1.5 * X[schema[1]].to_numpy() + 0.5 * rng.rand(n)
    model = lightgbm.LGBMRegressor(
        n_estimators=10,
        num_leaves=4,
        max_depth=3,
        learning_rate=0.1,
        min_child_samples=3,
        random_state=1,
        n_jobs=1,
        verbose=-1,
    )
    model.fit(X, y)
    return model


def _model_info() -> dict:
    return {
        "model_uri": "models:/demand_forecasting_final/2",
        "name": "demand_forecasting_final",
        "version": 2,
        "stage": "Production",
        "run_id": EXPECTED_RUN_ID,
        "source": "models:/m-e93cadaf963a4aa1b44b3e501379da92",
        "status": "READY",
    }


def _analyze(schema, tmp_path, background_size=40, explain_size=80, seed=sa.SEED):
    frame = _synthetic_frame(schema, n_rows=300)
    return sa.analyze(
        _tiny_estimator(schema), frame, schema, _model_info(),
        "lightgbm.sklearn.LGBMRegressor", "test",
        background_size=background_size, explain_size=explain_size, seed=seed,
        out_dir=Path(tmp_path) / "out", report_dir=Path(tmp_path) / "reports",
    )


@pytest.fixture(scope="module")
def real_schema():
    """Canonical 93-column Production schema (one live load for the suite)."""
    model = sa.load_pyfunc(sa.MODEL_URI)
    return sa.feature_schema(model)


class TestDataProtection:
    def test_module_never_reads_test_split_in_source(self):
        src = inspect.getsource(sa)
        tree = ast.parse(src)
        if (
            tree.body
            and isinstance(tree.body[0], ast.Expr)
            and isinstance(tree.body[0].value, ast.Constant)
            and isinstance(tree.body[0].value.value, str)
        ):
            lines = src.splitlines()
            body_lines = "\n".join(lines[tree.body[0].end_lineno:])
        else:
            body_lines = src
        assert "test.csv" not in body_lines
        assert "splits/test" not in body_lines

    def test_reference_week_must_be_before_test_start(self, tmp_path):
        schema = [f"f{i:02d}" for i in range(93)]
        frame = _synthetic_frame(schema)
        with pytest.raises(AssertionError, match="Test split"):
            sa.analyze(
                _tiny_estimator(schema), frame, schema, _model_info(),
                "lightgbm.sklearn.LGBMRegressor", "test",
                reference_week=sa.TEST_START_WEEK,
                out_dir=Path(tmp_path) / "out", report_dir=Path(tmp_path) / "reports",
            )


class TestProductionModelContract:
    def test_registry_production_contract(self):
        info = sa.model_info_from_registry()
        assert info["name"] == sa.MODEL_NAME
        assert info["version"] == sa.MODEL_VERSION == 2
        assert info["stage"] == "Production"
        assert info["run_id"] == EXPECTED_RUN_ID
        assert info["status"] == "READY"

    def test_frozen_schema_93_with_no_identity_columns(self, real_schema):
        assert len(real_schema) == 93
        assert not set(real_schema) & set(IDENTITY_COLS)
        assert real_schema[0] == "week_of_year"
        assert any(c.startswith("cat_") for c in real_schema)
        assert all(c in real_schema for c in sa.REGIME_COLS)


class TestSchemaFidelity:
    def test_analysis_keeps_schema_order(self, tmp_path, real_schema):
        results = _analyze(real_schema, tmp_path)
        assert results["schema"] == real_schema
        table = pd.read_parquet(Path(tmp_path) / "out" / "shap_global_values.parquet")
        assert list(table.columns[:4]) == ["product_id", "week", sa.PRED_UNITS_COL, "base_log1p"]

    def test_no_identity_columns_in_shap_matrix(self, tmp_path, real_schema):
        results = _analyze(real_schema, tmp_path)
        assert results["shap_values"].shape[1] == 93
        assert not set(IDENTITY_COLS) & set(real_schema)


class TestShapOutput:
    def test_dimensions_finite_and_additive(self, tmp_path, real_schema):
        frame = _synthetic_frame(real_schema, n_rows=300)
        est = _tiny_estimator(real_schema)
        results = sa.analyze(
            est, frame, real_schema, _model_info(), "lightgbm.sklearn.LGBMRegressor",
            "test", background_size=40, explain_size=80,
            out_dir=Path(tmp_path) / "out", report_dir=Path(tmp_path) / "reports",
        )
        sv = results["shap_values"]
        assert sv.shape == (80, 93)
        assert np.isfinite(sv).all()
        log_pred = results["base_values_log1p"] + sv.sum(axis=1)
        est_pred = np.asarray(
            est.predict(frame.iloc[results["explain_indices"]][real_schema]),
            dtype=float,
        )
        assert np.allclose(log_pred, est_pred, atol=1e-9)
        assert results["base_values_log1p"].shape == (80,)

    def test_deterministic_sampling(self):
        a = sa.define_sample(10_000, background_size=40, explain_size=80, seed=sa.SEED)
        b = sa.define_sample(10_000, background_size=40, explain_size=80, seed=sa.SEED)
        assert np.array_equal(a["background_indices"], b["background_indices"])
        assert np.array_equal(a["explain_indices"], b["explain_indices"])
        assert not set(a["background_indices"].tolist()) & set(a["explain_indices"].tolist())

    def test_local_examples_deterministic(self, tmp_path, real_schema):
        results = _analyze(real_schema, tmp_path)
        first = [ex["frame_index"] for ex in results["local_examples"]]
        again = sa.select_local_examples(
            results["frame"], results["explain_indices"], results["pred_units"], seed=sa.SEED
        )
        assert [ex["frame_index"] for ex in again] == first


class TestReproducibility:
    def test_repeated_runs_match(self, tmp_path, real_schema):
        est = _tiny_estimator(real_schema)
        frame = _synthetic_frame(real_schema, n_rows=300)

        def run(kind):
            return sa.analyze(
                est, frame, real_schema, _model_info(), "lightgbm.sklearn.LGBMRegressor",
                "test", background_size=40, explain_size=80,
                out_dir=Path(tmp_path) / kind, report_dir=Path(tmp_path) / "reports",
            )

        r1 = run("a")
        r2 = run("b")
        assert np.allclose(r1["shap_values"], r2["shap_values"], atol=TOL)
        assert np.allclose(
            r1["feature_stats"]["mean_abs_shap"].to_numpy(),
            r2["feature_stats"]["mean_abs_shap"].to_numpy(),
            atol=TOL,
        )


class TestArtifacts:
    def test_artifacts_exist_and_are_readable(self, tmp_path, real_schema):
        results = _analyze(real_schema, tmp_path)
        for key in ("beeswarm", "importance_plot", "parquet", "importance_csv", "metadata"):
            assert Path(results["files"][key]).exists(), key

        parquet = pd.read_parquet(results["files"]["parquet"])
        assert parquet.shape[0] == 80
        assert parquet[sa.PRED_UNITS_COL].notna().all()

        importance = pd.read_csv(results["files"]["importance_csv"])
        assert importance["feature"].tolist() == results["feature_stats"]["feature"].tolist()
        assert (importance["mean_abs_shap"] >= 0).all()

        metadata = json.load(open(results["files"]["metadata"], encoding="utf-8"))
        assert metadata["dataset"]["test_split_accessed"] is False
        assert metadata["sampling"]["seed"] == sa.SEED

        for path in results["files"]["local"]:
            assert Path(path).exists(), path

        assert Path(results["reports"]["md"]).exists()
        assert Path(results["reports"]["json"]).exists()

        with open(results["reports"]["json"], encoding="utf-8") as fh:
            report = json.load(fh)
        assert report["dataset"]["test_split_accessed"] is False
        assert report["feature_count"] == 93

    def test_importance_csv_matches_computed_values(self, tmp_path, real_schema):
        results = _analyze(real_schema, tmp_path)
        computed = np.abs(results["shap_values"]).mean(axis=0)
        importance = pd.read_csv(results["files"]["importance_csv"])
        order = [results["schema"].index(f) for f in importance["feature"]]
        assert np.allclose(computed[order], importance["mean_abs_shap"].to_numpy(), atol=TOL)


class TestInterpretation:
    def test_price_features_ranked_from_actual_shap(self, tmp_path, real_schema):
        results = _analyze(real_schema, tmp_path)
        stats_df = results["feature_stats"]
        for col in sa.PRICE_FEATURES:
            assert col in stats_df["feature"].tolist()
        price_rank = stats_df["feature"].tolist().index("price_prev")
        computed = np.abs(results["shap_values"]).mean(axis=0)
        assert np.isclose(
            computed[results["schema"].index("price_prev")],
            stats_df.loc[stats_df["feature"] == "price_prev", "mean_abs_shap"].iloc[0],
            atol=TOL,
        )
        assert price_rank >= 0 and price_rank < len(stats_df)

    def test_local_example_fields_complete(self, tmp_path, real_schema):
        results = _analyze(real_schema, tmp_path)
        assert results["local_examples"], "expected at least one local example"
        for ex in results["local_examples"]:
            assert ex["product_id"]
            assert ex["week"] == sa.LAST_VAL_WEEK
            assert ex["frame_index"] in results["explain_indices"].tolist()
            assert ex["predicted_demand_units"] >= 0
            assert Path(ex["visualization"]).exists()