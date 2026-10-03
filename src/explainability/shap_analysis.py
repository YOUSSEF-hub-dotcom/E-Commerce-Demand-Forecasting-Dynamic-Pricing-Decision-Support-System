"""Part-5 SHAP explainability for the Production demand model.

Explains ``models:/demand_forecasting_final/2`` (the existing Production PyFunc
model) using the appropriate tree explainer for its underlying estimator. This
is an explainability-only layer:

* nothing here trains, retunes, or registers any model;
* the Production model (and its stored estimator) is loaded read-only and never
  modified;
* the reference data come exclusively from Train/Validation history (default
  reference week = the last Part-4 validation week, strictly before the Test
  boundary) - the protected Test split is never read, loaded, scored, or used
  for sampling;
* all sampling is deterministic (fixed seed, ``numpy`` legacy ``RandomState``)
  and fully documented in the generated metadata/report.

Relationship between the MLflow PyFunc wrapper and the underlying estimator
-----------------------------------------------------------------------------
The registered model is packaged as ``src.models.pyfunc_wrapper.DemandForecaster``
(an ``mlflow.pyfunc.PythonModel``) whose ``load_context`` loads a joblib-serialized
``lightgbm.sklearn.LGBMRegressor``. The PyFunc wrapper validates each request
against the frozen 93-column schema and serves business-scale predictions by
``expm1(log_output)`` (+ physical floor at 0). The estimator itself was trained
on the ``demand_log1p`` target, so **raw SHAP values are contributions to the
log-domain prediction**; the unit-scale prediction is ``expm1(base + sum(shap))``.
The wrapper and the estimator are preserved exactly as registered - SHAP only
reads them.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.models.final_forecast import IDENTITY_COLS, REGIME_COLS, freeze_schema
from src.utils.logging_setup import get_logger

LOGGER = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPLAINABILITY_DIR = PROJECT_ROOT / "artifacts" / "explainability"
LOCAL_DIR = EXPLAINABILITY_DIR / "local"
REPORT_DIR = PROJECT_ROOT / "reports" / "forecasting"

MODEL_NAME = "demand_forecasting_final"
MODEL_VERSION = 2
MODEL_URI = f"models:/{MODEL_NAME}/{MODEL_VERSION}"

TEST_START_WEEK = "2018-05-21"
LAST_VAL_WEEK = "2018-05-14"

SEED = 20260924
BACKGROUND_SIZE = 400
EXPLAIN_SIZE = 1500
TOP_N = 20

PRODUCT_COL = "product_id"
WEEK_COL = "week"
PRED_UNITS_COL = "predicted_demand_units"

FEATURE_GROUPS = (
    ("lag", lambda c: c.startswith("demand_lag_")),
    ("rolling", lambda c: c.startswith("rolling_demand_") and c not in REGIME_COLS),
    ("price", lambda c: c == "price_prev"),
    ("price_change", lambda c: c == "price_chg_1w"),
    ("freight", lambda c: c == "freight_prev"),
    ("driver", lambda c: c == "category_share_prev"),
    ("calendar", lambda c: c in ("week_of_year", "month", "quarter", "year")),
    ("regime", lambda c: c in REGIME_COLS),
    ("categorical", lambda c: c.startswith("cat_")),
)


def feature_group(col: str) -> str:
    """Assign an actual-schema feature to its documented group."""
    for name, pred in FEATURE_GROUPS:
        if pred(col):
            return name
    return "other"


def model_info_from_registry(name: str = MODEL_NAME, version: int = MODEL_VERSION) -> dict:
    """Return the live registry record for the Production model version."""
    import mlflow

    from src.models.registry import TRACKING_URI

    mlflow.set_tracking_uri(TRACKING_URI)
    client = mlflow.tracking.MlflowClient()
    found = None
    for v in client.search_model_versions(f"name='{name}'"):
        if int(v.version) == int(version):
            found = v
            break
    if found is None:
        raise RuntimeError(f"shap_analysis: model {name} version {version} not in registry.")
    return {
        "model_uri": f"models:/{name}/{version}",
        "name": name,
        "version": int(found.version),
        "stage": found.current_stage,
        "run_id": found.run_id,
        "source": found.source,
        "status": found.status,
    }


def load_pyfunc(uri: str = MODEL_URI):
    """Load the Production PyFunc model through the canonical Part-4 loader."""
    from src.optimization.demand_baseline import load_production_model

    return load_production_model(uri)


def feature_schema(model) -> list[str]:
    """Ordered 93-column schema recorded on the Production model signature."""
    sig = getattr(model, "metadata", None)
    if sig is None or getattr(sig, "signature", None) is None:
        raise ValueError("shap_analysis: model has no metadata signature.")
    inputs = sig.signature.inputs
    if inputs is None:
        raise ValueError("shap_analysis: model signature has no inputs.")
    schema = [t.name for t in inputs]
    if len(schema) != 93:
        raise ValueError(f"shap_analysis: expected 93 feature columns, got {len(schema)}.")
    if set(schema) & set(IDENTITY_COLS):
        raise ValueError("shap_analysis: identity columns present in model input schema.")
    return schema


def underlying_estimator(model):
    """Unwrap the estimator behind the Production PyFunc wrapper (read-only).

    Prefers the public ``mlflow.pyfunc.PyFuncModel.unwrap_python_model()`` API;
    falls back to downloading the registered artifact``s ``model.joblib`` and
    deserializing it with joblib. The estimator is never modified or refit.
    """
    import joblib

    est = None
    source = None
    if hasattr(model, "unwrap_python_model"):
        try:
            pm = model.unwrap_python_model()
            if hasattr(pm, "_model"):
                est = pm._model
                source = "models.pyfunc_wrapper.DemandForecaster._model"
        except Exception as exc:  # private layout drift across mlflow versions
            LOGGER.warning("shap_analysis: unwrap_python_model failed (%s).", exc)
    if est is None:
        import mlflow

        from src.models.registry import TRACKING_URI

        mlflow.set_tracking_uri(TRACKING_URI)
        client = mlflow.tracking.MlflowClient()
        uri = client.get_model_version_download_uri(MODEL_NAME, MODEL_VERSION)
        local = mlflow.artifacts.download_artifacts(artifact_uri=uri)
        est = joblib.load(Path(local) / "artifacts" / "model.joblib")
        source = f"model.joblib artifact of {uri}"
    if not callable(getattr(est, "predict", None)):
        raise RuntimeError("shap_analysis: resolved estimator has no predict method.")
    if not type(est).__module__.startswith("lightgbm"):
        LOGGER.warning("shap_analysis: estimator type is %s (expected lightgbm).", type(est).__name__)
    return est, source


def resolve_estimator(uri: str = MODEL_URI) -> dict:
    """Load the PyFunc model + underlying estimator + canonical schema."""
    model = load_pyfunc(uri)
    schema = feature_schema(model)
    est, est_source = underlying_estimator(model)
    wrapper_cols = list(getattr(getattr(model, "unwrap_python_model", lambda: None)(), "feature_columns", []))
    if wrapper_cols and wrapper_cols != schema:
        raise ValueError("shap_analysis: wrapper feature_columns differ from signature schema.")
    return {
        "model": model,
        "estimator": est,
        "estimator_source": est_source,
        "schema": schema,
        "info": model_info_from_registry(),
        "model_type": f"{type(est).__module__}.{type(est).__name__}",
    }


def reference_frame(reference_week: str = LAST_VAL_WEEK, model=None,
                    prepared: dict[str, pd.DataFrame] | Path | None = None) -> pd.DataFrame:
    """Build the frozen 93-column reference-week design rows (Train/Val only).

    ``prepared`` may be the frozen-lifecycle prepared dict (``{"train","val"}``
    keys used as-is) or the fallback on-the-fly reconstruction over the dense
    weekly matrix. The reference week must be strictly before the Test split.
    The frame carries only ``product_id``, ``week``, and the 93 model features.
    """
    if reference_week >= TEST_START_WEEK:
        raise AssertionError(
            f"shap_analysis: reference week {reference_week} is on/after the Test "
            f"split ({TEST_START_WEEK}); explainability needs pre-Test history."
        )
    from src.optimization.demand_baseline import frozen_features_for_week
    from src.optimization.pricing_data import WEEK_COL

    if prepared is not None:
        import joblib

        if isinstance(prepared, (str, Path)):
            prepared = joblib.load(prepared)
        val = prepared["val"]
        schema = freeze_schema(val)
        rows = val[val[WEEK_COL].astype(str).str.startswith(reference_week)]
        if rows.empty:
            raise ValueError(f"shap_analysis: no rows for reference week {reference_week}.")
        rows = rows[[PRODUCT_COL, WEEK_COL, *schema]].copy()
        rows[WEEK_COL] = rows[WEEK_COL].astype(str)
        return rows
    return frozen_features_for_week(None, reference_week, model=model)


def define_sample(n_rows: int, background_size: int = BACKGROUND_SIZE,
                  explain_size: int = EXPLAIN_SIZE, seed: int = SEED) -> dict[str, np.ndarray]:
    """Deterministic background/explanation index split (no Test contact)."""
    rng = np.random.RandomState(seed)
    size = background_size + explain_size
    if size > n_rows:
        size = n_rows
    chosen = rng.choice(n_rows, size=size, replace=False)
    return {
        "seed": seed,
        "background_indices": chosen[:background_size],
        "explain_indices": chosen[background_size:size],
    }


def baseline_units(estimator, frame: pd.DataFrame, schema: list[str]) -> np.ndarray:
    """Business-scale baseline demand per row: ``expm1(log_pred)`` floored at 0."""
    log_pred = np.asarray(estimator.predict(frame[schema].astype("float64")), dtype=float)
    units = np.expm1(log_pred)
    return np.clip(units, a_min=0.0, a_max=None)


def compute_shap(estimator, schema: list[str], frame: pd.DataFrame,
                 background_indices: np.ndarray, explain_indices: np.ndarray,
                 n_jobs: int = 1) -> dict:
    """Tree SHAP values (log-domain) for the explain sample.

    LightGBM's internal feature binning makes its ``predict`` slightly sensitive
    to the input float representation at bin boundaries. Consequently the SHAP
    path-dependent ``base + sum(phi)`` can drift from ``model.predict`` on some
    rows. To keep the served prediction reproducible **exactly** from the
    attribution output, per-row base values are derived so that
    ``base_row = model_prediction - sum(phi)``; the per-feature attributions
    (``phi``) are unchanged and the additivity residual is measured and reported.
    """
    import shap

    bg = frame.iloc[background_indices][schema].astype("float64")
    ex = frame.iloc[explain_indices][schema].astype("float64")
    explainer = shap.TreeExplainer(estimator, data=bg, feature_names=schema)
    explanation = explainer(X=ex, check_additivity=False)
    values = np.asarray(explanation.values, dtype=float)
    phi_sum = values.sum(axis=1)
    log_pred = np.asarray(estimator.predict(ex), dtype=float)
    if values.shape != (len(explain_indices), len(schema)):
        raise ValueError(
            f"shap_analysis: SHAP matrix ({values.shape}) does not match "
            f"(n_explain={len(explain_indices)}, n_features={len(schema)})."
        )
    if not np.isfinite(values).all():
        bad = int((~np.isfinite(values)).sum())
        raise ValueError(f"shap_analysis: {bad} non-finite SHAP values.")
    global_expected = float(np.asarray(explainer.expected_value, dtype=float).ravel()[0])
    raw_residual = log_pred - (global_expected + phi_sum)
    base_values = log_pred - phi_sum
    return {
        "explainer_type": f"{type(explainer).__module__}.{type(explainer).__name__}",
        "expected_value_log1p": global_expected,
        "base_values_log1p": base_values,
        "values": values,
        "raw_additivity_residual": raw_residual,
        "additivity_residual": {
            "max_abs": float(np.max(np.abs(raw_residual))) if len(raw_residual) else 0.0,
            "mean_abs": float(np.mean(np.abs(raw_residual))) if len(raw_residual) else 0.0,
            "note": (
                "LightGBM bin-quantization drift between SHAP path sums and model.predict "
                "on rows near split boundaries; corrected by per-row base so that "
                "served_prediction == expm1(base_row + sum(phi)) exactly."
            ),
        },
    }


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def save_beeswarm(sv: np.ndarray, schema: list[str], ex_frame: pd.DataFrame,
                  path: Path, max_display: int = TOP_N) -> Path:
    """Top-``max_display`` SHAP beeswarm (importance + direction + distribution)."""
    import shap

    explanation = shap.Explanation(
        values=sv,
        data=ex_frame[schema].to_numpy(dtype=float),
        feature_names=schema,
    )
    plt = _plt()
    shap.plots.beeswarm(explanation, max_display=max_display, show=False)
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.gcf().savefig(path, dpi=200, bbox_inches="tight")
    plt.close("all")
    return path


def save_feature_importance(mean_abs_shap: np.ndarray, schema: list[str],
                            path: Path, max_display: int = TOP_N) -> Path:
    """Bar plot of mean |SHAP| for the top features."""
    import shap

    explanation = shap.Explanation(
        values=np.ones((1, len(schema))) * mean_abs_shap.reshape(1, -1),
        feature_names=schema,
    )
    plt = _plt()
    shap.plots.bar(explanation, max_display=max_display, show=False)
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.gcf().savefig(path, dpi=200, bbox_inches="tight")
    plt.close("all")
    return path


def save_waterfall(sv_row: np.ndarray, data_row: np.ndarray, base: float,
                   schema: list[str], path: Path, max_display: int = 12) -> Path:
    """Waterfall for a single observation (log-domain contributions)."""
    import shap

    explanation = shap.Explanation(
        values=sv_row,
        base_values=base,
        data=data_row,
        feature_names=schema,
    )
    plt = _plt()
    shap.plots.waterfall(explanation, max_display=max_display, show=False)
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.gcf().savefig(path, dpi=200, bbox_inches="tight")
    plt.close("all")
    return path


def select_local_examples(frame: pd.DataFrame, explain_indices: np.ndarray,
                          pred_units: np.ndarray, seed: int = SEED) -> list[dict]:
    """Deterministic representative examples bucketed by predicted demand."""
    rng = np.random.RandomState(seed + 1)
    pos = pred_units > 1e-6
    labels = []
    for i in explain_indices:
        pu = pred_units[i]
        if pu <= 1e-6:
            label = "zero_demand"
        elif pu < 0.01:
            label = "low_demand"
        elif pu < 0.05:
            label = "typical_demand"
        else:
            label = "high_demand"
        labels.append(label)
    frame_idx = {label: [] for label in ("zero_demand", "low_demand", "typical_demand", "high_demand")}
    for idx, label in zip(explain_indices, labels):
        frame_idx[label].append(int(idx))
    examples = []
    for label in ("typical_demand", "high_demand", "low_demand", "zero_demand"):
        cand = frame_idx[label]
        if not cand:
            continue
        order = rng.permutation(len(cand))
        chosen = cand[order[0]]
        row = frame.iloc[chosen]
        examples.append(
            {
                "name": label,
                "product_id": str(row[PRODUCT_COL]),
                "week": str(row[WEEK_COL]),
                "frame_index": int(chosen),
                "predicted_demand_units": float(pred_units[chosen]),
                "bucket": label,
            }
        )
    return examples


def feature_stats(values: np.ndarray, schema: list[str], ex_frame: pd.DataFrame,
                  explain_indices: np.ndarray, pred_units: np.ndarray) -> pd.DataFrame:
    """Per-feature mean |SHAP|, mean SHAP, direction (corr) and group."""
    mean_abs = np.abs(values).mean(axis=0)
    mean_val = values.mean(axis=0)
    data = ex_frame[schema].to_numpy(dtype=float)
    corr = []
    for j in range(values.shape[1]):
        col = data[:, j]
        sv_col = values[:, j]
        constant = np.allclose(col, col[0]) or np.allclose(sv_col, sv_col[0])
        corr.append(np.nan if constant else float(np.corrcoef(col, sv_col)[0, 1]))
    pred_corr = float(np.corrcoef(pred_units[explain_indices], values.sum(axis=1))[0, 1])
    rows = []
    for j, name in enumerate(schema):
        c = corr[j]
        direction = "higher value -> higher output" if c > 0.03 else (
            "higher value -> lower output" if c < -0.03 else "ambiguous")
        rows.append(
            {
                "feature": name,
                "group": feature_group(name),
                "mean_abs_shap": float(mean_abs[j]),
                "mean_shap": float(mean_val[j]),
                "direction_corr": c,
                "direction_label": direction,
            }
        )
    stats = pd.DataFrame(rows).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    return stats, float(pred_corr)


def write_artifacts(results: dict, out_dir: Path, local_dir: Path):
    """Persist the SHAP parquet, importance CSV, images and metadata."""
    out_dir.mkdir(parents=True, exist_ok=True)
    local_dir.mkdir(parents=True, exist_ok=True)

    schema = results["schema"]
    frame = results["frame"]
    explain_indices = results["explain_indices"]
    values = results["shap_values"]
    ex = frame.iloc[explain_indices].reset_index(drop=True)

    table = ex[[PRODUCT_COL, WEEK_COL]].copy()
    table[PRED_UNITS_COL] = results["pred_units"][explain_indices]
    table["base_log1p"] = results["base_values_log1p"]
    table["shap_log1p_sum"] = values.sum(axis=1)
    table["pred_log1p"] = table["base_log1p"] + table["shap_log1p_sum"]
    for j, name in enumerate(schema):
        table[f"shap_{name}"] = values[:, j]
    parquet_path = out_dir / "shap_global_values.parquet"
    table.to_parquet(parquet_path, index=False)

    stats, pred_corr = feature_stats(
        values, schema, ex, np.arange(len(explain_indices)), results["pred_units"][explain_indices]
    )
    csv_path = out_dir / "shap_feature_importance.csv"
    stats.to_csv(csv_path, index=False)

    beeswarm = save_beeswarm(values, schema, ex, out_dir / "shap_summary_beeswarm.png")
    import_bar = save_feature_importance(
        stats["mean_abs_shap"].to_numpy(), schema, out_dir / "shap_feature_importance.png"
    )
    results["top_features"] = stats.head(len(schema))["feature"].tolist()
    results["top_features_table"] = stats.to_dict(orient="records")

    local_files = []
    for example in results["local_examples"]:
        i = example["frame_index"]
        row_pos = int(np.flatnonzero(explain_indices == i)[0])
        name = example["name"]
        png = local_dir / f"example_{name}.png"
        save_waterfall(
            values[row_pos], ex[schema].to_numpy(dtype=float)[row_pos],
            results["base_values_log1p"][row_pos], schema, png,
        )
        local_files.append(str(png))
        example["visualization"] = str(png)

    metadata = {
        "model": {k: results["info"][k] for k in
                  ("model_uri", "name", "version", "stage", "run_id", "source", "status")},
        "underlying_estimator": {
            "type": results["model_type"],
            "estimator_source": results["estimator_source"],
            "relationship": (
                "registered PyFunc DemandForecaster wraps the estimator; the wrapper validates "
                "the 93-col schema and serves expm1-log-output units; SHAP explains the "
                "estimator's log1p-domain output."
            ),
        },
        "dataset": {
            "data_source": "frozen 93-col design rows for the reference week, Train/Val history only",
            "reference_week": results["reference_week"],
            "window_rows": int(len(frame)),
            "explanations_rows": int(len(explain_indices)),
            "background_rows": int(len(results["background_indices"])),
            "test_split_accessed": False,
        },
        "sampling": {
            "strategy": "numpy legacy RandomState, choice without replacement; "
                        "explain indices exclude background indices",
            "seed": int(results["seed"]),
            "background_size": int(results["background_size"]),
            "explain_size": int(len(explain_indices)),
        },
"shap": {
                "explainer": results["explainer_type"],
                "feature_count": len(schema),
                "feature_schema": schema,
                "scale": "log1p (estimator output); unit prediction = expm1(base + sum)",
                "expected_value_log1p": float(results["expected_value_log1p"]),
                "base_values": {
                    "mode": "per-row additivity-corrected",
                    "definition": (
                        "base_row = model_log_prediction - sum(phi); "
                        "served units = expm1(base_row + sum(phi)) exactly."
                    ),
                    "residual_before_correction": results["additivity_residual"],
                },
                "prediction_shap_corr": results["prediction_shap_corr"],
                "n_jobs_internal": 1,
            },
        "artifacts": {
            "parquet": str(parquet_path),
            "importance_csv": str(csv_path),
            "beeswarm": str(beeswarm),
            "importance_plot": str(import_bar),
            "local": [example["visualization"] for example in results["local_examples"]],
        },
        "generated_utc": datetime.now(timezone.utc).isoformat(),
    }
    metadata_path = out_dir / "metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, default=str), encoding="utf-8"
    )
    metadata["artifacts"]["metadata"] = str(metadata_path)
    results["metadata"] = metadata
    results["files"] = metadata["artifacts"]
    return results


def analyze(estimator, frame: pd.DataFrame, schema: list[str], model_info: dict,
            model_type: str, estimator_source: str,
            reference_week: str = LAST_VAL_WEEK,
            background_size: int = BACKGROUND_SIZE, explain_size: int = EXPLAIN_SIZE,
            seed: int = SEED, out_dir: Path = EXPLAINABILITY_DIR,
            local_dir: Path = None, report_dir: Path = REPORT_DIR) -> dict:
    """Core analysis: sample -> SHAP -> local examples -> artifacts -> reports."""
    if reference_week >= TEST_START_WEEK:
        raise AssertionError(
            f"shap_analysis: reference week {reference_week} on/after Test split "
            f"{TEST_START_WEEK}."
        )
    if any(c in schema for c in IDENTITY_COLS):
        raise ValueError("shap_analysis: identity columns leaked into the feature schema.")
    if len(schema) != 93:
        raise ValueError(f"shap_analysis: expected 93 features, got {len(schema)}.")

    sample = define_sample(len(frame), background_size, explain_size, seed)
    bg_idx, ex_idx = sample["background_indices"], sample["explain_indices"]

    pred_units = baseline_units(estimator, frame, schema)

    shap_out = compute_shap(estimator, schema, frame, bg_idx, ex_idx)

    ex_lookup = frame.iloc[ex_idx].reset_index(drop=True)
    stats, pred_corr = feature_stats(
        shap_out["values"], schema, ex_lookup, np.arange(len(ex_idx)), pred_units[ex_idx]
    )

    local_examples = select_local_examples(frame, ex_idx, pred_units, seed)

    results = {
        "estimator": estimator,
        "frame": frame,
        "schema": schema,
        "info": model_info,
        "model_type": model_type,
        "estimator_source": estimator_source,
        "reference_week": reference_week,
        "seed": seed,
        "background_size": background_size,
        "explain_size": len(ex_idx),
        "background_indices": bg_idx,
        "explain_indices": ex_idx,
        "pred_units": pred_units,
        "expected_value_log1p": shap_out["expected_value_log1p"],
        "base_values_log1p": shap_out["base_values_log1p"],
        "additivity_residual": shap_out["additivity_residual"],
        "shap_values": shap_out["values"],
        "explainer_type": shap_out["explainer_type"],
        "feature_stats": stats,
        "prediction_shap_corr": pred_corr,
        "local_examples": local_examples,
    }
    if local_dir is None:
        local_dir = out_dir / "local"
    results = write_artifacts(results, out_dir, local_dir)
    results["reports"] = write_report(results, report_dir)
    return results


def run_analysis(uri: str = MODEL_URI, reference_week: str = LAST_VAL_WEEK,
                 background_size: int = BACKGROUND_SIZE, explain_size: int = EXPLAIN_SIZE,
                 seed: int = SEED, out_dir: Path = EXPLAINABILITY_DIR,
                 local_dir: Path = None, report_dir: Path = REPORT_DIR,
                 prepared=None) -> dict:
    """End-to-end explainability run (real model + permitted reference data)."""
    resolved = resolve_estimator(uri)
    frame = reference_frame(reference_week, model=resolved["model"], prepared=prepared)
    return analyze(
        resolved["estimator"], frame, resolved["schema"], resolved["info"],
        resolved["model_type"], resolved["estimator_source"],
        reference_week=reference_week, background_size=background_size,
        explain_size=explain_size, seed=seed, out_dir=out_dir,
        local_dir=local_dir, report_dir=report_dir,
    )


# --------------------------------------------------------------------------
# Report generation
# --------------------------------------------------------------------------

def _direction_text(row: dict) -> str:
    label = row["direction_label"]
    if label == "higher value -> higher output":
        return "higher feature values are associated with higher predicted demand (log-domain)."
    if label == "higher value -> lower output":
        return "higher feature values are associated with lower predicted demand (log-domain)."
    return "no consistent monotone association in the explained sample."


PRICE_FEATURES = ["price_prev", "price_chg_1w", "freight_prev"]


def write_report(results: dict, report_dir: Path):
    """Write the Part-5 report (markdown + JSON)."""
    import json

    report_dir.mkdir(parents=True, exist_ok=True)
    md_path = report_dir / "part5_shap_explainability.md"
    json_path = report_dir / "part5_shap_explainability.json"

    stats = results["feature_stats"]
    schema = results["schema"]
    top = stats.head(12).to_dict(orient="records")
    price_rows = stats[stats["feature"].isin(PRICE_FEATURES)].to_dict(orient="records")
    groups = stats.groupby("group")["mean_abs_shap"].sum().sort_values(ascending=False)
    local = results["local_examples"]

    md_lines = []
    md_lines.append("# PART 5 — SHAP EXPLAINABILITY REPORT")
    md_lines.append("")
    md_lines.append("Date (UTC): " + datetime.now(timezone.utc).isoformat())
    md_lines.append("")
    md_lines.append("## Model")
    md_lines.append("")
    info = results["info"]
    md_lines.append(f"- **MLflow model URI:** `{info['model_uri']}`")
    md_lines.append(f"- **Version:** {info['version']} (stage: {info['stage']}, status: {info['status']})")
    md_lines.append(f"- **Run ID:** `{info['run_id']}`")
    md_lines.append(f"- **Model type:** {results['model_type']}")
    md_lines.append(f"- **Feature count:** {len(schema)}")
    md_lines.append(f"- **PyFunc relationship:** `DemandForecaster` wrapper validates the 93-col "
                    f"schema and serves `expm1(log-output)` units; SHAP explains the wrapped "
                    f"LightGBM estimator's log1p output.")
    md_lines.append("")
    md_lines.append("## Dataset")
    md_lines.append("")
    md_lines.append(f"- **Data source:** frozen 93-column design rows, Train/Validation history only")
    md_lines.append(f"- **Reference period:** {results['reference_week']} (strictly before `TEST_START_WEEK = {TEST_START_WEEK}`)")
    md_lines.append(f"- **Window rows:** {len(results['frame'])}")
    md_lines.append(f"- **Rows explained:** {results['explain_size']}")
    md_lines.append(f"- **Background sample:** {results['background_size']}")
    md_lines.append(f"- **Test split accessed:** FALSE")
    md_lines.append("")
    md_lines.append("## SHAP configuration")
    md_lines.append("")
    md_lines.append(f"- **Explainer:** {results['explainer_type']} (TreeExplainer on the Production estimator)")
    md_lines.append(f"- **Expected value (log1p):** {results['expected_value_log1p']:.6f}")
    md_lines.append(f"- **Background sample:** deterministic {results['background_size']} rows from the reference week")
    md_lines.append(f"- **Explanation sample:** deterministic {results['explain_size']} rows from the reference week (excluding background)")
    md_lines.append(f"- **Random seed:** {results['seed']} (`numpy` legacy `RandomState`)")
    md_lines.append("- **SHAP scale:** log1p — unit prediction = `expm1(base + sum(shap))`")
    md_lines.append("")
    md_lines.append("### Faithfulness (additivity)")
    md_lines.append("")
    res = results["additivity_residual"]
    md_lines.append(
        f"- LightGBM internal feature binning makes `model.predict` float-sensitive at split "
        f"boundaries, so raw SHAP `base + sum(phi)` can drift from the model output "
        f"(max abs {res['max_abs']:.6f}, mean abs {res['mean_abs']:.6f}, log1p scale)."
    )
    md_lines.append(
        "- Per-row base values are therefore additivity-corrected: "
        "`base_row = model_log_prediction - sum(phi)`, so the served unit prediction is "
        "reproduced **exactly** as `expm1(base_row + sum(phi))`. Per-feature SHAP attributions "
        "are unchanged."
    )
    md_lines.append("")
    md_lines.append("## Global findings (mean |SHAP|, top 12)")
    md_lines.append("")
    md_lines.append("| # | Feature | Group | Mean |SHAP| | Typical direction |")
    md_lines.append("|---|---------|-------|------:|--------------------|")
    for i, row in enumerate(top):
        md_lines.append(
            f"| {i + 1} | `{row['feature']}` | {row['group']} | {row['mean_abs_shap']:.5f} | "
            f"{_direction_text(row)} |"
        )
    md_lines.append("")
    md_lines.append("### Group-level mean |SHAP|")
    md_lines.append("")
    md_lines.append("| Group | Total mean |SHAP| | Feature count |")
    md_lines.append("|-------|------------:|:------------:|")
    counts = stats.groupby("group")["feature"].count()
    for group, total in groups.items():
        md_lines.append(f"| {group} | {total:.4f} | {int(counts[group])} |")
    md_lines.append("")
    md_lines.append("## Price-related features")
    md_lines.append("")
    if price_rows:
        md_lines.append("| Feature | Group | Mean |SHAP| | Typical direction |")
        md_lines.append("|---------|-------|------:|--------------------|")
        for row in price_rows:
            md_lines.append(
                f"| `{row['feature']}` | {row['group']} | {row['mean_abs_shap']:.5f} | "
                f"{_direction_text(row)} |"
            )
        md_lines.append("")
        md_lines.append(
            "Interpretation: the model associates these price/freight features with predicted "
            "demand, but SHAP is a model-attribution tool, not a demand-response estimator. "
            "Causal price-response belongs to the Part-4 elasticity analysis."
        )
    else:
        md_lines.append("None of `price_prev` / `price_chg_1w` / `freight_prev` are present in the schema.")
    md_lines.append("")
    md_lines.append("## Regime / temporal features")
    md_lines.append("")
    regime_rows = stats[stats["group"].isin(("regime", "calendar", "rolling", "lag"))].head(12)
    md_lines.append("| Feature | Group | Mean |SHAP| | Typical direction |")
    md_lines.append("|---------|-------|------:|--------------------|")
    for _, row in regime_rows.iterrows():
        md_lines.append(
            f"| `{row['feature']}` | {row['group']} | {row['mean_abs_shap']:.5f} | "
            f"{_direction_text(row)} |"
        )
    md_lines.append("")
    md_lines.append("## Local explanations")
    md_lines.append("")
    md_lines.append("| Example | Product | Week | Predicted demand (units) | Waterfall |")
    md_lines.append("|---------|---------|------|--------------------------:|-----------|")
    for ex in local:
        md_lines.append(
            f"| {ex['name']} | `{ex['product_id']}` | `{ex['week']}` | "
            f"{ex['predicted_demand_units']:.4f} | `{ex['visualization']}` |"
        )
    md_lines.append("")
    md_lines.append("## Limitations")
    md_lines.append("")
    md_lines.append("- SHAP explains the model's behavior; it does not establish causality.")
    md_lines.append("- Elasticity remains the appropriate framework for price-response analysis.")
    md_lines.append("- SHAP importance does not imply business causality.")
    md_lines.append("- Results depend on the reference data distribution (single validation week).")
    md_lines.append("- Correlated features can distribute SHAP attribution across multiple variables.")
    md_lines.append("")

    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    report_json = {
        "phase": "part5_shap_explainability",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "model": {k: results["info"][k] for k in
                  ("model_uri", "name", "version", "stage", "run_id", "source", "status")},
        "model_type": results["model_type"],
        "feature_count": len(schema),
        "feature_schema": schema,
        "dataset": {
            "data_source": "frozen 93-column design rows from Train/Validation history",
            "reference_week": results["reference_week"],
            "window_rows": int(len(results["frame"])),
            "rows_explained": int(results["explain_size"]),
            "background_rows": int(results["background_size"]),
            "test_split_accessed": False,
        },
        "shap": {
            "explainer": results["explainer_type"],
            "explainer_relationship": (
                "TreeExplainer applied to the wrapped lightgbm.sklearn.LGBMRegressor; "
                "SHAP values are log1p-domain contributions."
            ),
            "expected_value_log1p": float(results["expected_value_log1p"]),
            "additivity_residual_before_correction": results["additivity_residual"],
            "base_values": {
                "mode": "per-row additivity-corrected",
                "definition": "base_row = model_log_prediction - sum(phi)",
            },
            "prediction_shap_corr": float(results["prediction_shap_corr"]),
            "seed": int(results["seed"]),
            "scale": "log1p",
        },
        "global_findings_top": top,
        "group_importance": {
            str(k): {"total_mean_abs_shap": float(v), "feature_count": int(counts[k])}
            for k, v in groups.items()
        },
        "price_related": price_rows,
        "regime_temporal": regime_rows.to_dict(orient="records"),
        "local_examples": local,
        "limitations": [
            "SHAP explains model behavior, not causality.",
            "Elasticity is the appropriate framework for price-response analysis.",
            "SHAP importance does not imply business causality.",
            "Results depend on the reference data distribution.",
            "Correlated features can distribute SHAP attribution across multiple variables.",
        ],
        "artifacts": results["files"],
    }
    json_path.write_text(json.dumps(report_json, indent=2, default=str), encoding="utf-8")
    return {"md": md_path, "json": json_path}


def main() -> None:
    """CLI entry point for the Part-5 explainability run."""
    import argparse

    parser = argparse.ArgumentParser(description="Part-5 SHAP explainability run")
    parser.add_argument("--reference-week", default=LAST_VAL_WEEK)
    parser.add_argument("--background-size", type=int, default=BACKGROUND_SIZE)
    parser.add_argument("--explain-size", type=int, default=EXPLAIN_SIZE)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    results = run_analysis(
        reference_week=args.reference_week,
        background_size=args.background_size,
        explain_size=args.explain_size,
        seed=args.seed,
    )
    print(
        "shap_analysis: explained",
        len(results["explain_indices"]),
        "rows; artifacts under",
        results["files"],
    )


if __name__ == "__main__":
    main()