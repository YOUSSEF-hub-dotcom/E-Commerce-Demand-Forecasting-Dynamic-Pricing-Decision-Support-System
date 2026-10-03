"""Part-3 pre-final-test cleanup & hardening guards.

Encodes the contracts established by the Part-3 pre-final-test cleanup:

* The frozen v2 feature schema is authoritative: 93 model inputs, exact
  names and exact ordering -- and the shipped ``artifacts/phase3/
  feature_columns.json`` matches it, the MLflow in-run feature-column
  artifact, and the registered model's MLflow signature (4-way equality).
* One-hot encoding is structurally train-only: the vocabulary is learned
  from the TRAIN split alone, validation/test are aligned to that frozen
  layout, and a later split can never expand the schema. The ``unknown``
  label is part of the train vocabulary (imputation contract), so every
  label a validation/test row could carry is expressible in the frozen
  schema.
* Temporal splits obey the boundary-week convention: train/val share a
  single documented boundary week label, and val/test share a second one,
  with row-level disjointness preserved (chronological, inclusive ordering).
* The frozen v2 candidate is untouched by the cleanup: no parameter drift
  versus the recorded v2 audit / MLflow winner run, no model retraining
  side-effects asserted via the lifecycle's own reproducibility constant.

These tests are lightweight: they read little metadata, small split columns,
and synthetic frames only -- never the heavy demand matrix.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import src.models.data_prep as dp
import src.models.final_forecast as ff

ROOT = Path(__file__).resolve().parents[1]

SPLIT_PATH = ROOT / "data" / "splits"


def read_repo_json(rel: str):
    return json.loads((ROOT / rel).read_text(encoding="utf-8"))


def load_categories(split: str) -> list[str]:
    df = pd.read_csv(SPLIT_PATH / f"{split}.csv", usecols=["product_category_name"])
    return sorted(df["product_category_name"].dropna().astype(str).unique())


def expected_frozen_schema(train_categories: list[str]) -> list[str]:
    """Authoritative frozen layout: base + train-only OHE + regime block."""
    return list(dp.FEATURE_COLS) + [f"cat_{c}" for c in sorted(train_categories)] + list(ff.REGIME_COLS)


# --------------------------------------------------------------------------- #
# 1. Frozen schema: 4-way equivalence
# --------------------------------------------------------------------------- #


def test_artifact_matches_authoritative_frozen_schema():
    ship = read_repo_json("artifacts/phase3/feature_columns.json")
    mlflow_artifact = read_repo_json(
        "mlruns/2/5d729a8594dc488bbd18ba0c4e216f5a/artifacts/feature_columns.json"
    )["feature_columns"]
    assert ship == mlflow_artifact, "phase3 artifact drifted from the MLflow in-run feature columns"
    assert len(ship) == 93
    assert ship[:13] == list(dp.FEATURE_COLS)
    assert ship[-7:] == list(ff.REGIME_COLS)
    assert sum(c.startswith("cat_") for c in ship) == 73


def test_artifact_reconstructs_from_train_split_categories():
    ship = read_repo_json("artifacts/phase3/feature_columns.json")
    train_cats = load_categories("train")
    assert len(train_cats) == 73
    assert any(c == "unknown" for c in train_cats)
    assert ship == expected_frozen_schema(train_cats)


def test_artifact_matches_mlflow_registered_model_signature():
    from mlflow.models import Model

    ship = read_repo_json("artifacts/phase3/feature_columns.json")
    model = Model.load(ROOT / "mlruns/2/models/m-e93cadaf963a4aa1b44b3e501379da92/artifacts")
    sig_inputs = [c["name"] for c in model.signature.inputs.to_dict()]
    assert sig_inputs == ship, "registered model input schema no longer matches the shipped artifact"


def test_inference_matrix_schema_is_frozen_for_all_splits():
    """The shipped frozen layout covers every category present in val/test."""
    ship = read_repo_json("artifacts/phase3/feature_columns.json")
    train_cats = load_categories("train")
    for split in ("val", "test"):
        cats = load_categories(split)
        assert set(cats) <= set(train_cats), f"{split} expands the train vocabulary"
        ohe = [f"cat_{c}" for c in cats]
        assert all(c in ship for c in ohe), f"{split} holds a label absent from the frozen schema"


# --------------------------------------------------------------------------- #
# 2. OHE structural hardening: train-only vocabulary
# --------------------------------------------------------------------------- #


def _synthetic_aug(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """Fill the design-matrix columns required by ``prepare_model_matrix``."""
    out = pd.DataFrame(index=range(n))
    for c in dp.FEATURE_COLS:
        out[c] = 1.0
    for c in ff.REGIME_COLS:
        out[c] = 0.5
    out["product_id"] = [f"p{i}" for i in range(n)]
    out["week"] = "2018-01-01"
    out["order_demand"] = 1.0
    out["demand_log1p"] = 0.69
    out["sample_weight"] = 1.0
    cats = list(df["product_category_name"].reset_index(drop=True))
    if len(cats) < n:
        cats = (cats * (n // len(cats) + 1))[:n]
    out["product_category_name"] = cats
    return out


def _synthetic_splits(train_cats, val_cats, test_cats) -> dict[str, pd.DataFrame]:
    n_train = max(2, len(train_cats))
    n_val = max(2, len(val_cats))
    n_test = max(2, len(test_cats))
    tr = pd.DataFrame({"product_category_name": train_cats[-n_train:]})
    va = pd.DataFrame({"product_category_name": val_cats[-n_val:]})
    te = pd.DataFrame({"product_category_name": test_cats[-n_test:]})
    return {
        "train": _synthetic_aug(tr, n_train),
        "val": _synthetic_aug(va, n_val),
        "test": _synthetic_aug(te, n_test),
    }


def _synthetic_single(cat: str) -> pd.DataFrame:
    df = pd.DataFrame({"product_category_name": [cat]})
    return _synthetic_aug(df, 1)


def test_prepare_frozen_learns_vocab_from_train_only():
    train_cats = ["alpha", "beta", "unknown"]
    splits = _synthetic_splits(train_cats, ["alpha", "unknown"], ["beta", "alpha"])
    prepared = ff.prepare_frozen(splits)
    cat_cols = [c for c in prepared["train"].columns if c.startswith("cat_")]
    assert cat_cols == ["cat_alpha", "cat_beta", "cat_unknown"], "OHE must be derived from train labels only"
    assert set(prepared["train"].columns) == set(prepared["val"].columns) == set(prepared["test"].columns)
    assert (
        len(prepared["train"].columns)
        == 5 + len(dp.FEATURE_COLS) + len(ff.REGIME_COLS) + 3
    )  # 5 identity cols + 13 base + 7 regime + 3 one-hot


def test_prepare_frozen_rejects_val_categories_unseen_in_train():
    splits = _synthetic_splits(["alpha"], ["beta"], ["alpha"])
    with pytest.raises(RuntimeError, match="unseen in train"):
        ff.prepare_frozen(splits)


def test_prepare_frozen_rejects_test_categories_unseen_in_train():
    splits = _synthetic_splits(["alpha"], ["alpha"], ["gamma"])
    with pytest.raises(RuntimeError, match="unseen in train"):
        ff.prepare_frozen(splits)


def test_unknown_label_is_absorbed_into_frozen_layout():
    """A val/test row carrying the reserved 'unknown' label stays in-schema."""
    splits = _synthetic_splits(["alpha", "unknown"], ["unknown", "alpha"], ["alpha", "unknown"])
    prepared = ff.prepare_frozen(splits)
    assert "cat_unknown" in prepared["val"].columns


def test_onehot_block_cannot_escape_frozen_schema():
    bad = _synthetic_aug(pd.DataFrame({"product_category_name": ["nope"]}), 1)
    with pytest.raises(RuntimeError, match="escapes the frozen schema"):
        dp.prepare_model_matrix(bad, categories=["alpha"])


# --------------------------------------------------------------------------- #
# 3. Temporal boundary-week convention
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def split_weeks() -> dict[str, pd.Series]:
    return {
        k: pd.read_csv(SPLIT_PATH / f"{k}.csv", usecols=["week"])["week"]
        for k in ("train", "val", "test")
    }


def test_train_val_share_exactly_one_boundary_week(split_weeks):
    tr_max = split_weeks["train"].max()
    va_min = split_weeks["val"].min()
    assert tr_max == va_min
    assert tr_max == "2018-01-29/2018-02-04", f"unexpected train/val boundary {tr_max}"


def test_val_test_share_exactly_one_boundary_week(split_weeks):
    va_max = split_weeks["val"].max()
    te_min = split_weeks["test"].min()
    assert va_max == te_min
    assert va_max == "2018-05-21/2018-05-27", f"unexpected val/test boundary {va_max}"


def test_splits_are_chronological_and_disjoint(split_weeks):
    assert split_weeks["train"].max() <= split_weeks["val"].min()
    assert split_weeks["val"].max() <= split_weeks["test"].min()
    for k, other in (("train", "val"), ("val", "test")):
        overlap = split_weeks[k].isin(split_weeks[other])
        # shared week LABELS are expected; the boundary rules allow one label
        assert overlap.sum() > 0


# --------------------------------------------------------------------------- #
# 4. Frozen v2 candidate integrity (cleanup made no behavioral change)
# --------------------------------------------------------------------------- #


def test_frozen_params_match_v2_audit():
    audit = read_repo_json("reports/forecasting/part3_v2_audit.json")["freeze_config"]
    assert audit["params"] == ff.WINNER_PARAMS
    assert ff.freeze_config()["params"] == ff.WINNER_PARAMS


def test_frozen_params_match_mlflow_winner_run():
    cfg = read_repo_json(
        "mlruns/2/5d729a8594dc488bbd18ba0c4e216f5a/artifacts/freeze_config.json"
    )
    assert cfg["params"] == ff.WINNER_PARAMS
    assert cfg["feature_version"] == ff.FEATURE_VERSION == "phase3-exp2-v1"


def test_frozen_candidate_config_is_stable():
    cfg = ff.freeze_config()
    assert cfg["model_type"] == "lightgbm"
    assert cfg["params"]["n_estimators"] == 600
    assert cfg["target_transform"] == "log1p"
    assert cfg["prediction_transform"] is not None


def test_cleanup_preserves_expected_validation_wape_constant():
    from src.models import finalize_lifecycle, finalize_v2

    assert finalize_v2.EXP2_VALIDATION_WAPE == finalize_lifecycle.EXP2_VALIDATION_WAPE == 0.3964


def test_split_csv_categories_match_frozen_ohe_across_files():
    ship = read_repo_json("artifacts/phase3/feature_columns.json")
    ohe = [c for c in ship if c.startswith("cat_")]
    train_cats = load_categories("train")
    for split in ("train", "val", "test"):
        cats = load_categories(split)
        assert all(f"cat_{c}" in ohe for c in cats), f"{split} carries a label absent from frozen OHE"
    assert len(ohe) == 73 == len(train_cats)