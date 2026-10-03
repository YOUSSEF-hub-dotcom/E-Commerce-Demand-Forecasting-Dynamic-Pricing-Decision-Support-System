"""Part-3 B-cohort error analysis guards (Train + Validation only).

Encodes the authorized B-cohort diagnosis contract established by
``src/models/b_cohort_error_analysis.py``:

* The analysis is *diagnosis only*: it never reads ``data/splits/test.csv``,
  never predicts on the Test split, and never modifies, retrains, or tunes the
  frozen V2 model (loaded read-only from ``models:/demand_forecasting_final/2``).
* The emitted report embeds the explicit Test Protection Statement verbatim
  (machine-readable ``test_protection_statement`` field in the JSON companion
  and the literal statement in the Markdown deliverable).
* The Validation reproduction of the frozen V2 metrics is exact:
  global val WAPE 0.3964 and B-cohort val WAPE 0.5101460932701322 (the aligned
  V2 validation references).
* Cohort definitions are the official ones: the module consumes the same
  ``cohort_flags`` function and the same B / B2 constants as the final gate.

These tests are read-only: JSON/document reads and file-system checks only. They
never create runs, versions, tags, transitions, or rewrite files.
"""

import json
from pathlib import Path

import pytest

import src.models.b_cohort_error_analysis as bca
import src.models.final_forecast as ff

ROOT = Path(__file__).resolve().parents[1]

REPORT_JSON = ROOT / "reports" / "forecasting" / "part3_b_cohort_error_analysis.json"
REPORT_MD = ROOT / "reports" / "forecasting" / "part3_b_cohort_error_analysis.md"
MODULE = ROOT / "src" / "models" / "b_cohort_error_analysis.py"

PROTECTION_STATEMENT = (
    "No new experimentation, tuning, feature selection, or model decision was "
    "based on the Test split. The Test split (data/splits/test.csv) is never "
    "read by this analysis; the B WAPE=0.628239 result from the post-hoc "
    "evaluation is preserved as fixed historical evidence, and Validation is "
    "used as the development/selection evaluation set."
)

VAL_GLOBAL_WAPE_REF = 0.3964
VAL_B_WAPE_REF = 0.5101460932701322


def read_json(rel: str) -> dict:
    return json.loads((ROOT / rel).read_text(encoding="utf-8"))


def test_report_json_exists_and_is_valid():
    assert REPORT_JSON.is_file()
    data = json.loads(REPORT_JSON.read_text(encoding="utf-8"))
    assert "profile" in data and "reproducibility" in data


def test_json_embeds_verbatim_test_protection_statement():
    data = json.loads(REPORT_JSON.read_text(encoding="utf-8"))
    assert data["test_protection_statement"] == PROTECTION_STATEMENT


def test_md_report_exists_and_embeds_test_protection_statement():
    assert REPORT_MD.is_file()
    assert PROTECTION_STATEMENT in REPORT_MD.read_text(encoding="utf-8")


def test_validation_metrics_reproduced_exactly():
    data = json.loads(REPORT_JSON.read_text(encoding="utf-8"))
    rep = data["reproducibility"]
    assert rep["global_wape_matches"] is True
    assert rep["B_wape_matches"] is True
    assert abs(rep["global_wape_recomputed"] - VAL_GLOBAL_WAPE_REF) <= 2e-3
    assert abs(rep["B_wape_recomputed"] - VAL_B_WAPE_REF) <= 2e-3


def test_b_cohort_profile_frozen_reference():
    data = json.loads(REPORT_JSON.read_text(encoding="utf-8"))
    b = data["profile"]["B-sustained"]
    assert b["products"] == 37
    assert b["rows"] == 589
    assert abs(b["wape"] - VAL_B_WAPE_REF) <= 1e-9
    assert abs(b["bias_pct"] - (-18.227)) <= 0.001


def test_module_never_reads_test_split():
    src = MODULE.read_text(encoding="utf-8")
    assert "read_csv" not in src and "read_parquet" not in src
    assert "splits/test" not in src.replace("st split (data/splits/test.csv)", ""), (
        "path `data/splits/test.csv` must appear only inside the verbatim disclosure"
    )
    assert src.count("splits/test.csv") == 1


def test_module_uses_frozen_v2_and_scores_validation_only():
    assert bca.MODEL_VERSION == 2
    assert bca.MODEL_URI == "models:/demand_forecasting_final/2"
    src = MODULE.read_text(encoding="utf-8")
    assert "build_prepared(schema_len_expected=93)" in src
    assert "freeze_schema(train)" in src
    assert ".astype(\"float64\")" in src
    assert "cohort_flags(train" in src
    assert "test_X" not in src and "pred_demand_holdout" not in src


def test_cohort_function_is_the_official_cohort_flags():
    assert bca.cohort_flags is ff.cohort_flags
    assert bca.COHORT_ORDER == ["B-sustained", "B2-mid", "C-long-tail"]


def test_cohort_constants_match_the_official_gate():
    assert ff.B_MIN_POS_WEEKS == 6
    assert ff.B_MIN_AVG_WEEKLY == 1.0
    assert ff.B2_MIN_AVG_WEEKLY == 0.5
    assert bca.COHORT_ORDER == ["B-sustained", "B2-mid", "C-long-tail"]


def test_report_covers_all_required_sections():
    md = REPORT_MD.read_text(encoding="utf-8")
    for heading in [
        "B-cohort profile",
        "Error analysis",
        "B vs B2 vs C",
        "Root-cause hypotheses",
        "Recommended next experiment",
        "Explicit Test Protection Statement",
    ]:
        assert heading in md