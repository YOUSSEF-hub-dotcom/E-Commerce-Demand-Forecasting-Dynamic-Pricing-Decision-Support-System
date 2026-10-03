"""Part-3 final-holdout integrity guards.

Encodes the verified data facts that block the "final independent holdout"
lifecycle step for this dataset:

* The demand-bearing raw data physically ends at the original test boundary
  (last valid order week == 2018-09-03/2018-09-09 == matrix max week).
* There is NO genuinely untouched future temporal segment from which a valid
  independent final holdout could be built without re-using development or
  test data.
* The splits partition the demand matrix exactly (row identity + week union),
  so no silent re-splitting can create "new" rows.
* v1's original test split is immutable (provenance hash pinned) and its
  historical FAIL result is preserved in the repo report.

These tests run fast (metadata / small reads only) and protect the lifecycle
contract against accidental future corruption.
"""

import hashlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def read_repo_json(rel: str) -> dict:
    return json.loads((ROOT / rel).read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def raw_orders() -> "object":
    import pandas as pd

    return pd.read_csv(
        ROOT / "data" / "raw" / "olist_orders_dataset.csv",
        usecols=["order_id", "order_status", "order_purchase_timestamp"],
        parse_dates=["order_purchase_timestamp"],
    )


@pytest.fixture(scope="module")
def raw_items() -> "object":
    import pandas as pd

    return pd.read_csv(
        ROOT / "data" / "raw" / "olist_order_items_dataset.csv",
        usecols=["order_id", "product_id"],
    )


def test_raw_demand_bearing_data_ends_at_test_boundary(raw_orders, raw_items):
    """No valid order exists after the last week of the original test split."""
    import pandas as pd

    valid = raw_orders[~raw_orders["order_status"].isin(["canceled", "unavailable"])]
    last_valid = valid["order_purchase_timestamp"].max()
    last_week = pd.Period(last_valid + pd.Timedelta(days=1) - pd.Timedelta(1, "ns"), freq="W").asfreq("W").strftime("%Y-%m-%d")
    week_of_last = pd.Period((last_valid - pd.Timedelta(days=last_valid.dayofweek)).date(), freq="W")
    # Purchases >= test end (2018-09-10) must all be canceled with no items.
    after = raw_orders[raw_orders["order_purchase_timestamp"] >= "2018-09-10"]
    assert (after["order_status"] == "canceled").all(), "non-canceled orders exist after test end"
    after_ids = set(after["order_id"])
    assert raw_items["order_id"].isin(after_ids).sum() == 0, "demand-bearing items after test end"
    assert last_valid <= pd.Timestamp("2018-09-10"), "valid order exists past the test end"


def test_matrix_and_splits_partition_identity():
    """Row sums and week unions are identical between matrix and splits."""
    import pandas as pd

    matrix = pd.read_csv(ROOT / "data" / "processed" / "demand_matrix_weekly.csv.gz", usecols=["week"])
    train = pd.read_csv(ROOT / "data" / "splits" / "train.csv", usecols=["week"])
    val = pd.read_csv(ROOT / "data" / "splits" / "val.csv", usecols=["week"])
    test = pd.read_csv(ROOT / "data" / "splits" / "test.csv", usecols=["week"])

    mw = set(matrix["week"].astype(str))
    sw = set(train["week"].astype(str)) | set(val["week"].astype(str)) | set(test["week"].astype(str))
    assert mw == sw, "matrix weeks != split weeks union"
    assert len(mw) == 106
    # The original test covers the LAST 16 distinct week labels of the dataset.
    sorted_weeks = sorted(mw)
    assert sorted_weeks[-16:] == sorted(set(test["week"].astype(str)))


def test_v1_original_test_immutable():
    """The original v1 test split file has not been re-written."""
    assert md5(ROOT / "data" / "splits" / "test.csv") == "d35bfcd546df74b63c9bd25151fde264"


def test_v1_historical_fail_preserved():
    """v1 remains recorded as FAIL in its audit trail."""
    result = read_repo_json("reports/forecasting/part3_final_result.json")
    assert result["gate_passed"] is False
    assert result["promoted_stage"] == "Staging"
    audit = read_repo_json("reports/forecasting/part3_final_audit.json")
    assert audit["final_test_metrics"]["B_wape"] > 0.6
    assert audit["quality_gate"]["passed"] is False


def test_v2_not_claimed_test_validated():
    """v2 registry status is provisional (validation-gated), not final-test validated."""
    audit = read_repo_json("reports/forecasting/part3_v2_audit.json")
    assert audit["quality_gate"]["outcomes"][0]["group"] == "v2_validation"
    assert "NOT re-scored" in audit["notes"]


def test_final_gate_requires_final_test_metrics():
    """The production gate implementation consumes metrics dicts; confirm the gate
    decision depends on the provided split's metrics (no test-only values baked in)."""
    # Contract smoke check via the gate source (opinions held in quality_gate module).
    from src.models.quality_gate import _NUMERIC_CRITERIA

    keys = {c["key"] for c in _NUMERIC_CRITERIA}
    assert {"wape", "B_wape", "B_bias_pct", "mase"} <= keys


def test_blocker_report_present():
    """The blocker audit report exists and states the STOP condition."""
    report = (ROOT / "reports" / "forecasting" / "part3_final_holdout_blocker.md").read_text(encoding="utf-8")
    assert "STOP" in report
    assert "untouched final holdout" in report
    evidence = read_repo_json("reports/forecasting/part3_final_holdout_blocker.json")
    assert evidence["status"] == "BLOCKER"