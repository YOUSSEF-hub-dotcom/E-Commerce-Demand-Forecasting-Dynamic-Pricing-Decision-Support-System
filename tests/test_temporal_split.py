"""Unit tests for the chronological temporal split (no-leakage contract).

Proves: chronological ordering, strict non-overlap, exact full coverage,
determinism, ratio correctness, and the fail-loud guards.
"""

import pandas as pd
import pytest

from src.data.temporal_split import assert_chronological_order, temporal_split


def make_series(n: int = 100) -> pd.DataFrame:
    """Synthetic chronologically-sortable frame."""
    return pd.DataFrame(
        {
            "value": range(n),
            "t": pd.to_datetime("2017-01-01") + pd.to_timedelta(range(n), unit="D"),
        }
    )


class TestTemporalSplit:
    """Chronological, no-leakage partitioning."""

    def test_split_shapes_match_ratios(self):
        df = make_series(100)
        splits = temporal_split(df, "t", ratios=(0.70, 0.15, 0.15))
        assert len(splits["train"]) == 70
        assert len(splits["val"]) == 15
        assert len(splits["test"]) == 15

    def test_full_coverage_no_rows_lost(self):
        df = make_series(137)  # awkward length exercises rounding.
        splits = temporal_split(df, "t")
        combined = pd.concat(splits.values()).sort_values("t").reset_index(drop=True)
        assert len(combined) == len(df)
        assert set(combined["value"]) == set(df["value"])

    def test_strict_non_overlap(self):
        df = make_series(200)
        splits = temporal_split(df, "t")
        sets = [set(s["value"]) for s in splits.values()]
        assert sets[0].isdisjoint(sets[1])
        assert sets[1].isdisjoint(sets[2])
        assert sets[0].isdisjoint(sets[2])

    def test_chronological_guard_passes(self):
        df = make_series(200)
        splits = temporal_split(df, "t")
        # No exception raised = ordering valid.
        assert_chronological_order(splits, "t")
        assert splits["train"]["t"].max() <= splits["val"]["t"].min()
        assert splits["val"]["t"].max() <= splits["test"]["t"].min()

    def test_determinism(self):
        df = make_series(200)
        s1 = temporal_split(df, "t")
        s2 = temporal_split(df, "t")
        for key in s1:
            pd.testing.assert_frame_equal(s1[key], s2[key])

    def test_ratios_must_sum_to_one(self):
        with pytest.raises(ValueError):
            temporal_split(make_series(10), "t", ratios=(0.5, 0.5, 0.5))

    def test_empty_frame_raises(self):
        with pytest.raises(ValueError):
            temporal_split(make_series(0), "t")

    def test_missing_date_column_raises(self):
        with pytest.raises(ValueError):
            temporal_split(make_series(10), "nope")