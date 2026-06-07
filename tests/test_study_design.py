from __future__ import annotations

import pandas as pd
import pytest

from tenure import StudyDesign, TenureValidationError
from tenure._frame import ENTRY, EVENT, EXIT, ID, ORIGIN, STATUS


def _df():
    return pd.DataFrame(
        {
            "cid": ["a", "b", "c"],
            "start": ["2025-01-01", "2025-02-01", "2024-06-01"],
            "churn": ["2025-03-01", None, None],
        }
    )


def _build(df, **kwargs):
    return StudyDesign.from_event_dates(
        df,
        id_col="cid",
        origin_col="start",
        churn_date_col="churn",
        active_as_of="2025-06-01",
        **kwargs,
    )


def test_basic_derivation():
    sd = _build(_df())
    table = sd.derive()
    assert list(table.columns)[:6] == [ID, ORIGIN, ENTRY, EXIT, EVENT, STATUS]
    assert table[EVENT].tolist() == [1, 0, 0]
    assert table[STATUS].tolist() == ["churn", "active", "active"]
    # 'a' churned 2025-03-01 from 2025-01-01 -> 59 days
    assert abs(table[EXIT].iloc[0] - 59.0) < 1e-9
    assert (table[ENTRY] == 0.0).all()
    assert sd.entry_modeled is False
    assert sd.n == 3


def test_missing_column_raises():
    with pytest.raises(TenureValidationError):
        _build(_df().rename(columns={"start": "origin_date"}))


def test_duplicate_id_raises():
    df = _df()
    df.loc[2, "cid"] = "a"
    with pytest.raises(TenureValidationError):
        _build(df)


def test_event_observed_from_models_delayed_entry():
    sd = _build(_df(), event_observed_from="2024-09-01")
    assert sd.entry_modeled is True
    table = sd.derive()
    # 'c' started 2024-06-01, observed-from 2024-09-01 -> entry ~92 days
    assert table[ENTRY].iloc[2] > 80.0
    # 'a'/'b' started in 2025 (after observed-from) -> entry clipped to 0
    assert table[ENTRY].iloc[0] == 0.0


def test_group_cols_preserved():
    df = _df()
    df["plan"] = ["basic", "premium", "standard"]
    sd = _build(df, group_cols=["plan"])
    assert "plan" in sd.derive().columns
    assert sd.group_cols == ["plan"]
