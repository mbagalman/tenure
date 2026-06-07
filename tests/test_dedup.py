from __future__ import annotations

import pandas as pd
import pytest

from tenure import StudyDesign, TenureValidationError


def _dup_df():
    return pd.DataFrame(
        {
            "cid": ["a", "a", "b"],
            "start": ["2025-01-01", "2025-01-01", "2025-02-01"],
            "churn": ["2025-02-01", "2025-03-01", None],
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


def test_error_is_the_default():
    with pytest.raises(TenureValidationError):
        _build(_dup_df())


def test_keep_first_is_silent():
    sd = _build(_dup_df(), dedup_policy="keep-first")
    assert sd.n == 2


def test_keep_most_recent_warns():
    with pytest.warns(UserWarning, match="keep-most-recent"):
        sd = _build(_dup_df(), dedup_policy="keep-most-recent")
    assert sd.n == 2


def test_unknown_policy_raises():
    with pytest.raises(TenureValidationError):
        _build(_dup_df(), dedup_policy="bogus")
