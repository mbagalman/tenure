from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from lifelines import KaplanMeierFitter

import tenure
from tenure import StudyDesign, TenureValidationError
from tenure._frame import ENTRY, EVENT, EXIT, as_estimator_frame


def _interval_df():
    # A: two intervals, churns at the end of the second; B: single censored interval;
    # C: two intervals, censored. Covariate `usage` changes over time.
    return pd.DataFrame(
        {
            "cid": ["A", "A", "B", "C", "C"],
            "origin": ["2024-01-01"] * 5,
            "start": ["2024-01-01", "2024-02-01", "2024-01-01", "2024-01-01", "2024-02-01"],
            "end": ["2024-02-01", "2024-03-01", "2024-04-01", "2024-02-01", "2024-05-01"],
            "event": [0, 1, 0, 0, 0],
            "usage": ["low", "high", "low", "low", "high"],
        }
    )


def _build(df=None, **kwargs):
    return StudyDesign.from_intervals(
        _interval_df() if df is None else df,
        id_col="cid",
        origin_col="origin",
        interval_start_col="start",
        interval_end_col="end",
        event_col="event",
        covariate_cols=["usage"],
        **kwargs,
    )


def test_from_intervals_builds_multi_row_canonical():
    design = _build()
    table = design.derive()
    assert design.interval is True
    assert design.n == 5  # one row per interval, repeated ids allowed
    # A's 2nd interval: entry 31d (2024-02-01 - origin), exit 60d (2024-03-01 - origin), event.
    a2 = table.iloc[1]
    assert abs(a2[ENTRY] - 31.0) < 1e-9
    assert abs(a2[EXIT] - 60.0) < 1e-9
    assert a2[EVENT] == 1
    assert table[EVENT].tolist() == [0, 1, 0, 0, 0]
    assert "usage" in table.columns  # time-varying covariate carried


def test_counting_process_km_matches_lifelines():
    design = _build()
    km = tenure.KaplanMeier().fit(design)
    ef = as_estimator_frame(design.derive())
    kmf = KaplanMeierFitter().fit(durations=ef.duration, event_observed=ef.event, entry=ef.entry)
    times = np.array([15.0, 45.0, 75.0, 120.0])
    ours = km.survival_at(times, group="overall").sort_values("time")["survival"].to_numpy()
    assert np.allclose(ours, kmf.survival_function_at_times(times).to_numpy(), atol=1e-9)


def test_start_after_end_raises():
    df = _interval_df()
    df.loc[0, "end"] = "2023-12-01"  # end before start
    with pytest.raises(TenureValidationError, match="interval_start < interval_end"):
        _build(df)


def test_non_contiguous_intervals_raise():
    df = _interval_df()
    df.loc[1, "start"] = "2024-02-15"  # gap: previous interval ended 2024-02-01
    with pytest.raises(TenureValidationError, match="contiguous"):
        _build(df)


def test_event_before_terminal_interval_raises():
    df = _interval_df()
    df.loc[0, "event"] = 1  # event on A's first (non-terminal) interval
    with pytest.raises(TenureValidationError, match="terminal"):
        _build(df)


def test_origin_varies_within_id_raises():
    df = _interval_df()
    df.loc[1, "origin"] = "2024-06-01"  # A's two rows now disagree on origin
    with pytest.raises(TenureValidationError, match="origin varies"):
        _build(df)


def test_single_spell_design_is_not_flagged_interval():
    df = tenure.load_svod_demo(with_left_truncation=False, seed=0)
    design = tenure.StudyDesign.from_event_dates(
        df,
        id_col="customer_id",
        origin_col="signup_date",
        churn_date_col="churn_date",
        active_as_of="2026-05-31",
    )
    assert design.interval is False
