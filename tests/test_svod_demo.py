"""End-to-end Phase 0 slice: the synthetic SVOD trap flows through StudyDesign + audit."""

from __future__ import annotations

import pandas as pd
import pytest

import tenure
from tenure import AuditBlockedError
from tenure.datasets import ACTIVE_AS_OF, ANALYSIS_START


def test_load_is_deterministic_and_shaped():
    a = tenure.load_svod_demo(seed=0)
    b = tenure.load_svod_demo(seed=0)
    pd.testing.assert_frame_equal(a, b)
    assert {"customer_id", "signup_date", "churn_date", "plan", "channel"}.issubset(a.columns)
    assert len(a) > 0


def test_truncation_scenario_has_pre_window_customers():
    df = tenure.load_svod_demo(with_left_truncation=True, seed=1)
    assert (df["signup_date"] < ANALYSIS_START).any()


def test_clean_scenario_has_no_pre_window_customers():
    df = tenure.load_svod_demo(with_left_truncation=False, seed=1)
    assert (df["signup_date"] >= ANALYSIS_START).all()


def _study(df, **kwargs):
    return tenure.StudyDesign.from_event_dates(
        df,
        id_col="customer_id",
        origin_col="signup_date",
        churn_date_col="churn_date",
        active_as_of=ACTIVE_AS_OF,
        analysis_start=ANALYSIS_START,
        **kwargs,
    )


def test_naive_truncated_design_blocks():
    df = tenure.load_svod_demo(with_left_truncation=True, seed=2)
    study = _study(df, includes_pre_entry_churners=False)
    with pytest.raises(AuditBlockedError) as excinfo:
        tenure.audit(study, strictness="block")
    assert any(r.check_id == "TNR001" for r in excinfo.value.report.blocks)


def test_corrected_design_passes():
    df = tenure.load_svod_demo(with_left_truncation=True, seed=2)
    study = _study(df, event_observed_from=ANALYSIS_START)
    report = tenure.audit(study, strictness="block")
    assert report.ok


def test_clean_demo_is_clean():
    df = tenure.load_svod_demo(with_left_truncation=False, seed=3)
    report = tenure.audit(_study(df), strictness="block")
    assert report.clean


def test_truth_constants_are_sane():
    truth = tenure.svod_demo_truth()
    assert truth.survival_at(0) == 1.0
    assert 0.0 < truth.survival_at(365) < 1.0
    assert truth.rmst_days(365) > 0.0
    assert truth.ltv(10.0, horizon_days=365) > 0.0
    # RMST through a horizon never exceeds the horizon itself.
    assert truth.rmst_days(365) <= 365.0
