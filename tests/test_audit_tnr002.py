from __future__ import annotations

import pandas as pd
import pytest

import tenure
from tenure import AuditBlockedError, StudyDesign
from tenure.audit.report import Status


def _all_at_window_df(n=10):
    return pd.DataFrame(
        {
            "cid": [f"c{i}" for i in range(n)],
            "start": ["2024-01-01"] * n,  # every origin == analysis_start
            "churn": [None] * n,
        }
    )


def _design(df, **kwargs):
    return StudyDesign.from_event_dates(
        df,
        id_col="cid",
        origin_col="start",
        churn_date_col="churn",
        active_as_of="2026-05-31",
        analysis_start="2024-01-01",
        **kwargs,
    )


def _tnr002(report):
    return next(r for r in report.results if r.check_id == "TNR002")


def test_pass_when_origins_spread():
    df = tenure.load_svod_demo(with_left_truncation=False, seed=0)
    design = tenure.StudyDesign.from_event_dates(
        df,
        id_col="customer_id",
        origin_col="signup_date",
        churn_date_col="churn_date",
        active_as_of="2026-05-31",
        analysis_start="2024-01-01",
    )
    assert _tnr002(tenure.audit(design, strictness="block")).status is Status.PASS


def test_block_when_origins_collapsed_to_window():
    design = _design(_all_at_window_df())
    with pytest.raises(AuditBlockedError) as excinfo:
        tenure.audit(design, strictness="block")
    assert any(r.check_id == "TNR002" for r in excinfo.value.report.blocks)


def test_attestation_clears_window_origin():
    design = _design(_all_at_window_df(), attest_origin_correct=True)
    assert _tnr002(tenure.audit(design, strictness="block")).status is Status.PASS


def test_not_applicable_without_analysis_start():
    df = _all_at_window_df()
    design = StudyDesign.from_event_dates(
        df, id_col="cid", origin_col="start", churn_date_col="churn", active_as_of="2026-05-31"
    )
    report = tenure.audit(design, strictness="block")
    assert all(r.check_id != "TNR002" for r in report.results)
