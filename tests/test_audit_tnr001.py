from __future__ import annotations

import pandas as pd
import pytest

import tenure
from tenure import AuditBlockedError, StudyDesign
from tenure.audit.report import Status


def _design(rows, *, active_as_of="2026-05-31", analysis_start="2024-01-01", **kwargs):
    df = pd.DataFrame(rows)
    return StudyDesign.from_event_dates(
        df,
        id_col="cid",
        origin_col="start",
        churn_date_col="churn",
        active_as_of=active_as_of,
        analysis_start=analysis_start,
        **kwargs,
    )


# An old customer (pre-analysis_start signup) still active at the snapshot, plus one
# in-window customer.
_ROWS = [
    {"cid": "old", "start": "2022-05-01", "churn": None},
    {"cid": "new", "start": "2025-01-01", "churn": None},
]


def _tnr001(report):
    return next(r for r in report.results if r.check_id == "TNR001")


def test_block_on_unmodeled_window_cut():
    design = _design(_ROWS, includes_pre_entry_churners=False)
    with pytest.raises(AuditBlockedError) as excinfo:
        tenure.audit(design, strictness="block")
    assert any(r.check_id == "TNR001" for r in excinfo.value.report.blocks)


def test_warn_downgrade_under_warn_strictness():
    design = _design(_ROWS, includes_pre_entry_churners=False)
    report = tenure.audit(design, strictness="warn")  # block -> warn, no raise
    assert report.ok
    assert _tnr001(report).status is Status.WARN


def test_pass_when_delayed_entry_modeled():
    design = _design(_ROWS, event_observed_from="2024-01-01")
    report = tenure.audit(design, strictness="block")
    assert report.ok
    assert _tnr001(report).status is Status.PASS


def test_pass_full_historical_cohort():
    design = _design(_ROWS, includes_pre_entry_churners=True)
    report = tenure.audit(design, strictness="block")
    assert _tnr001(report).status is Status.PASS


def test_warn_when_completeness_unknown():
    design = _design(_ROWS)  # includes_pre_entry_churners left as None
    report = tenure.audit(design, strictness="block")  # WARN does not raise
    assert _tnr001(report).status is Status.WARN


def test_clean_cohort_has_no_findings():
    rows = [
        {"cid": "a", "start": "2025-01-01", "churn": None},
        {"cid": "b", "start": "2025-03-01", "churn": "2025-06-01"},
    ]
    design = _design(rows)  # all origins after analysis_start
    report = tenure.audit(design, strictness="block")
    assert report.clean
    assert _tnr001(report).status is Status.PASS
