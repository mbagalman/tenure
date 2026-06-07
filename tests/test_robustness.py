"""FR-RB-1: degenerate inputs give a clear result or a clear error -- never a crash."""

from __future__ import annotations

import pandas as pd
import pytest

import tenure
from tenure import StudyDesign, TenureValidationError


def _event_design(rows, *, active_as_of="2026-05-31", **kwargs):
    df = pd.DataFrame(rows)
    return StudyDesign.from_event_dates(
        df,
        id_col="cid",
        origin_col="start",
        churn_date_col="churn",
        active_as_of=active_as_of,
        **kwargs,
    )


def _summary(design):
    km = tenure.KaplanMeier().fit(design)
    return tenure.summarize(km, period_margin=12.0, ltv_horizon=365.0)


def test_all_censored_cohort_is_well_defined():
    # Everyone still active -> KM survival stays 1.0; no crash, retention is high.
    design = _event_design(
        [{"cid": f"c{i}", "start": "2025-01-01", "churn": None} for i in range(20)]
    )
    km = tenure.KaplanMeier().fit(design)
    retention = tenure.retention_at(km, [30, 180])
    assert (retention["retention"] == 1.0).all()
    _summary(design)  # does not crash


def test_single_customer():
    design = _event_design([{"cid": "a", "start": "2025-01-01", "churn": "2025-03-01"}])
    assert _summary(design).table.shape[0] == 1  # one "overall" row


def test_identical_tenures():
    rows = [{"cid": f"c{i}", "start": "2025-01-01", "churn": "2025-02-01"} for i in range(10)]
    _summary(_event_design(rows))  # all churn at the same tenure; no crash


def test_immediate_churn_tenure_zero():
    rows = [{"cid": f"c{i}", "start": "2025-01-01", "churn": "2025-01-01"} for i in range(10)]
    _summary(_event_design(rows))  # tenure 0; must not crash


def test_all_events_positive_tenure_drops_retention():
    rows = [{"cid": f"c{i}", "start": "2024-01-01", "churn": "2024-04-01"} for i in range(30)]
    km = tenure.KaplanMeier().fit(_event_design(rows))
    # Everyone churned by ~90 days -> retention at 180 is 0.
    assert tenure.retention_at(km, [180], min_at_risk=1)["retention"].iloc[0] == 0.0


def test_zero_rows_raises_clear_error():
    df = pd.DataFrame(
        {"cid": ["a"], "start": ["2025-01-01"], "exit": ["2025-02-01"], "status": ["upgrade"]}
    )
    with pytest.raises(TenureValidationError, match="zero rows"):
        StudyDesign.from_status(
            df,
            id_col="cid",
            origin_col="start",
            exit_col="exit",
            status_col="status",
            status_map={"upgrade": "exclude"},  # the only row is excluded -> empty
            active_as_of="2025-06-01",
        )


def test_null_exit_raises_clear_error():
    df = pd.DataFrame(
        {
            "cid": ["a", "b"],
            "start": ["2025-01-01", "2025-01-01"],
            "exit": ["2025-02-01", None],
            "status": ["churn", "active"],
        }
    )
    with pytest.raises(TenureValidationError, match="null"):
        StudyDesign.from_status(
            df,
            id_col="cid",
            origin_col="start",
            exit_col="exit",
            status_col="status",
            status_map={"churn": "event", "active": "censored"},
            active_as_of="2025-06-01",
        )
