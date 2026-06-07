"""NFR-AUDIT-1: a correctly designed cohort returns all-pass, zero warnings (no crying wolf)."""

from __future__ import annotations

import tenure
from tenure.audit.report import Status


def test_clean_cohort_passes_every_check():
    df = tenure.load_svod_demo(with_left_truncation=False, seed=0)
    design = tenure.StudyDesign.from_event_dates(
        df,
        id_col="customer_id",
        origin_col="signup_date",
        churn_date_col="churn_date",
        active_as_of="2026-05-31",
        analysis_start="2024-01-01",
        group_cols=["plan", "channel"],
    )
    report = tenure.audit(design, strictness="block")

    assert report.clean  # no blocks, no warnings
    assert all(r.status is Status.PASS for r in report.results)
    # The applicable registered checks actually ran (TNR003 is N/A for the event-date schema).
    fired = {r.check_id for r in report.results}
    assert {"TNR001", "TNR002", "TNR004"}.issubset(fired)
