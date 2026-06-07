from __future__ import annotations

import pandas as pd

import tenure
from tenure import Status, StudyDesign


def _tnr004(report):
    return next(r for r in report.results if r.check_id == "TNR004")


def test_naive_static_shows_illusory_protection_corrected_recovers_truth():
    r = tenure.naive_vs_corrected_immortal_demo(seed=0)
    assert r["true_hazard_ratio"] == 1.0
    # Naive static 'ever-upgraded' invents a protective effect (HR well below the true 1.0).
    assert r["naive_hazard_ratio"] < 0.7
    # The time-varying encoding recovers the truth (HR ~ 1)...
    assert abs(r["corrected_hazard_ratio"] - 1.0) < 0.1
    # ...and is strictly closer to it than the biased naive estimate.
    assert abs(r["corrected_hazard_ratio"] - 1.0) < abs(r["naive_hazard_ratio"] - 1.0)


def test_immortal_demo_values_are_pinned():
    # Deterministic regression gate (seed 0), the time-varying analogue of NFR-CORR-3.
    r = tenure.naive_vs_corrected_immortal_demo(seed=0)
    assert abs(r["naive_hazard_ratio"] - 0.615356) < 5e-3
    assert abs(r["corrected_hazard_ratio"] - 1.024471) < 5e-3


def test_demo_audits_warn_naive_and_pass_interval():
    r = tenure.naive_vs_corrected_immortal_demo(seed=0)
    assert _tnr004(r["naive_audit"]).status is Status.WARN
    assert _tnr004(r["corrected_audit"]).status is Status.PASS


def _late_upgrade_intervals(n: int = 40) -> pd.DataFrame:
    # `upgraded` = 1 appears ONLY on the late [60, 120) interval, so a STATIC immortal-time
    # heuristic would flag it (min tenure | X=1 sits far above 0). An interval design must not.
    origin = pd.Timestamp("2024-01-01")
    rows = []
    for i in range(n):
        cid = f"u{i}"
        churn = i % 3 == 0
        rows.append(
            {
                "cid": cid,
                "origin": origin,
                "start": origin,
                "end": origin + pd.Timedelta(days=60),
                "event": 0,
                "upgraded": 0,
            }
        )
        rows.append(
            {
                "cid": cid,
                "origin": origin,
                "start": origin + pd.Timedelta(days=60),
                "end": origin + pd.Timedelta(days=120),
                "event": int(churn),
                "upgraded": 1,
            }
        )
    return pd.DataFrame(rows)


def test_interval_design_passes_tnr004_without_crying_wolf():
    design = StudyDesign.from_intervals(
        _late_upgrade_intervals(),
        id_col="cid",
        origin_col="origin",
        interval_start_col="start",
        interval_end_col="end",
        event_col="event",
        covariate_cols=["upgraded"],
    )
    report = tenure.audit(design, strictness="warn")
    assert _tnr004(report).status is Status.PASS
    # The immortal-time check is the one that would have fired on a static 'ever-upgraded' frame.
    assert "TNR004" not in [w.check_id for w in report.warnings]
