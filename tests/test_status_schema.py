from __future__ import annotations

import pandas as pd
import pytest

import tenure
from tenure import StudyDesign, TenureValidationError
from tenure._frame import EVENT, STATUS
from tenure.audit.report import Status

# a: churn (event @ 2025-03-01); b/d: active (censored @ snapshot); c: upgrade (exclude);
# e: migration (censored @ 2025-02-15 -> BEFORE snapshot -> informative).
STATUS_MAP = {"churn": "event", "active": "censored", "upgrade": "exclude", "migration": "censored"}


def _status_df():
    return pd.DataFrame(
        {
            "cid": ["a", "b", "c", "d", "e"],
            "start": ["2025-01-01"] * 5,
            "exit": ["2025-03-01", "2025-06-01", "2025-04-01", "2025-06-01", "2025-02-15"],
            "status": ["churn", "active", "upgrade", "active", "migration"],
        }
    )


def _build(df=None, *, status_map=None, **kwargs):
    return StudyDesign.from_status(
        _status_df() if df is None else df,
        id_col="cid",
        origin_col="start",
        exit_col="exit",
        status_col="status",
        status_map=status_map or STATUS_MAP,
        active_as_of="2025-06-01",
        **kwargs,
    )


def _tnr003(report):
    return next(r for r in report.results if r.check_id == "TNR003")


def test_from_status_basic_mapping():
    design = _build()
    table = design.derive()
    assert design.n == 4  # upgrade row excluded
    assert design.n_excluded == 1
    assert table[EVENT].tolist() == [1, 0, 0, 0]  # only the churn is an event
    assert "upgrade" not in set(table[STATUS])  # excluded status is gone
    assert set(table[STATUS]) == {"churn", "active", "migration"}
    assert design.status_map == STATUS_MAP


def test_bad_intent_value_raises():
    with pytest.raises(TenureValidationError):
        _build(status_map={"churn": "event", "active": "keep"})


def test_duplicate_id_raises():
    df = _status_df()
    df.loc[4, "cid"] = "a"
    with pytest.raises(TenureValidationError):
        _build(df=df)


def test_informative_censoring_warns():
    design = _build()
    assert design.informative_censoring_statuses == ["migration"]
    report = tenure.audit(design, strictness="block")  # WARN does not raise
    assert _tnr003(report).status is Status.WARN
    assert "migration" in _tnr003(report).message


def test_unmapped_status_blocks():
    df = pd.concat(
        [
            _status_df(),
            pd.DataFrame(
                {
                    "cid": ["f"],
                    "start": ["2025-01-01"],
                    "exit": ["2025-05-01"],
                    "status": ["paused"],
                }
            ),
        ],
        ignore_index=True,
    )
    design = _build(df=df)
    assert design.unmapped_statuses == ["paused"]
    assert design.n_unmapped == 1
    with pytest.raises(tenure.AuditBlockedError) as excinfo:
        tenure.audit(design, strictness="block")
    assert any(r.check_id == "TNR003" for r in excinfo.value.report.blocks)
    # Bypassed under warn -> downgraded to a warning, no raise.
    assert _tnr003(tenure.audit(design, strictness="warn")).status is Status.WARN


def test_clean_status_design_passes():
    df = pd.DataFrame(
        {
            "cid": ["a", "b", "c"],
            "start": ["2025-01-01"] * 3,
            "exit": ["2025-03-01", "2025-06-01", "2025-06-01"],
            "status": ["churn", "active", "active"],
        }
    )
    design = StudyDesign.from_status(
        df,
        id_col="cid",
        origin_col="start",
        exit_col="exit",
        status_col="status",
        status_map={"churn": "event", "active": "censored"},
        active_as_of="2025-06-01",
    )
    report = tenure.audit(design, strictness="block")
    assert _tnr003(report).status is Status.PASS
    assert report.clean


def test_tnr003_not_applicable_to_event_date_schema():
    df = tenure.load_svod_demo(with_left_truncation=False, seed=0)
    design = tenure.StudyDesign.from_event_dates(
        df,
        id_col="customer_id",
        origin_col="signup_date",
        churn_date_col="churn_date",
        active_as_of="2026-05-31",
    )
    assert design.status_map is None
    assert design.n_excluded == 0
    report = tenure.audit(design, strictness="block")
    assert all(r.check_id != "TNR003" for r in report.results)


def test_from_status_feeds_estimators():
    design = _build()
    km = tenure.KaplanMeier().fit(design)
    out = tenure.rmst(km, horizon=120.0, min_at_risk=1)
    assert out.iloc[0]["rmst"] > 0
