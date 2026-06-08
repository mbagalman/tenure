from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import tenure
from tenure import StudyDesign

CUTOFF = "2024-06-01"  # tenure 152 from a 2024-01-01 origin (2024 is a leap year)
ACTIVE_AS_OF = "2025-01-01"


def _events_df() -> pd.DataFrame:
    # A churns before the cutoff; B is active at the cutoff and churns after; C is active through
    # the end of observation; E signs up after the cutoff (not at risk at it).
    return pd.DataFrame(
        {
            "cid": ["A", "B", "C", "E"],
            "signup": pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-01", "2024-08-01"]),
            "churn": [pd.Timestamp("2024-03-01"), pd.Timestamp("2024-09-01"), pd.NaT, pd.NaT],
        }
    )


def _events_design() -> StudyDesign:
    return StudyDesign.from_event_dates(
        _events_df(),
        id_col="cid",
        origin_col="signup",
        churn_date_col="churn",
        active_as_of=ACTIVE_AS_OF,
    )


def test_test_cohort_membership_and_eval_clock():
    _, test = tenure.temporal_holdout(_events_design(), CUTOFF)
    # B (churns after) and C (active through end) are at risk at the cutoff. A churned before it;
    # E signed up after it -- both excluded from the test cohort.
    assert set(test.ids) == {"B", "C"}
    t = test.table.set_index("id")
    assert np.isclose(t.loc["B", "eval_start"], 152.0)  # tenure at cutoff
    assert t.loc["B", "eval_event"] == 1
    assert np.isclose(t.loc["B", "eval_duration"], 92.0)  # 2024-06-01 -> 2024-09-01
    assert t.loc["C", "eval_event"] == 0
    assert np.isclose(t.loc["C", "eval_duration"], 214.0)  # 2024-06-01 -> 2025-01-01
    assert test.prediction_time == pd.Timestamp(CUTOFF)


def test_train_is_censored_at_cutoff_with_no_leakage():
    train, _ = tenure.temporal_holdout(_events_design(), CUTOFF)
    table = train.derive()
    # Only A's pre-cutoff churn is an event in training; B and C are censored at the cutoff; E is
    # excluded (signed up after the cutoff).
    assert set(table["id"]) == {"A", "B", "C"}
    assert int(table["event"].sum()) == 1
    # Leakage guarantee: no training event ends after the cutoff.
    events = table[table["event"] == 1]
    end_dates = events["origin"] + pd.to_timedelta(events["exit_tenure"], unit="D")
    assert (end_dates <= pd.Timestamp(CUTOFF)).all()
    # The censored-at-cutoff train design still fits the existing estimators unchanged.
    assert tenure.KaplanMeier().fit(train).survival_ is not None


def _interval_panel() -> pd.DataFrame:
    o = pd.Timestamp("2024-01-01")

    def d(days):
        return o + pd.Timedelta(days=days)

    # P: [0,100) low -> [100,250) high, churns at 250 (the second interval crosses the cutoff at
    # tenure 152). Q: a single interval [200,300) -- its delayed entry is after the cutoff.
    return pd.DataFrame(
        {
            "cid": ["P", "P", "Q"],
            "origin": [o, o, o],
            "start": [d(0), d(100), d(200)],
            "end": [d(100), d(250), d(300)],
            "event": [0, 1, 0],
            "usage": ["low", "high", "mid"],
        }
    )


def _interval_design() -> StudyDesign:
    return StudyDesign.from_intervals(
        _interval_panel(),
        id_col="cid",
        origin_col="origin",
        interval_start_col="start",
        interval_end_col="end",
        event_col="event",
        covariate_cols=["usage"],
    )


def test_interval_crossing_is_split_and_late_entry_excluded():
    train, test = tenure.temporal_holdout(_interval_design(), CUTOFF)
    # Q's entry (tenure 200) is after the cutoff (152) -> in neither train nor test.
    assert set(test.ids) == {"P"}
    assert "Q" not in train.derive()["id"].tolist()

    row = test.table.set_index("id").loc["P"]
    assert np.isclose(row["eval_start"], 152.0)
    assert np.isclose(row["eval_duration"], 98.0)  # 250 - 152
    assert row["eval_event"] == 1
    assert row["usage"] == "high"  # covariate as of the cutoff (the crossing interval)

    # P's post-cutoff path starts exactly at the cutoff tenure (152), not at 100.
    p_path = test.paths[test.paths["id"] == "P"].sort_values("start")
    start0 = (p_path["start"].iloc[0] - pd.Timestamp("2024-01-01")) / pd.Timedelta(days=1)
    assert np.isclose(start0, 152.0)

    # P's training spell is censored at the cutoff: no event, last exit at tenure 152.
    p_train = train.derive().query("id == 'P'")
    assert int(p_train["event"].sum()) == 0
    assert np.isclose(p_train["exit_tenure"].max(), 152.0)


def test_random_split_warns_val001_and_partitions_disjointly():
    with pytest.warns(UserWarning, match="VAL001"):
        train_ids, test_ids = tenure.random_split(_events_design(), test_fraction=0.5, seed=0)
    assert set(train_ids).isdisjoint(set(test_ids))


def test_validation_result_carries_contract():
    result = tenure.ValidationResult(
        table=pd.DataFrame({"metric": ["demo"], "estimate": [0.7]}),
        metadata={"metric": "demo", "estimate": 0.7},
    )
    assert result.estimate == 0.7
    assert "demo" in repr(result)


def test_cutoff_after_everyone_raises():
    with pytest.raises(tenure.TenureValidationError, match="at risk"):
        tenure.temporal_holdout(_events_design(), "2030-01-01")


def test_temporal_holdout_blocks_on_unaudited_unmapped_design():
    # Repro for [P1]: an unaudited design with unmapped rows must raise TenureValidationError.
    df = pd.DataFrame(
        {
            "cid": [1, 2],
            "signup": pd.to_datetime(["2024-01-01", "2024-01-01"]),
            "exit": pd.to_datetime(["2024-06-01", "2024-07-01"]),
            "status": ["churn", "upgraded"],
        }
    )
    status_map = {"churn": "event"}  # "upgraded" is unmapped
    design = StudyDesign.from_status(
        df,
        id_col="cid",
        origin_col="signup",
        exit_col="exit",
        status_col="status",
        status_map=status_map,
        active_as_of="2025-01-01",
    )
    assert design.n_unmapped == 1
    # Bypassing ensure_estimable() by splitting should be blocked
    with pytest.raises(tenure.TenureValidationError, match="status_map"):
        tenure.temporal_holdout(design, CUTOFF)


def test_temporal_holdout_cutoff_boundary_is_inclusive():
    # Repro for [P2]: verify that an event exactly on the cutoff date
    # is preserved in train and excluded from test.
    df = pd.DataFrame(
        {
            "cid": ["F", "G"],
            "signup": pd.to_datetime(["2024-01-01", "2024-01-01"]),
            "churn": [pd.Timestamp(CUTOFF), pd.Timestamp(CUTOFF) + pd.Timedelta(days=1)],
        }
    )
    design = StudyDesign.from_event_dates(
        df,
        id_col="cid",
        origin_col="signup",
        churn_date_col="churn",
        active_as_of=ACTIVE_AS_OF,
    )
    train, test = tenure.temporal_holdout(design, CUTOFF)

    # F exits exactly at cutoff -> event preserved in train, excluded from test
    assert "F" in train.derive()["id"].tolist()
    assert train.derive().set_index("id").loc["F", "event"] == 1
    assert "F" not in test.ids

    # G exits after cutoff -> censored in train, included in test
    assert "G" in train.derive()["id"].tolist()
    assert train.derive().set_index("id").loc["G", "event"] == 0
    assert "G" in test.ids
