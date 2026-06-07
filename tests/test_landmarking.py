from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import tenure
from tenure import CoxPH, StudyDesign, TenureValidationError, landmark
from tenure._frame import ENTRY


def _intervals(df: pd.DataFrame, covariate_cols: list[str]) -> StudyDesign:
    return StudyDesign.from_intervals(
        df,
        id_col="cid",
        origin_col="origin",
        interval_start_col="start",
        interval_end_col="end",
        event_col="event",
        covariate_cols=covariate_cols,
    )


def _handcrafted() -> pd.DataFrame:
    o = pd.Timestamp("2024-01-01")

    def d(days):
        return o + pd.Timedelta(days=days)

    # A: [0,30) low -> [30,90) high, churns at 90. B: [0,45) low, censored. C: [0,60) low ->
    # [60,120) high, censored at 120.
    return pd.DataFrame(
        {
            "cid": ["A", "A", "B", "C", "C"],
            "origin": [o] * 5,
            "start": [d(0), d(30), d(0), d(0), d(60)],
            "end": [d(30), d(90), d(45), d(60), d(120)],
            "event": [0, 1, 0, 0, 0],
            "usage": ["low", "high", "low", "low", "high"],
        }
    )


def test_landmark_keeps_at_risk_with_covariate_as_of_landmark():
    lm = landmark(_intervals(_handcrafted(), ["usage"]), 50.0)
    table = lm.derive()
    # B (final exit 45 <= 50) is dropped; A and C remain.
    assert set(table["id"]) == {"A", "C"}
    # Delayed entry at the landmark for everyone (clock unchanged).
    assert np.allclose(table[ENTRY].to_numpy(), 50.0)
    # Covariate as of L=50: A sits in [30,90)=high; C sits in [0,60)=low.
    usage = dict(zip(table["id"], table["usage"], strict=True))
    assert usage == {"A": "high", "C": "low"}
    # Terminal event is carried (A churns; C is censored).
    event = dict(zip(table["id"], table["event"], strict=True))
    assert event == {"A": 1, "C": 0}
    # The landmark design is itself an interval design (single interval per subject).
    assert lm.interval is True


def _many(n: int = 200, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    o = pd.Timestamp("2024-01-01")
    rows = []
    for i in range(n):
        cid = f"s{i}"
        score = float(rng.normal())
        churn = rng.random() < 0.5
        end = 90 if churn else 120
        rows.append(
            {
                "cid": cid,
                "origin": o,
                "start": o,
                "end": o + pd.Timedelta(days=60),
                "event": 0,
                "score": score,
            }
        )
        rows.append(
            {
                "cid": cid,
                "origin": o,
                "start": o + pd.Timedelta(days=60),
                "end": o + pd.Timedelta(days=end),
                "event": int(churn),
                "score": score,
            }
        )
    return pd.DataFrame(rows)


def test_landmark_static_cox_fits():
    lm = landmark(_intervals(_many(), ["score"]), 30.0)  # all at risk at 30 (min exit 90 > 30)
    cox = CoxPH().fit(lm)
    assert "score" in cox.fitter.params_.index
    assert np.allclose(lm.derive()[ENTRY].to_numpy(), 30.0)


def test_landmark_past_all_exits_raises():
    with pytest.raises(TenureValidationError, match="at risk"):
        landmark(_intervals(_handcrafted(), ["usage"]), 1000.0)


def test_landmark_requires_interval_design():
    df = tenure.load_svod_demo(with_left_truncation=False, seed=0)
    single = StudyDesign.from_event_dates(
        df,
        id_col="customer_id",
        origin_col="signup_date",
        churn_date_col="churn_date",
        active_as_of="2026-05-31",
    )
    with pytest.raises(TenureValidationError, match="interval design"):
        landmark(single, 30.0)


def test_calendar_covariate_rides_intervals():
    # A calendar-derived covariate (promo_active) that varies per interval is estimable by the
    # time-varying Cox -- the dual-clock capability (A5): the tenure clock and a calendar covariate
    # coexist on the same interval rows.
    rng = np.random.default_rng(1)
    o = pd.Timestamp("2024-01-01")
    rows = []
    for i in range(200):
        cid = f"s{i}"
        churn = rng.random() < 0.5
        rows.append(
            {
                "cid": cid,
                "origin": o,
                "start": o,
                "end": o + pd.Timedelta(days=30),
                "event": 0,
                "promo_active": int(rng.random() < 0.5),
            }
        )
        rows.append(
            {
                "cid": cid,
                "origin": o,
                "start": o + pd.Timedelta(days=30),
                "end": o + pd.Timedelta(days=60),
                "event": int(churn),
                "promo_active": int(rng.random() < 0.5),
            }
        )
    design = _intervals(pd.DataFrame(rows), ["promo_active"])
    tvc = tenure.TimeVaryingCox().fit(design)
    assert tvc.summary["covariate"].tolist() == ["promo_active"]
