from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import tenure
from tenure._frame import as_estimator_frame, to_tenure, unit_factor


def test_unit_factor():
    assert unit_factor("day") == 1.0
    assert unit_factor("week") == 7.0
    assert abs(unit_factor("month") - 30.4375) < 1e-9


def test_unit_factor_rejects_unknown():
    with pytest.raises(ValueError):
        unit_factor("fortnight")


def test_to_tenure_days_array():
    origin = pd.Series(pd.to_datetime(["2025-01-01", "2025-01-01"]))
    end = pd.to_datetime(["2025-01-11", "2025-02-01"])
    out = to_tenure(end, origin, "day")
    assert abs(out[0] - 10.0) < 1e-9
    assert abs(out[1] - 31.0) < 1e-9


def test_to_tenure_scalar_broadcast_and_month():
    origin = pd.Series(pd.to_datetime(["2025-01-01", "2025-01-15"]))
    out = to_tenure("2025-02-01", origin, "month")
    assert abs(out[0] - 31.0 / 30.4375) < 1e-9
    assert abs(out[1] - 17.0 / 30.4375) < 1e-9


def test_as_estimator_frame_shapes():
    df = tenure.load_svod_demo(with_left_truncation=False, seed=0)
    sd = tenure.StudyDesign.from_event_dates(
        df,
        id_col="customer_id",
        origin_col="signup_date",
        churn_date_col="churn_date",
        active_as_of="2026-05-31",
    )
    ef = as_estimator_frame(sd.derive())
    n = len(df)
    assert ef.entry.shape == (n,)
    assert ef.duration.shape == (n,)
    assert ef.event.shape == (n,)
    assert np.all(ef.duration >= 0)
    assert set(np.unique(ef.event)).issubset({0, 1})
