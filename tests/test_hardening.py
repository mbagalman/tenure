"""Correctness-hardening regression tests (v0.3.1) -- the edge cases a second review surfaced.

Covers: strict churn-date parsing, all-censored RMST/LTV, the unmapped-status fit block, and
unknown categorical levels in prediction. (Time-varying Cox centering is anchored to an independent
CoxPHFitter oracle in test_time_varying_cox.py.)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import tenure
from tenure import StudyDesign, TenureValidationError

# --- Finding 1: malformed churn dates must not be silently treated as active -------------------


def test_present_but_unparseable_churn_date_raises():
    df = pd.DataFrame(
        {
            "cid": [1, 2, 3],
            "signup": pd.to_datetime(["2024-01-01"] * 3),
            "churn": ["2024-06-01", "not-a-date", "2024-07-01"],
        }
    )
    with pytest.raises(TenureValidationError, match="not parseable"):
        StudyDesign.from_event_dates(
            df,
            id_col="cid",
            origin_col="signup",
            churn_date_col="churn",
            active_as_of="2025-01-01",
        )


def test_null_and_blank_churn_dates_stay_active():
    df = pd.DataFrame(
        {
            "cid": [1, 2, 3],
            "signup": pd.to_datetime(["2024-01-01"] * 3),
            "churn": [
                pd.NaT,
                "",
                "2024-06-01",
            ],  # genuinely-missing -> active; present date -> churn
        }
    )
    design = StudyDesign.from_event_dates(
        df, id_col="cid", origin_col="signup", churn_date_col="churn", active_as_of="2025-01-01"
    )
    table = design.derive()
    event = dict(zip(table["id"], table["event"], strict=True))
    assert event == {1: 0, 2: 0, 3: 1}


# --- Finding 2: an all-censored cohort must not collapse RMST/LTV to zero -----------------------


def _all_censored_km(n: int = 60):
    df = pd.DataFrame(
        {
            "cid": range(n),
            "signup": pd.to_datetime(["2024-01-01"] * n),
            "churn": [pd.NaT] * n,  # nobody churns; all censored at active_as_of (tenure ~365d)
        }
    )
    design = StudyDesign.from_event_dates(
        df, id_col="cid", origin_col="signup", churn_date_col="churn", active_as_of="2025-01-01"
    )
    return tenure.KaplanMeier().fit(design)


def test_all_censored_rmst_is_supported_not_zero():
    km = _all_censored_km()
    row = tenure.rmst(km, horizon=500).iloc[0]  # horizon beyond the ~365d of follow-up
    # Survival is 1 throughout, so RMST equals the supported horizon -- NOT zero.
    assert row["rmst"] > 0.0
    assert np.isclose(row["rmst"], row["effective_horizon"])
    assert row["effective_horizon"] >= 360.0  # ~365d last observation, not collapsed to 0
    assert bool(row["truncated"]) is True


def test_all_censored_ltv_is_positive():
    km = _all_censored_km()
    ltv = tenure.survival_weighted_ltv(km, period_margin=12.0, horizon=365.0).iloc[0]["ltv"]
    assert float(ltv) > 0.0


# --- Finding 4: unmapped statuses are dropped, so fitting must require an audit first -----------

_UNMAPPED_DF = pd.DataFrame(
    {
        "cid": [1, 2, 3, 4, 5, 6],
        "signup": pd.to_datetime(["2024-01-01"] * 6),
        "exit": pd.to_datetime(["2024-06-01", "2024-07-01", "2024-08-01"] * 2),
        "status": [
            "churn",
            "active",
            "churn",
            "active",
            "churn",
            "upgraded",
        ],  # 'upgraded' unmapped
    }
)
_STATUS_MAP = {"churn": "event", "active": "censored"}


def _unmapped_design():
    return StudyDesign.from_status(
        _UNMAPPED_DF,
        id_col="cid",
        origin_col="signup",
        exit_col="exit",
        status_col="status",
        status_map=_STATUS_MAP,
        active_as_of="2025-01-01",
    )


def test_unmapped_status_blocks_low_level_fit():
    design = _unmapped_design()
    assert design.n_unmapped == 1
    with pytest.raises(TenureValidationError, match="status_map"):
        tenure.KaplanMeier().fit(design)


def test_audit_acknowledges_unmapped_then_fit_allowed():
    design = _unmapped_design()
    tenure.audit(design, strictness="warn")  # block downgraded to warn; marks the design audited
    km = tenure.KaplanMeier().fit(design)  # no longer blocked
    assert km.survival_ is not None


def test_guided_workflow_handles_unmapped_in_warn_mode():
    study = tenure.RetentionStudy.from_status(
        _UNMAPPED_DF,
        strictness="warn",
        id_col="cid",
        origin_col="signup",
        exit_col="exit",
        status_col="status",
        status_map=_STATUS_MAP,
        active_as_of="2025-01-01",
    )
    with pytest.warns(UserWarning, match="TNR003"):
        result = study.run()  # audits (warn) then fits the SAME design object -> not blocked
    assert result.curves is not None


# --- Finding 5: unknown categorical levels must not silently encode as baseline -----------------


def _categorical_cox():
    n = 60
    df = pd.DataFrame(
        {
            "cid": range(n),
            "signup": pd.to_datetime(["2024-01-01"] * n),
            "churn": [
                pd.Timestamp("2024-03-01") + pd.Timedelta(days=i * 3) if i % 2 == 0 else pd.NaT
                for i in range(n)
            ],
            "plan": ["basic" if i % 3 else "premium" for i in range(n)],
        }
    )
    study = StudyDesign.from_event_dates(
        df,
        id_col="cid",
        origin_col="signup",
        churn_date_col="churn",
        active_as_of="2025-06-01",
        covariate_cols=["plan"],
    )
    return tenure.CoxPH().fit(study)


def test_unknown_categorical_level_raises_in_prediction():
    cox = _categorical_cox()
    with pytest.raises(TenureValidationError, match="not seen at fit time"):
        cox.predict_survival(pd.DataFrame({"plan": ["enterprise"]}))


def test_known_categorical_level_predicts_fine():
    cox = _categorical_cox()
    sf = cox.predict_survival(pd.DataFrame({"plan": ["premium"]}))
    assert sf.groups  # a curve was produced for the known level
