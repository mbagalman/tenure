from __future__ import annotations

import numpy as np
import pandas as pd

import tenure
from tenure import StudyDesign
from tenure.audit.report import Status


def _immortal_df(n=400, seed=0):
    """A covariate that is 1 only for customers who survived past tenure 200 days."""
    rng = np.random.default_rng(seed)
    signup = pd.Timestamp("2024-01-01")
    tenure_days = rng.integers(1, 400, size=n)
    churn = signup + pd.to_timedelta(tenure_days, unit="D")
    return pd.DataFrame(
        {
            "cid": [f"c{i}" for i in range(n)],
            "start": signup,
            "churn": churn,
            "ever_upgraded": (tenure_days > 200).astype(int),
        }
    )


def _design(df, **kwargs):
    return StudyDesign.from_event_dates(
        df,
        id_col="cid",
        origin_col="start",
        churn_date_col="churn",
        active_as_of="2025-06-01",
        group_cols=["ever_upgraded"],
        **kwargs,
    )


def _tnr004(report):
    return next(r for r in report.results if r.check_id == "TNR004")


def test_warns_on_immortal_signature():
    report = tenure.audit(_design(_immortal_df()), strictness="block")  # WARN, no raise
    result = _tnr004(report)
    assert result.status is Status.WARN
    assert "ever_upgraded" in result.details["covariates"]


def test_attestation_clears_the_warning():
    design = _design(_immortal_df(), attest_invariant_covariates=["ever_upgraded"])
    assert _tnr004(tenure.audit(design, strictness="block")).status is Status.PASS


def test_pass_on_origin_time_covariate():
    # plan is assigned at signup (time-invariant) -> no immortal-time signature.
    df = tenure.load_svod_demo(with_left_truncation=False, seed=0)
    design = tenure.StudyDesign.from_event_dates(
        df,
        id_col="customer_id",
        origin_col="signup_date",
        churn_date_col="churn_date",
        active_as_of="2026-05-31",
        group_cols=["plan"],
    )
    assert _tnr004(tenure.audit(design, strictness="block")).status is Status.PASS


def test_not_applicable_without_group_cols():
    df = _immortal_df()
    design = StudyDesign.from_event_dates(
        df, id_col="cid", origin_col="start", churn_date_col="churn", active_as_of="2025-06-01"
    )
    report = tenure.audit(design, strictness="block")
    assert all(r.check_id != "TNR004" for r in report.results)
