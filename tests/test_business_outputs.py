from __future__ import annotations

import numpy as np
import pandas as pd

import tenure
from tenure._frame import ENTRY, EVENT, EXIT
from tenure.estimators.survival import GroupCurve

_L_MONTH = 30.4375


def _design(*, by_plan=False, **kwargs):
    df = tenure.load_svod_demo(with_left_truncation=False, seed=0)
    return tenure.StudyDesign.from_event_dates(
        df,
        id_col="customer_id",
        origin_col="signup_date",
        churn_date_col="churn_date",
        active_as_of="2026-05-31",
        analysis_start="2024-01-01",
        group_cols=["plan"] if by_plan else None,
        **kwargs,
    )


def _fit_overall():
    return tenure.KaplanMeier().fit(_design())


def _fit_by_plan():
    return tenure.KaplanMeier().fit(_design(by_plan=True), by="plan")


# --- exact step-function math -------------------------------------------------------------


def test_group_curve_integral_is_exact():
    z = np.zeros(3)
    curve = GroupCurve(
        times=np.array([0.0, 1.0, 2.0]),
        survival=np.array([1.0, 0.5, 0.25]),
        ci_lower=z,
        ci_upper=z,
        median=np.inf,
    )
    assert np.isclose(curve.integral(0.0, 3.0), 1.75)  # 1*1 + 0.5*1 + 0.25*1
    assert np.isclose(curve.integral(0.0, 1.5), 1.25)  # 1*1 + 0.5*0.5
    assert curve.integral(2.0, 2.0) == 0.0


def test_rmst_matches_hand_computed_km():  # AC-3c (analytic exactness)
    # 4 subjects: events at 1, 2, 4; censored at 3. KM: S(1)=.75, S(2)=.5, S(4)=0.
    # RMST(4) = 1*1 + 0.75*1 + 0.5*2 = 2.75
    table = pd.DataFrame(
        {ENTRY: [0.0, 0.0, 0.0, 0.0], EXIT: [1.0, 2.0, 3.0, 4.0], EVENT: [1, 1, 0, 1]}
    )
    km = tenure.KaplanMeier().fit(table)
    out = tenure.rmst(km, horizon=4.0, min_at_risk=1)
    assert np.isclose(out.iloc[0]["rmst"], 2.75, atol=1e-9)


# --- retention ----------------------------------------------------------------------------


def test_retention_at_matches_survival_at():
    km = _fit_by_plan()
    ret = tenure.retention_at(km, [30, 365])
    for _, row in ret.iterrows():
        s = km.survival_at([row["horizon"]], group=row["group"]).iloc[0]["survival"]
        assert np.isclose(row["retention"], s, atol=1e-12)
    assert ret.loc[ret["horizon"] == 30, "supported"].all()


# --- LTV: period-correct, D-S2 reduction --------------------------------------------------


def test_ltv_reduces_to_rmst_over_period_length():  # AC-15 anchor (d=0)
    km = _fit_by_plan()
    rm = tenure.rmst(km, horizon=365.0)
    lt = tenure.survival_weighted_ltv(km, period_margin=12.0, horizon=365.0, period="month")
    merged = rm.merge(lt, on="group")
    for _, row in merged.iterrows():
        assert np.isclose(row["ltv"], (12.0 / _L_MONTH) * row["rmst"], atol=1e-9)


def test_ltv_period_units_differ():  # period-correctness (AC-15)
    km = _fit_overall()
    rm = tenure.rmst(km, horizon=365.0).iloc[0]["rmst"]
    lt_month = tenure.survival_weighted_ltv(km, period_margin=12.0, horizon=365.0, period="month")
    lt_day = tenure.survival_weighted_ltv(km, period_margin=12.0, horizon=365.0, period="day")
    assert np.isclose(lt_month.iloc[0]["ltv"], (12.0 / _L_MONTH) * rm, atol=1e-9)
    assert np.isclose(lt_day.iloc[0]["ltv"], 12.0 * rm, atol=1e-9)


def test_discounting_reduces_ltv():
    km = _fit_overall()
    base = tenure.survival_weighted_ltv(km, period_margin=12.0, horizon=365.0).iloc[0]["ltv"]
    disc = tenure.survival_weighted_ltv(
        km, period_margin=12.0, horizon=365.0, discount_rate=0.02
    ).iloc[0]["ltv"]
    assert disc < base


# --- horizon support / truncate-and-relabel (TNR005, FR-BO-2) -----------------------------


def test_rmst_truncates_past_support_no_extrapolation():
    km = _fit_overall()
    out = tenure.rmst(km, horizon=5000.0).iloc[0]
    assert out["truncated"]
    assert out["effective_horizon"] < 5000.0
    again = tenure.rmst(km, horizon=out["effective_horizon"]).iloc[0]
    assert np.isclose(out["rmst"], again["rmst"], atol=1e-9)


def test_tnr005_warns_on_unsupported_horizon():
    from tenure.audit.checks.tnr005_horizon_support import evaluate_horizon_support

    km = _fit_overall()
    findings = evaluate_horizon_support(km.survival_, [365.0, 5000.0], min_at_risk=10)
    assert all(f.check_id == "TNR005" for f in findings)
    assert any(f.details["requested"] == 5000.0 for f in findings)
    # A well-supported horizon does not warn.
    assert all(f.details["requested"] != 365.0 for f in findings)


# --- SummaryReport ------------------------------------------------------------------------


def test_summary_report_table_and_metadata():
    km = _fit_by_plan()
    report = tenure.summarize(km, period_margin=12.0, ltv_horizon=5000.0)
    assert {"group", "rmst", "rmst_horizon", "ltv", "ltv_horizon"}.issubset(report.table.columns)
    assert any(c.startswith("retention@") for c in report.table.columns)
    assert report.metadata["currency"] == "USD"
    assert report.metadata["truncated_groups"]  # 5000d horizon is past support
    assert report.metadata["horizon_support_warnings"]
    assert report.metadata["audit_verdict"] == "not attached"
    md = report.to_markdown()
    assert "Retention & LTV summary" in md
    csv = report.to_csv()
    assert "# currency: USD" in csv
    assert "ltv" in csv


def test_summary_report_records_audit_provenance():  # FR-BO-5
    design = _design()
    audit_report = tenure.audit(design, strictness="block")  # clean cohort
    km = tenure.KaplanMeier().fit(design)
    report = tenure.summarize(km, period_margin=12.0, ltv_horizon=365.0, audit_report=audit_report)
    assert report.metadata["audit_verdict"] == "clean (no findings)"
