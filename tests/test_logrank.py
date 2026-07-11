from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from lifelines.statistics import multivariate_logrank_test

import tenure
from tenure import StudyDesign, TenureValidationError
from tenure._frame import ENTRY, EVENT, EXIT


def _df(n=600, seed=0, hazard_ratio=1.6):
    """Two tiers; premium lives ~hazard_ratio longer. hazard_ratio=1.0 => same distribution."""
    rng = np.random.default_rng(seed)
    signup = pd.Timestamp("2024-01-01")
    tier = rng.choice(["basic", "premium"], size=n)
    scale = np.where(tier == "premium", 200.0 * hazard_ratio, 200.0)
    lifetime = rng.exponential(scale)
    churn = pd.Series(signup + pd.to_timedelta(lifetime, unit="D"))
    churn_date = churn.where(churn <= pd.Timestamp("2026-05-31"))
    return pd.DataFrame(
        {
            "cid": [f"c{i}" for i in range(n)],
            "start": signup,
            "churn": churn_date,
            "tier": tier,
        }
    )


def _design(df=None, **kwargs):
    return StudyDesign.from_event_dates(
        _df() if df is None else df,
        id_col="cid",
        origin_col="start",
        churn_date_col="churn",
        active_as_of="2026-05-31",
        group_cols=["tier"],
        **kwargs,
    )


def test_matches_lifelines_no_delayed_entry():  # reference-match (the correctness gate)
    design = _design()
    table = design.derive()
    report = tenure.logrank_test(design, by="tier")

    ref = multivariate_logrank_test(
        table[EXIT].to_numpy(float),
        table["tier"].to_numpy(),
        table[EVENT].to_numpy(int),
    )
    assert np.isclose(report.test_statistic, ref.test_statistic, atol=1e-9)
    assert np.isclose(report.p_value, ref.p_value, atol=1e-12)
    assert report.degrees_of_freedom == 1


def test_multi_group_matches_lifelines():  # 3-group chi-square, df=2
    rng = np.random.default_rng(3)
    signup = pd.Timestamp("2024-01-01")
    plan = rng.choice(["basic", "standard", "premium"], size=900)
    scale = np.select(
        [plan == "basic", plan == "standard", plan == "premium"], [180.0, 240.0, 320.0]
    )
    lifetime = rng.exponential(scale)
    churn = pd.Series(signup + pd.to_timedelta(lifetime, unit="D"))
    df = pd.DataFrame(
        {
            "cid": [f"c{i}" for i in range(900)],
            "start": signup,
            "churn": churn.where(churn <= pd.Timestamp("2026-05-31")),
            "plan": plan,
        }
    )
    design = StudyDesign.from_event_dates(
        df,
        id_col="cid",
        origin_col="start",
        churn_date_col="churn",
        active_as_of="2026-05-31",
        group_cols=["plan"],
    )
    table = design.derive()
    report = tenure.logrank_test(design, by="plan")
    ref = multivariate_logrank_test(
        table[EXIT].to_numpy(float), table["plan"].to_numpy(), table[EVENT].to_numpy(int)
    )
    assert report.degrees_of_freedom == 2
    assert np.isclose(report.test_statistic, ref.test_statistic, atol=1e-9)
    assert np.isclose(report.p_value, ref.p_value, atol=1e-12)


def test_identical_groups_not_significant():
    design = _design(_df(hazard_ratio=1.0, seed=7))
    report = tenure.logrank_test(design, by="tier")
    assert not report.significant()
    assert report.p_value > 0.05


def test_separated_groups_significant():
    design = _design(_df(hazard_ratio=2.0, seed=1))
    report = tenure.logrank_test(design, by="tier")
    assert report.significant()
    assert report.p_value < 0.01


def test_delayed_entry_changes_statistic_and_is_not_lifelines_naive():
    # A window-cut design has real delayed entry; the entry-aware statistic must differ from the
    # entry-ignoring lifelines call on the same durations (proving entry is actually honored).
    df = tenure.load_svod_demo(with_left_truncation=True)
    design = StudyDesign.from_event_dates(
        df,
        id_col="customer_id",
        origin_col="signup_date",
        churn_date_col="churn_date",
        active_as_of="2026-05-31",
        analysis_start="2024-01-01",
        event_observed_from="2024-01-01",
        group_cols=["plan"],
    )
    tenure.audit(design)
    table = design.derive()
    assert table[ENTRY].to_numpy(float).max() > 0.0  # delayed entry is present

    report = tenure.logrank_test(design, by="plan")
    naive = multivariate_logrank_test(
        table[EXIT].to_numpy(float), table["plan"].to_numpy(), table[EVENT].to_numpy(int)
    )
    # Entry-aware != entry-ignoring: the whole point of handling left truncation.
    assert not np.isclose(report.test_statistic, naive.test_statistic)


def test_no_delayed_entry_equals_zero_entry_path():
    # With entry all zero, the entry-aware path must reduce to the standard result.
    design = _design()
    table = design.derive()
    assert np.allclose(table[ENTRY].to_numpy(float), 0.0)
    report = tenure.logrank_test(design, by="tier")
    ref = multivariate_logrank_test(
        table[EXIT].to_numpy(float), table["tier"].to_numpy(), table[EVENT].to_numpy(int)
    )
    assert np.isclose(report.test_statistic, ref.test_statistic, atol=1e-9)


def test_report_table_contract():
    design = _design()
    report = tenure.logrank_test(design, by="tier")
    assert list(report.table.columns) == ["group", "n", "observed", "expected"]
    assert set(report.table["group"]) == {"basic", "premium"}
    # Observed and expected totals reconcile (each death is one observed and one expected event).
    assert np.isclose(report.table["observed"].sum(), report.table["expected"].sum())
    assert report.table["n"].sum() == len(design.derive())


def test_single_group_raises():
    df = _df()
    df["tier"] = "basic"  # collapse to one level
    design = _design(df)
    with pytest.raises(TenureValidationError, match="at least two groups"):
        tenure.logrank_test(design, by="tier")


def test_by_none_raises():
    design = _design()
    with pytest.raises(TenureValidationError, match="at least two groups"):
        tenure.logrank_test(design, by=None)


def test_summary_string():
    design = _design(_df(hazard_ratio=2.0, seed=1))
    report = tenure.logrank_test(design, by="tier")
    assert "log-rank" in report.summary
    assert "differ" in report.summary
