from __future__ import annotations

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

import tenure
from tenure import (
    HybridGroupCurve,
    KaplanMeier,
    ParametricSurvival,
    StudyDesign,
    TenureValidationError,
    hybrid_survival,
)
from tenure.estimators.survival import GroupCurve, SurvivalFunction


def _df(n=800, seed=0, scale_basic=200.0, scale_premium=320.0):
    rng = np.random.default_rng(seed)
    signup = pd.Timestamp("2024-01-01")
    tier = rng.choice(["basic", "premium"], size=n)
    scale = np.where(tier == "premium", scale_premium, scale_basic)
    lifetime = rng.exponential(scale)
    churn = pd.Series(signup + pd.to_timedelta(lifetime, unit="D"))
    return pd.DataFrame(
        {
            "cid": [f"c{i}" for i in range(n)],
            "start": signup,
            "churn": churn.where(churn <= pd.Timestamp("2026-05-31")),
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
        covariate_cols=kwargs.pop("covariate_cols", None) or [],
        **kwargs,
    )


@pytest.fixture(scope="module")
def fitted():
    design = _design()
    km = KaplanMeier().fit(design, by="tier")
    para = ParametricSurvival("exponential").fit(design, by="tier")
    return km, para, hybrid_survival(km, para, horizon=2000)


def test_right_continuous_at_boundary(fitted):
    # The boundary can coincide with a KM event time, where the KM itself jumps -- so the splice
    # contract is RIGHT-continuity (like the KM): the tail anchors at S_emp(boundary) exactly,
    # and approaches it from above with no splice-introduced gap.
    km, _, hyb = fitted
    for group in hyb.groups:
        c = hyb.curve(group)
        at_b = c.at(np.array([c.boundary]))[0][0]
        emp_at_b = km.survival_.curve(group).at(np.array([c.boundary]))[0][0]
        assert np.isclose(at_b, emp_at_b)  # anchor is the empirical value at the boundary
        just_after = c.at(np.array([c.boundary * (1.0 + 1e-12)]))[0][0]
        assert np.isclose(just_after, at_b, atol=1e-9)  # no new jump introduced by the splice


def test_identical_to_km_inside_data(fitted):
    km, _, hyb = fitted
    t = np.array([10.0, 30.0, 90.0, 200.0, 365.0])
    for group in hyb.groups:
        h = hyb.curve(group)
        k = km.survival_.curve(group)
        s_h, lo_h, hi_h = h.at(t)
        s_k, lo_k, hi_k = k.at(t)
        assert np.allclose(s_h, s_k)
        assert np.allclose(lo_h, lo_k)  # empirical CI preserved inside data
        assert np.allclose(hi_h, hi_k)
        assert np.isclose(h.integral(0.0, h.boundary), k.integral(0.0, h.boundary))


def test_tail_matches_analytic_exponential(fitted):
    # Exponential tail is memoryless: S_hyb(t > b) = S_emp(b) * exp(-(t - b) / lambda).
    _, para, hyb = fitted
    scale = para.params_.set_index(["group", "parameter"]).loc[("premium", "scale"), "value"]
    c = hyb.curve("premium")
    b = c.boundary
    s_emp_b = c.at(np.array([b]))[0][0]
    for t in [b + 50.0, b + 300.0, b + 900.0]:
        expected = s_emp_b * np.exp(-(t - b) / scale)
        got = c.at(np.array([t]))[0][0]
        assert np.isclose(got, expected, rtol=1e-9)


def test_rmst_matches_analytic_tail_integral(fitted):
    # RMST(H) = KM area on [0, b] + S_emp(b) * lambda * (1 - exp(-(H - b)/lambda)) beyond.
    km, para, hyb = fitted
    scale = para.params_.set_index(["group", "parameter"]).loc[("premium", "scale"), "value"]
    c = hyb.curve("premium")
    b = c.boundary
    s_emp_b = c.at(np.array([b]))[0][0]
    horizon = 1500.0
    expected = km.survival_.curve("premium").integral(0.0, b) + s_emp_b * scale * (
        1.0 - np.exp(-(horizon - b) / scale)
    )
    got = tenure.rmst(hyb, horizon=horizon).set_index("group").loc["premium", "rmst"]
    assert np.isclose(got, expected, rtol=1e-9)


def test_rmst_untruncated_and_ordered(fitted):
    km, para, hyb = fitted
    horizon = 2000.0
    km_r = tenure.rmst(km, horizon=horizon)
    hy_r = tenure.rmst(hyb, horizon=horizon)
    assert km_r["truncated"].all()
    assert not hy_r["truncated"].any()
    assert (hy_r["effective_horizon"] == horizon).all()
    # More area than the truncated KM; not identical to the pure model (empirical early segment).
    assert (hy_r["rmst"].to_numpy() > km_r["rmst"].to_numpy()).all()
    pa_r = tenure.rmst(para, horizon=horizon)
    assert not np.allclose(hy_r["rmst"].to_numpy(), pa_r["rmst"].to_numpy())


def test_business_outputs_and_plot(fitted):
    _, _, hyb = fitted
    ret = tenure.retention_at(hyb, [90, 1500])
    assert ret["supported"].all()  # parametric tail supports the far horizon
    ltv = tenure.survival_weighted_ltv(hyb, period_margin=12.0, horizon=1500.0, period="month")
    assert (ltv["ltv"] > 0).all()
    ax = tenure.plot_survival(hyb, at_risk=True)
    # 2 step curves + 2 splice vlines; the figure carries the data/model note.
    assert len(ax.lines) == 4
    texts = [t.get_text() for t in ax.figure.texts]
    assert any("model tail" in t for t in texts)


def test_ci_collapses_to_point_in_tail(fitted):
    _, _, hyb = fitted
    c = hyb.curve("basic")
    t = np.array([c.boundary + 100.0])
    s, lo, hi = c.at(t)
    assert s[0] == lo[0] == hi[0]


def test_provenance_recorded(fitted):
    km, para, hyb = fitted
    c = hyb.curve("premium")
    assert isinstance(c, HybridGroupCurve)
    assert c.boundary > 0.0
    assert c.empirical is km.survival_.curve("premium")
    assert c.modeled is para.survival_.curve("premium")
    assert c.scale > 0.0


def test_group_mismatch_raises(fitted):
    km, _, _ = fitted
    design = _design()
    overall = ParametricSurvival("weibull").fit(design)  # by=None -> ['overall']
    with pytest.raises(TenureValidationError, match="Group labels differ"):
        hybrid_survival(km, overall)


def test_step_tail_does_not_launder_extrapolation():
    # A step-curve tail (here: a second KM standing in for a Cox profile curve) flattens where
    # its data ends; the hybrid must stay truncated at a far horizon rather than integrating the
    # flat tail as if it were a projection.
    design = _design()
    km = KaplanMeier().fit(design, by="tier")
    hyb = hybrid_survival(km, km)  # model tail = the same step curves
    r = tenure.rmst(hyb, horizon=3000.0)
    assert r["truncated"].all()
    assert (r["effective_horizon"] < 3000.0).all()


def test_median_in_tail_matches_analytic():
    # Short window + long lifetimes: S_emp(boundary) > 0.5, so the median sits in the model tail.
    # For an exponential tail: median = b + lambda * ln(2 * S_emp(b)).
    df = _df(seed=3, scale_basic=2000.0, scale_premium=2500.0)
    design = _design(df)
    km = KaplanMeier().fit(design, by="tier")
    para = ParametricSurvival("exponential").fit(design, by="tier")
    hyb = hybrid_survival(km, para)

    scale = para.params_.set_index(["group", "parameter"]).loc[("basic", "scale"), "value"]
    c = hyb.curve("basic")
    s_emp_b = c.at(np.array([c.boundary]))[0][0]
    assert s_emp_b > 0.5  # median genuinely beyond the data
    expected = c.boundary + scale * np.log(2.0 * s_emp_b)
    assert np.isclose(c.median, expected, rtol=1e-6)


def test_zero_model_survival_at_boundary_raises():
    times = np.array([0.0, 10.0])
    emp = GroupCurve(
        times=times,
        survival=np.array([1.0, 0.8]),
        ci_lower=np.array([1.0, 0.7]),
        ci_upper=np.array([1.0, 0.9]),
        median=float("inf"),
        risk_times=times,
        n_at_risk=np.array([100.0, 50.0]),
        last_event_time=10.0,
    )
    dead = GroupCurve(
        times=times,
        survival=np.array([1.0, 0.0]),  # model dead before the boundary
        ci_lower=np.array([1.0, 0.0]),
        ci_upper=np.array([1.0, 0.0]),
        median=5.0,
        risk_times=times,
        n_at_risk=np.array([100.0, 50.0]),
        last_event_time=10.0,
    )
    with pytest.raises(TenureValidationError, match="0 at the splice boundary"):
        hybrid_survival(SurvivalFunction({"g": emp}), SurvivalFunction({"g": dead}), min_at_risk=10)
