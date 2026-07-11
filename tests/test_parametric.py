from __future__ import annotations

import matplotlib
import numpy as np
import pandas as pd
import pytest
from lifelines import (
    ExponentialFitter,
    LogLogisticFitter,
    LogNormalFitter,
    WeibullFitter,
)

matplotlib.use("Agg")

import tenure
from tenure import ParametricSurvival, StudyDesign, TenureValidationError
from tenure._frame import as_estimator_frame


def _df(n=800, seed=0):
    rng = np.random.default_rng(seed)
    signup = pd.Timestamp("2024-01-01")
    tier = rng.choice(["basic", "premium"], size=n)
    scale = np.where(tier == "premium", 320.0, 200.0)
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


_LIFELINES = {
    "weibull": WeibullFitter,
    "exponential": ExponentialFitter,
    "lognormal": LogNormalFitter,
    "loglogistic": LogLogisticFitter,
}


@pytest.mark.parametrize("distribution", ["weibull", "exponential", "lognormal", "loglogistic"])
def test_survival_matches_lifelines(distribution):  # reference-match (the correctness gate)
    design = _design()
    model = ParametricSurvival(distribution).fit(design, by="tier")

    table = design.derive()
    ef = as_estimator_frame(table.loc[(table["tier"] == "premium").to_numpy()])
    ref_fitter = _LIFELINES[distribution]().fit(ef.duration, ef.event, entry=ef.entry)

    t = np.array([10.0, 30.0, 90.0, 200.0, 365.0, 900.0])  # includes beyond-support extrapolation
    ref = ref_fitter.survival_function_at_times(t).to_numpy()
    ours = model.survival_.survival_at(t, group="premium")["survival"].to_numpy()
    assert np.allclose(ref, ours, atol=1e-9)


def test_params_match_lifelines_weibull():
    design = _design()
    model = ParametricSurvival("weibull").fit(design, by="tier")
    table = design.derive()
    ef = as_estimator_frame(table.loc[(table["tier"] == "premium").to_numpy()])
    ref = WeibullFitter().fit(ef.duration, ef.event, entry=ef.entry)

    params = model.params_
    prem = params[params["group"] == "premium"].set_index("parameter")["value"]
    assert np.isclose(prem["scale"], ref.lambda_, atol=1e-6)
    assert np.isclose(prem["shape"], ref.rho_, atol=1e-6)


def test_extrapolates_past_km_support():
    # The headline capability: RMST/retention beyond the last event time is a principled model
    # projection (not truncated), where Kaplan-Meier must truncate-and-relabel.
    design = _design()
    km = tenure.KaplanMeier().fit(design, by="tier")
    para = ParametricSurvival("weibull").fit(design, by="tier")

    horizon = 3000.0  # far beyond the ~2.4y observation window
    km_rmst = tenure.rmst(km, horizon=horizon)
    para_rmst = tenure.rmst(para, horizon=horizon)

    assert km_rmst["truncated"].all()  # KM cannot reach the horizon
    assert not para_rmst["truncated"].any()  # the model extrapolates to it
    assert (para_rmst["effective_horizon"] == horizon).all()
    # And the extrapolated RMST exceeds the truncated KM RMST (more area captured).
    assert (para_rmst["rmst"].to_numpy() > km_rmst["rmst"].to_numpy()).all()


def test_retention_supported_flag_true_when_extrapolating():
    design = _design()
    para = ParametricSurvival("weibull").fit(design, by="tier")
    ret = tenure.retention_at(para, [90, 3000])
    assert ret["supported"].all()  # a parametric model supports any horizon


def test_rmst_matches_analytic_exponential():
    # Independent oracle: for an exponential fit, RMST(H) = scale * (1 - exp(-H/scale)).
    design = _design()
    model = ParametricSurvival("exponential").fit(design, by="tier")
    scale = model.params_.set_index(["group", "parameter"]).loc[("premium", "scale"), "value"]
    horizon = 500.0
    analytic = scale * (1.0 - np.exp(-horizon / scale))
    got = tenure.rmst(model, horizon=horizon)
    prem = got[got["group"] == "premium"]["rmst"].iloc[0]
    assert np.isclose(prem, analytic, rtol=1e-6)


def test_business_outputs_and_plot_consume_parametric():  # A3/A8 interchangeability
    design = _design()
    model = ParametricSurvival("weibull").fit(design, by="tier")
    _ = tenure.retention_at(model, [30, 365])
    _ = tenure.rmst(model, horizon=365)
    _ = tenure.survival_weighted_ltv(model, period_margin=12.0, horizon=365.0, period="month")
    summary = tenure.summarize(model, period_margin=12.0, ltv_horizon=365.0)
    assert not summary.table.empty
    ax = tenure.plot_survival(model, at_risk=True)
    assert ax is not None


def test_median_survival():
    design = _design()
    model = ParametricSurvival("weibull").fit(design, by="tier")
    med = model.median_survival()
    assert set(med["group"]) == {"basic", "premium"}
    # Premium (longer scale) has the larger median.
    m = med.set_index("group")["median"]
    assert m["premium"] > m["basic"]


def test_delayed_entry_reference_match():
    # Window-cut design with real delayed entry must still match lifelines fit with entry.
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
    assert table["entry_tenure"].to_numpy(float).max() > 0.0

    model = ParametricSurvival("weibull").fit(design, by="plan")
    ef = as_estimator_frame(table.loc[(table["plan"] == "premium").to_numpy()])
    ref = WeibullFitter().fit(ef.duration, ef.event, entry=ef.entry)
    t = np.array([100.0, 365.0, 1000.0])
    assert np.allclose(
        ref.survival_function_at_times(t).to_numpy(),
        model.survival_.survival_at(t, group="premium")["survival"].to_numpy(),
        atol=1e-9,
    )


def test_overall_single_curve():
    design = _design()
    model = ParametricSurvival("weibull").fit(design)  # by=None
    assert model.survival_.groups == ["overall"]


def test_unknown_distribution_raises():
    with pytest.raises(TenureValidationError, match="Unknown distribution"):
        ParametricSurvival("gompertz")


def test_not_fitted_raises():
    with pytest.raises(RuntimeError, match="not fitted"):
        _ = ParametricSurvival("weibull").survival_
