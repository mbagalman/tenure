from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
from lifelines import CoxPHFitter

import tenure
from tenure import CoxPH, StudyDesign, TenureValidationError
from tenure._frame import ENTRY, EVENT, EXIT


def _crossing_df(n=2000, seed=0):
    """Crossing hazards by plan (Weibull shapes 0.6 vs 2.2) => PH genuinely fails for plan.

    Age scales the Weibull scale by exp(-beta*age/shape), which makes age's hazard effect
    exactly proportional WITHIN each stratum -- so once plan is stratified away, PH holds.
    """
    rng = np.random.default_rng(seed)
    signup = pd.Timestamp("2024-01-01")
    plan = rng.choice(["basic", "premium"], size=n)
    age = rng.integers(20, 70, size=n).astype(float)
    shape = np.where(plan == "premium", 2.2, 0.6)
    beta = 0.02
    scale = 300.0 * np.exp(-beta * (age - 45.0) / shape)
    lifetime = scale * rng.weibull(shape)
    churn = pd.Series(signup + pd.to_timedelta(lifetime, unit="D"))
    return pd.DataFrame(
        {
            "cid": [f"c{i}" for i in range(n)],
            "start": signup,
            "churn": churn.where(churn <= pd.Timestamp("2026-05-31")),
            "plan": plan,
            "age": age,
        }
    )


def _design(df=None, **kwargs):
    return StudyDesign.from_event_dates(
        _crossing_df() if df is None else df,
        id_col="cid",
        origin_col="start",
        churn_date_col="churn",
        active_as_of="2026-05-31",
        covariate_cols=["plan", "age"],
        **kwargs,
    )


def test_coefficients_match_lifelines_stratified():  # AC: reference-match
    design = _design()
    cox = CoxPH(strata=["plan"]).fit(design)

    table = design.derive()
    ref = design.encode_covariates(table)
    ref = ref.drop(columns=["plan_premium"])
    ref["plan"] = table["plan"].astype(str).to_numpy()
    ref["d"] = table[EXIT].to_numpy(float)
    ref["e"] = table[EVENT].to_numpy(int)
    ref["t"] = table[ENTRY].to_numpy(float)
    reference = CoxPHFitter().fit(
        ref, duration_col="d", event_col="e", entry_col="t", strata=["plan"]
    )

    assert list(cox.fitter.params_.index) == ["age"]
    assert np.allclose(cox.fitter.params_.to_numpy(), reference.params_.to_numpy(), atol=1e-9)


def test_ph_remedy_loop():  # the detect -> stratify -> resolved narrative, end to end
    # Seed pinned like the demo regression gates: the structural claims (plan flagged, then
    # stratified out of the test) are deterministic, but "age passes within strata" asserts a
    # statistical test's tail -- the rank-transform Schoenfeld test mildly over-rejects under
    # strong effects, so some seeds land p < 0.05 by size distortion. Seed 1 is decisive both
    # ways (plan p ~ 1e-154 unstratified; age p = 0.56 stratified).
    design = _design(_crossing_df(seed=1))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        diag = CoxPH().fit(design).proportional_hazards_test()
    assert "plan_premium" in diag.violations
    # The warning suggests the RAW covariate name, usable directly as strata=.
    messages = [str(w.message) for w in caught]
    assert any("strata=['plan']" in m for m in messages)

    strat = CoxPH(strata=["plan"]).fit(design)
    diag2 = strat.proportional_hazards_test(warn=False)
    tested = diag2.table["covariate"].tolist()
    assert "plan_premium" not in tested  # stratified away: no coefficient, no PH assumption
    assert tested == ["age"]
    assert diag2.ok  # within-stratum age effect is proportional by construction


def test_per_stratum_baselines_differ():
    design = _design()
    strat = CoxPH(strata=["plan"]).fit(design)
    curves = strat.predict_survival(strat.profile_grid("plan"))
    ret = tenure.retention_at(curves, [60, 400]).set_index(["group", "horizon"])
    basic_60 = ret.loc[("basic", 60.0), "retention"]
    premium_60 = ret.loc[("premium", 60.0), "retention"]
    # Shape 0.6 (basic) churns early; shape 2.2 (premium) churns late: curves differ early on.
    assert premium_60 > basic_60 + 0.05


def test_business_outputs_consume_stratified_curves():  # A3/A8 interchangeability
    design = _design()
    strat = CoxPH(strata=["plan"]).fit(design)
    curves = strat.predict_survival(strat.profile_grid("plan"))
    assert not tenure.rmst(curves, horizon=365).empty
    ltv = tenure.survival_weighted_ltv(curves, period_margin=12.0, horizon=365.0, period="month")
    assert (ltv["ltv"] > 0).all()


def test_scoring_works_stratified():
    design = _design()
    strat = CoxPH(strata=["plan"]).fit(design)
    scores = tenure.churn_risk_scores(strat, horizon=365.0)
    t = scores.table
    assert np.isfinite(t["risk_score"]).all()
    assert t["survival_at_horizon"].between(0.0, 1.0).all()
    assert len(t) == len(design.derive())


def test_encode_for_prediction_carries_strata():
    design = _design()
    strat = CoxPH(strata=["plan"]).fit(design)
    encoded = strat.encode_for_prediction(design)
    assert "plan" in encoded.columns  # raw labels for baseline selection
    assert "age" in encoded.columns
    assert "plan_premium" not in encoded.columns


def test_profiles_missing_strata_column_raises():
    design = _design()
    strat = CoxPH(strata=["plan"]).fit(design)
    with pytest.raises(TenureValidationError, match="strata column"):
        strat.predict_survival({"age": 40.0})


def test_unknown_strata_level_raises():
    design = _design()
    strat = CoxPH(strata=["plan"]).fit(design)
    with pytest.raises(TenureValidationError, match="not seen at fit time"):
        strat.predict_survival({"age": 40.0, "plan": "platinum"})


def test_strata_not_a_covariate_raises():
    design = _design()
    with pytest.raises(TenureValidationError, match="not a covariate_col"):
        CoxPH(strata=["channel"]).fit(design)


def test_numeric_strata_raises():
    design = _design()
    with pytest.raises(TenureValidationError, match="numeric"):
        CoxPH(strata=["age"]).fit(design)


def test_all_covariates_stratified_raises():
    df = _crossing_df()
    design = StudyDesign.from_event_dates(
        df,
        id_col="cid",
        origin_col="start",
        churn_date_col="churn",
        active_as_of="2026-05-31",
        covariate_cols=["plan"],
    )
    with pytest.raises(TenureValidationError, match="at least one must remain"):
        CoxPH(strata=["plan"]).fit(design)


def test_strata_accepts_bare_string():
    design = _design()
    strat = CoxPH(strata="plan").fit(design)
    assert strat.strata == ["plan"]
    assert list(strat.fitter.params_.index) == ["age"]


def test_delayed_entry_stratified_matches_lifelines():
    df = tenure.load_svod_demo(with_left_truncation=True)
    design = StudyDesign.from_event_dates(
        df,
        id_col="customer_id",
        origin_col="signup_date",
        churn_date_col="churn_date",
        active_as_of="2026-05-31",
        analysis_start="2024-01-01",
        event_observed_from="2024-01-01",
        covariate_cols=["plan", "channel"],
    )
    tenure.audit(design)
    table = design.derive()
    assert table[ENTRY].to_numpy(float).max() > 0.0  # real delayed entry

    cox = CoxPH(strata=["channel"]).fit(design)

    ref = design.encode_covariates(table)
    ref = ref.drop(columns=[c for c in ref.columns if c.startswith("channel_")])
    ref["channel"] = table["channel"].astype(str).to_numpy()
    ref["d"] = table[EXIT].to_numpy(float)
    ref["e"] = table[EVENT].to_numpy(int)
    ref["t"] = table[ENTRY].to_numpy(float)
    reference = CoxPHFitter().fit(
        ref, duration_col="d", event_col="e", entry_col="t", strata=["channel"]
    )
    assert np.allclose(
        cox.fitter.params_.sort_index().to_numpy(),
        reference.params_.sort_index().to_numpy(),
        atol=1e-9,
    )
