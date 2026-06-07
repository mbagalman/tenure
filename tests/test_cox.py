from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from lifelines import CoxPHFitter

import tenure
from tenure import CoxPH, StudyDesign, TenureValidationError
from tenure._frame import ENTRY, EVENT, EXIT
from tenure.audit.report import Status


def _cox_df(n=800, seed=0):
    """Premium tier lives ~1.6x longer; age is a noise numeric covariate."""
    rng = np.random.default_rng(seed)
    signup = pd.Timestamp("2024-01-01")
    tier = rng.choice(["basic", "premium"], size=n)
    age = rng.integers(18, 70, size=n).astype(float)
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
            "age": age,
        }
    )


def _design(df=None, *, covariate_cols=("tier", "age"), **kwargs):
    return StudyDesign.from_event_dates(
        _cox_df() if df is None else df,
        id_col="cid",
        origin_col="start",
        churn_date_col="churn",
        active_as_of="2026-05-31",
        covariate_cols=list(covariate_cols),
        **kwargs,
    )


def test_categorical_one_hot_mapping():
    design = _design()
    assert design.covariate_mappings["tier"] == {
        "kind": "categorical",
        "levels": ["basic", "premium"],
        "baseline": "basic",
    }
    assert design.covariate_mappings["age"] == {"kind": "numeric"}
    encoded = design.encode_covariates(design.derive())
    assert "tier_premium" in encoded.columns
    assert "tier_basic" not in encoded.columns  # baseline dropped
    assert "age" in encoded.columns


def test_cox_coefficients_match_lifelines():  # AC: reference-match
    design = _design()
    cox = CoxPH().fit(design)

    table = design.derive()
    ref = design.encode_covariates(table)
    ref["d"] = table[EXIT].to_numpy(float)
    ref["e"] = table[EVENT].to_numpy(int)
    ref["t"] = table[ENTRY].to_numpy(float)
    reference = CoxPHFitter().fit(ref, duration_col="d", event_col="e", entry_col="t")

    assert np.allclose(
        cox.fitter.params_.sort_index().to_numpy(),
        reference.params_.sort_index().to_numpy(),
        atol=1e-9,
    )


def test_predict_survival_premium_outlives_basic():
    cox = CoxPH().fit(_design())
    sf = cox.predict_survival(cox.profile_grid("tier"))
    assert set(sf.groups) == {"basic", "premium"}
    retention = tenure.retention_at(sf, [365], min_at_risk=1).set_index("group")["retention"]
    assert retention["premium"] > retention["basic"]
    # Survival is a valid, non-increasing probability.
    curve = sf.survival_at([0, 90, 365], group="premium").sort_values("time")["survival"]
    assert ((curve >= 0) & (curve <= 1)).all()
    assert curve.is_monotonic_decreasing


def test_predict_survival_raw_label_profile_and_index_label():
    cox = CoxPH().fit(_design())
    profiles = pd.DataFrame({"tier": ["premium"], "age": [40.0]}, index=["VIP profile"])
    sf = cox.predict_survival(profiles)  # raw labels, custom index
    assert sf.groups == ["VIP profile"]


def test_business_outputs_consume_cox_curves():  # A8 / tiers-agree
    cox = CoxPH().fit(_design())
    sf = cox.predict_survival(cox.profile_grid("tier"))
    retention = tenure.retention_at(sf, [180])
    for _, row in retention.iterrows():
        direct = sf.survival_at([180], group=row["group"]).iloc[0]["survival"]
        assert np.isclose(row["retention"], direct)
    assert (tenure.survival_weighted_ltv(sf, period_margin=12.0, horizon=365.0)["ltv"] > 0).all()


def test_null_covariate_raises():
    df = _cox_df()
    df.loc[0, "age"] = np.nan
    with pytest.raises(TenureValidationError, match="null"):
        _design(df)


def test_cox_requires_covariate_cols():
    design = StudyDesign.from_event_dates(
        _cox_df(),
        id_col="cid",
        origin_col="start",
        churn_date_col="churn",
        active_as_of="2026-05-31",
    )
    with pytest.raises(TenureValidationError):
        CoxPH().fit(design)


def test_cox_not_fitted_raises():
    with pytest.raises(RuntimeError):
        CoxPH().predict_survival(pd.DataFrame({"tier": ["basic"], "age": [40.0]}))


def test_tnr004_scans_covariate_cols():
    n = 400
    rng = np.random.default_rng(1)
    signup = pd.Timestamp("2024-01-01")
    tenure_days = rng.integers(1, 400, size=n)
    churn = signup + pd.to_timedelta(tenure_days, unit="D")
    df = pd.DataFrame(
        {
            "cid": [f"c{i}" for i in range(n)],
            "start": signup,
            "churn": churn,
            "ever_upgraded": (tenure_days > 200).astype(int),
        }
    )
    design = StudyDesign.from_event_dates(
        df,
        id_col="cid",
        origin_col="start",
        churn_date_col="churn",
        active_as_of="2025-06-01",
        covariate_cols=["ever_upgraded"],
    )
    report = tenure.audit(design, strictness="block")
    finding = next(r for r in report.results if r.check_id == "TNR004")
    assert finding.status is Status.WARN
    assert "ever_upgraded" in finding.details["covariates"]
