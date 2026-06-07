from __future__ import annotations

import numpy as np
import pandas as pd

import tenure
from tenure import CoxPH, StudyDesign


def _cox_df(n=600, seed=0):
    rng = np.random.default_rng(seed)
    signup = pd.Timestamp("2024-01-01")
    tier = rng.choice(["basic", "premium"], size=n)
    age = rng.integers(18, 70, size=n).astype(float)
    lifetime = rng.exponential(np.where(tier == "premium", 320.0, 200.0))
    churn = pd.Series(signup + pd.to_timedelta(lifetime, unit="D"))
    return pd.DataFrame(
        {
            "cid": [f"c{i}" for i in range(n)],
            "start": signup,
            "churn": churn.where(churn <= pd.Timestamp("2026-05-31")),
            "tier": tier,
            "age": age,
        }
    )


def _fit():
    design = StudyDesign.from_event_dates(
        _cox_df(),
        id_col="cid",
        origin_col="start",
        churn_date_col="churn",
        active_as_of="2026-05-31",
        covariate_cols=["tier", "age"],
    )
    return CoxPH().fit(design)


def test_scores_table_shape_and_columns():
    cox = _fit()
    scores = tenure.churn_risk_scores(cox, horizon=365.0)
    assert list(scores.table.columns) == [
        "id",
        "risk_score",
        "survival_at_horizon",
        "risk_percentile",
    ]
    assert scores.metadata["n_customers"] == cox.design.n
    assert len(scores.table) == cox.design.n


def test_risk_score_matches_lifelines_partial_hazard():
    cox = _fit()
    scores = tenure.churn_risk_scores(cox, horizon=365.0)
    encoded = cox.encode_for_prediction(cox.design)
    expected = cox.fitter.predict_partial_hazard(encoded).to_numpy()
    assert np.allclose(scores.table["risk_score"].to_numpy(), expected, atol=1e-9)


def test_survival_at_horizon_matches_lifelines():
    cox = _fit()
    scores = tenure.churn_risk_scores(cox, horizon=180.0)
    encoded = cox.encode_for_prediction(cox.design)
    expected = cox.fitter.predict_survival_function(encoded, times=[180.0]).iloc[0].to_numpy()
    assert np.allclose(scores.table["survival_at_horizon"].to_numpy(), expected, atol=1e-9)


def test_percentile_is_a_clean_rank():
    cox = _fit()
    table = tenure.churn_risk_scores(cox, horizon=365.0).table.sort_values("risk_score")
    pct = table["risk_percentile"]
    assert ((pct > 0) & (pct <= 1)).all()
    # Percentile rises with risk score (ties share a percentile under averaged ranking).
    assert pct.is_monotonic_increasing
    assert np.isclose(pct.iloc[-1], pct.max())


def test_premium_customers_are_lower_risk():
    cox = _fit()
    table = tenure.churn_risk_scores(cox, horizon=365.0).table
    table["tier"] = cox.design.derive()["tier"].to_numpy()
    mean_risk = table.groupby("tier")["risk_score"].mean()
    assert mean_risk["premium"] < mean_risk["basic"]


def test_provenance_metadata():
    cox = _fit()
    report = tenure.audit(cox.design, strictness="block")  # event-date, no analysis_start -> clean
    scores = tenure.churn_risk_scores(cox, horizon=365.0, audit_report=report)
    assert scores.metadata["audit_verdict"] == "clean (no findings)"
    assert scores.metadata["covariates"] == ["tier", "age"]
    csv = scores.to_csv()
    assert "# horizon: 365.0" in csv
    assert "risk_score" in csv


def test_explicit_design_matches_default():
    cox = _fit()
    default = tenure.churn_risk_scores(cox, horizon=365.0).table
    explicit = tenure.churn_risk_scores(cox, cox.design, horizon=365.0).table
    pd.testing.assert_frame_equal(default, explicit)
