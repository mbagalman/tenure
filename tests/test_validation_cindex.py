from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from lifelines.utils import concordance_index

import tenure
from tenure import StudyDesign, TenureValidationError


def _scored_events(n: int = 800, seed: int = 0) -> StudyDesign:
    """Customers with staggered signups and a numeric `x` that shortens survival (raises hazard)."""
    rng = np.random.default_rng(seed)
    origin0 = pd.Timestamp("2022-01-01")
    active_as_of = pd.Timestamp("2024-01-01")
    rows = []
    for i in range(n):
        signup = origin0 + pd.Timedelta(days=int(rng.integers(0, 200)))
        x = float(rng.normal())
        t = float(rng.exponential(300.0) * np.exp(-0.6 * x))  # higher x -> shorter survival
        churn = signup + pd.Timedelta(days=t)
        rows.append(
            {
                "cid": i,
                "signup": signup,
                "churn": churn if churn <= active_as_of else pd.NaT,
                "x": x,
            }
        )
    return StudyDesign.from_event_dates(
        pd.DataFrame(rows),
        id_col="cid",
        origin_col="signup",
        churn_date_col="churn",
        active_as_of=active_as_of,
        covariate_cols=["x"],
    )


def _cohort(durations, events) -> tenure.TestCohort:
    table = pd.DataFrame(
        {
            "id": range(len(durations)),
            "eval_start": 0.0,
            "eval_duration": np.asarray(durations, dtype=float),
            "eval_event": np.asarray(events, dtype=int),
        }
    )
    return tenure.TestCohort(
        table=table,
        paths=pd.DataFrame(),
        prediction_time=pd.Timestamp("2022-10-01"),
        time_unit="day",
        covariate_cols=[],
    )


def test_concordance_matches_lifelines_oracle_for_cox():
    design = _scored_events()
    train, test = tenure.temporal_holdout(design, "2022-10-01")
    cox = tenure.CoxPH().fit(train)
    result = tenure.concordance(cox, test)

    # INDEPENDENT oracle: extract risk straight from the fitter and call lifelines ourselves.
    encoded = cox.design.encode_covariates(test.table).reindex(
        columns=cox.fitter.params_.index, fill_value=0.0
    )
    risk = cox.fitter.predict_partial_hazard(encoded).to_numpy()
    expected = concordance_index(test.table["eval_duration"], -risk, test.table["eval_event"])
    assert np.isclose(result.estimate, expected)
    assert result.metadata["metric"] == "c_index"
    assert result.metadata["model_type"] == "CoxPH"
    assert result.metadata["n_test"] == test.n


def test_cox_discriminates_above_chance():
    design = _scored_events()
    train, test = tenure.temporal_holdout(design, "2022-10-01")
    cox = tenure.CoxPH().fit(train)
    assert tenure.concordance(cox, test).estimate > 0.55  # the covariate genuinely discriminates


def test_concordance_orientation_is_hand_verifiable():
    # Higher risk must mean shorter survival -> perfect ranking scores 1.0, inverted scores 0.0.
    cohort = _cohort([10, 20, 30, 40], [1, 1, 1, 1])
    assert np.isclose(tenure.concordance(np.array([4.0, 3, 2, 1]), cohort).estimate, 1.0)
    assert np.isclose(tenure.concordance(np.array([1.0, 2, 3, 4]), cohort).estimate, 0.0)


def test_overall_km_is_uninformative():
    design = _scored_events()
    train, test = tenure.temporal_holdout(design, "2022-10-01")
    km = tenure.KaplanMeier().fit(train)
    # Overall KM is constant-risk across subjects (DV4-5): exactly 0.5 (all ties).
    assert abs(tenure.concordance(km, test, horizon=180).estimate - 0.5) < 1e-9


def _scored_intervals(n: int = 400, seed: int = 1) -> StudyDesign:
    rng = np.random.default_rng(seed)
    o0 = pd.Timestamp("2022-01-01")
    rows = []
    for i in range(n):
        signup = o0 + pd.Timedelta(days=int(rng.integers(0, 150)))
        x = float(rng.normal())
        t = float(rng.exponential(300.0) * np.exp(-0.6 * x))
        churned = t <= 700
        end_t = min(t, 700.0)
        if end_t > 60:
            rows.append(
                {
                    "cid": i,
                    "origin": signup,
                    "start": signup,
                    "end": signup + pd.Timedelta(days=60),
                    "event": 0,
                    "x": x,
                }
            )
            rows.append(
                {
                    "cid": i,
                    "origin": signup,
                    "start": signup + pd.Timedelta(days=60),
                    "end": signup + pd.Timedelta(days=end_t),
                    "event": int(churned),
                    "x": x,
                }
            )
        else:
            rows.append(
                {
                    "cid": i,
                    "origin": signup,
                    "start": signup,
                    "end": signup + pd.Timedelta(days=end_t),
                    "event": int(churned),
                    "x": x,
                }
            )
    return StudyDesign.from_intervals(
        pd.DataFrame(rows),
        id_col="cid",
        origin_col="origin",
        interval_start_col="start",
        interval_end_col="end",
        event_col="event",
        covariate_cols=["x"],
    )


def test_concordance_runs_for_time_varying_cox():
    design = _scored_intervals()
    train, test = tenure.temporal_holdout(design, "2022-06-01")
    tvc = tenure.TimeVaryingCox().fit(train)
    result = tenure.concordance(tvc, test)
    assert 0.0 <= result.estimate <= 1.0
    assert result.metadata["model_type"] == "TimeVaryingCox"


def test_km_without_horizon_raises():
    design = _scored_events()
    train, test = tenure.temporal_holdout(design, "2022-10-01")
    km = tenure.KaplanMeier().fit(train)
    with pytest.raises(TenureValidationError, match="horizon"):
        tenure.concordance(km, test)


def test_risk_length_mismatch_raises():
    cohort = _cohort([1, 2, 3], [1, 1, 1])
    with pytest.raises(TenureValidationError, match="length"):
        tenure.concordance(np.array([1.0, 2.0]), cohort)
