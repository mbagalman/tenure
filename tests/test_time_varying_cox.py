from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from lifelines import CoxTimeVaryingFitter

import tenure
from tenure import StudyDesign, TenureValidationError, TimeVaryingCox
from tenure._frame import ENTRY, EVENT, EXIT, ID


def _tv_intervals(n: int = 300, seed: int = 0) -> pd.DataFrame:
    """Two 30-day intervals per subject; a time-varying numeric `score` drives churn risk.

    Higher `score` during the second interval raises churn probability, so the recovered Cox
    coefficient for `score` should be clearly positive.
    """
    rng = np.random.default_rng(seed)
    origin = pd.Timestamp("2024-01-01")
    m1 = origin + pd.Timedelta(days=30)
    m2 = origin + pd.Timedelta(days=60)
    rows = []
    for i in range(n):
        cid = f"s{i}"
        score1 = float(rng.normal(0.0, 1.0))
        score2 = float(rng.normal(0.0, 1.0))
        churn = rng.random() < 1.0 / (1.0 + np.exp(-1.5 * score2))
        rows.append(
            {"cid": cid, "origin": origin, "start": origin, "end": m1, "event": 0, "score": score1}
        )
        rows.append(
            {
                "cid": cid,
                "origin": origin,
                "start": m1,
                "end": m2,
                "event": int(churn),
                "score": score2,
            }
        )
    return pd.DataFrame(rows)


def _design(df: pd.DataFrame | None = None) -> StudyDesign:
    return StudyDesign.from_intervals(
        _tv_intervals() if df is None else df,
        id_col="cid",
        origin_col="origin",
        interval_start_col="start",
        interval_end_col="end",
        event_col="event",
        covariate_cols=["score"],
    )


def test_coefficients_match_lifelines():
    design = _design()
    tvc = TimeVaryingCox().fit(design)

    # Build the same start-stop frame lifelines sees and fit a bare CoxTimeVaryingFitter.
    table = design.derive()
    ref = design.encode_covariates(table)
    ref["__id__"] = table[ID].to_numpy()
    ref["__start__"] = table[ENTRY].to_numpy(dtype=float)
    ref["__stop__"] = table[EXIT].to_numpy(dtype=float)
    ref["__event__"] = table[EVENT].to_numpy(dtype=int)
    ctv = CoxTimeVaryingFitter().fit(
        ref, id_col="__id__", event_col="__event__", start_col="__start__", stop_col="__stop__"
    )

    ours = tvc.fitter.params_.sort_index()
    theirs = ctv.params_.sort_index()
    assert list(ours.index) == list(theirs.index)
    assert np.allclose(ours.to_numpy(), theirs.to_numpy(), atol=1e-9)


def test_recovers_positive_coefficient():
    tvc = TimeVaryingCox().fit(_design())
    summary = tvc.summary
    score = summary.loc[summary["covariate"] == "score"].iloc[0]
    assert score["coef"] > 0.0
    assert score["hazard_ratio"] > 1.0
    assert score["p_value"] < 0.05


def test_summary_shape():
    tvc = TimeVaryingCox().fit(_design())
    assert list(tvc.summary.columns) == ["covariate", "coef", "hazard_ratio", "p_value"]
    assert tvc.summary["covariate"].tolist() == ["score"]


def test_risk_scores_are_time_varying_and_ordered():
    tvc = TimeVaryingCox().fit(_design())
    scores = tvc.risk_scores()
    assert list(scores.columns) == ["id", "interval_start", "interval_stop", "risk_score"]
    assert len(scores) == 600  # one row per interval

    # risk_score == exp(predict_log_partial_hazard) on the same encoded design.
    encoded = tvc.encode_for_prediction(tvc.design)
    expected = np.exp(tvc.fitter.predict_log_partial_hazard(encoded).to_numpy(dtype=float))
    assert np.allclose(scores["risk_score"].to_numpy(), expected, atol=1e-12)

    # risk_score = exp(coef * centered score) with coef > 0 -> strictly increasing in `score`,
    # so the rank order is identical (Pearson is < 1 only because exp is nonlinear).
    table = tvc.design.derive()
    order_by_score = np.argsort(table["score"].to_numpy(dtype=float), kind="stable")
    order_by_risk = np.argsort(scores["risk_score"].to_numpy(), kind="stable")
    assert np.array_equal(order_by_score, order_by_risk)

    # A subject's two intervals carry (generally) different scores => different risk.
    first_two = scores.iloc[:2]["risk_score"].to_numpy()
    assert not np.isclose(first_two[0], first_two[1])


def test_requires_interval_design():
    df = tenure.load_svod_demo(with_left_truncation=False, seed=0)
    df["plan_code"] = (df["customer_id"].astype(str).str[-1].astype(int) % 2).astype(float)
    single = StudyDesign.from_event_dates(
        df,
        id_col="customer_id",
        origin_col="signup_date",
        churn_date_col="churn_date",
        active_as_of="2026-05-31",
        covariate_cols=["plan_code"],
    )
    with pytest.raises(TenureValidationError, match="interval"):
        TimeVaryingCox().fit(single)


def test_requires_covariates():
    plain = StudyDesign.from_intervals(
        _tv_intervals(n=50),
        id_col="cid",
        origin_col="origin",
        interval_start_col="start",
        interval_end_col="end",
        event_col="event",
    )
    with pytest.raises(TenureValidationError, match="covariate_cols"):
        TimeVaryingCox().fit(plain)


def test_predict_before_fit_raises():
    with pytest.raises(RuntimeError, match="not fitted"):
        TimeVaryingCox().risk_scores()
