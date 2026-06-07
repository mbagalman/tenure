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


# --- Path survival (DV3-4) --------------------------------------------------------------------


def _varied_intervals(n: int = 400, seed: int = 0) -> pd.DataFrame:
    """Counting-process data with VARIED event times (a non-degenerate baseline hazard)."""
    rng = np.random.default_rng(seed)
    o = pd.Timestamp("2024-01-01")
    rows = []
    for i in range(n):
        x = float(rng.normal())
        t = float(rng.exponential(40.0) * np.exp(-0.4 * x))
        stop = min(max(1, round(t)), 120)
        event = 1 if round(t) <= 120 else 0
        if stop > 30:
            rows.append(
                {
                    "cid": i,
                    "origin": o,
                    "start": o,
                    "end": o + pd.Timedelta(days=30),
                    "event": 0,
                    "score": x,
                }
            )
            rows.append(
                {
                    "cid": i,
                    "origin": o,
                    "start": o + pd.Timedelta(days=30),
                    "end": o + pd.Timedelta(days=stop),
                    "event": event,
                    "score": x,
                }
            )
        else:
            rows.append(
                {
                    "cid": i,
                    "origin": o,
                    "start": o,
                    "end": o + pd.Timedelta(days=stop),
                    "event": event,
                    "score": x,
                }
            )
    return pd.DataFrame(rows)


def _varied_design() -> StudyDesign:
    return StudyDesign.from_intervals(
        _varied_intervals(),
        id_col="cid",
        origin_col="origin",
        interval_start_col="start",
        interval_end_col="end",
        event_col="event",
        covariate_cols=["score"],
    )


def _path(segments: list[tuple[float, float, float]]) -> StudyDesign:
    """A single hypothetical customer's covariate path: list of (start_day, stop_day, score)."""
    o = pd.Timestamp("2024-01-01")
    rows = [
        {
            "cid": "H",
            "origin": o,
            "start": o + pd.Timedelta(days=a),
            "end": o + pd.Timedelta(days=b),
            "event": 0,
            "score": sc,
        }
        for (a, b, sc) in segments
    ]
    return StudyDesign.from_intervals(
        pd.DataFrame(rows),
        id_col="cid",
        origin_col="origin",
        interval_start_col="start",
        interval_end_col="end",
        event_col="event",
        covariate_cols=["score"],
    )


def test_constant_path_matches_static_cox_oracle():
    # INDEPENDENT oracle: a constant covariate path through the time-varying Cox must equal a
    # static lifelines CoxPHFitter at that covariate value (its predict_survival_function handles
    # centering correctly). This catches the centering bug a self-referential baseline check misses.
    from lifelines import CoxPHFitter

    rng = np.random.default_rng(0)
    n = 600
    x = rng.normal(size=n)
    t = rng.exponential(40.0, size=n) * np.exp(-0.4 * x)
    stop = np.clip(np.round(t), 1, 120).astype(float)
    event = (np.round(t) <= 120).astype(int)

    cph = CoxPHFitter().fit(
        pd.DataFrame({"dur": stop, "event": event, "x": x}), duration_col="dur", event_col="event"
    )
    o = pd.Timestamp("2024-01-01")
    interval_df = pd.DataFrame(
        {
            "cid": np.arange(n),
            "origin": o,
            "start": o,
            "end": o + pd.to_timedelta(stop, unit="D"),
            "event": event,
            "x": x,
        }
    )
    design = StudyDesign.from_intervals(
        interval_df,
        id_col="cid",
        origin_col="origin",
        interval_start_col="start",
        interval_end_col="end",
        event_col="event",
        covariate_cols=["x"],
    )
    tvc = TimeVaryingCox().fit(design)

    x0 = 0.8
    o2 = pd.Timestamp("2024-01-01")
    path = StudyDesign.from_intervals(
        pd.DataFrame(
            {
                "cid": ["H"],
                "origin": [o2],
                "start": [o2],
                "end": [o2 + pd.Timedelta(days=120)],
                "event": [0],
                "x": [x0],
            }
        ),
        id_col="cid",
        origin_col="origin",
        interval_start_col="start",
        interval_end_col="end",
        event_col="event",
        covariate_cols=["x"],
    )
    sf = tvc.predict_survival(path)

    ref = cph.predict_survival_function(pd.DataFrame({"x": [x0]}))
    rt = ref.index.to_numpy(dtype=float)
    rs = ref.iloc[:, 0].to_numpy(dtype=float)
    test_times = rt[(rt > 0) & (rt <= 120)][:15]
    got = sf.survival_at(test_times, group="path").sort_values("time")["survival"].to_numpy()
    expected = np.array(
        [rs[int(np.searchsorted(rt, q, side="right")) - 1] for q in np.sort(test_times)]
    )
    assert np.allclose(got, expected, atol=1e-6)


def test_time_varying_path_matches_manual_integration():
    tvc = TimeVaryingCox().fit(_varied_design())
    sf = tvc.predict_survival(_path([(0.0, 30.0, 1.0), (30.0, 110.0, -1.0)]))

    beta = float(tvc.fitter.params_["score"])
    norm_mean = float(tvc.fitter._norm_mean["score"])  # lifelines centers covariates
    bch = tvc.fitter.baseline_cumulative_hazard_
    bt = bch.index.to_numpy(dtype=float)
    bv = bch.iloc[:, 0].to_numpy(dtype=float)

    def h0(t):
        idx = int(np.searchsorted(bt, t, side="right")) - 1
        return bv[idx] if idx >= 0 else 0.0

    # Centered partial hazards, consistent with the (centered) baseline cumulative hazard.
    p1 = float(np.exp(beta * (1.0 - norm_mean)))
    p2 = float(np.exp(beta * (-1.0 - norm_mean)))
    candidates = bt[(bt > 30) & (bt <= 110)]
    t = float(candidates[len(candidates) // 2])
    expected = float(np.exp(-(p1 * (h0(30.0) - h0(0.0)) + p2 * (h0(t) - h0(30.0)))))
    got = float(sf.survival_at([t], group="path").iloc[0]["survival"])
    assert np.isclose(got, expected, atol=1e-9)


def test_path_survival_feeds_business_outputs():
    # The predicted curve is a SurvivalFunction, so retention/RMST consume it unchanged (A3/A8).
    tvc = TimeVaryingCox().fit(_varied_design())
    sf = tvc.predict_survival(_path([(0.0, 30.0, 0.5), (30.0, 100.0, 0.5)]))
    retention = tenure.retention_at(sf, [30, 60])
    assert retention["retention"].between(0.0, 1.0).all()
    assert float(tenure.rmst(sf, horizon=90).iloc[0]["rmst"]) > 0.0


def test_predict_survival_requires_interval_path():
    tvc = TimeVaryingCox().fit(_varied_design())
    df = tenure.load_svod_demo(with_left_truncation=False, seed=0)
    single = StudyDesign.from_event_dates(
        df,
        id_col="customer_id",
        origin_col="signup_date",
        churn_date_col="churn_date",
        active_as_of="2026-05-31",
    )
    with pytest.raises(TenureValidationError, match="from_intervals"):
        tvc.predict_survival(single)


def test_predict_survival_requires_single_subject_path():
    tvc = TimeVaryingCox().fit(_varied_design())
    o = pd.Timestamp("2024-01-01")
    df = pd.DataFrame(
        {
            "cid": ["A", "B"],
            "origin": [o, o],
            "start": [o, o],
            "end": [o + pd.Timedelta(days=30), o + pd.Timedelta(days=30)],
            "event": [0, 0],
            "score": [0.1, 0.2],
        }
    )
    two = StudyDesign.from_intervals(
        df,
        id_col="cid",
        origin_col="origin",
        interval_start_col="start",
        interval_end_col="end",
        event_col="event",
        covariate_cols=["score"],
    )
    with pytest.raises(TenureValidationError, match="single hypothetical"):
        tvc.predict_survival(two)
