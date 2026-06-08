from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import tenure
from tenure import StudyDesign, TenureValidationError
from tenure.validation.metrics import _conditional_survival_matrix, _ipcw_brier_scores

# --- IPCW Brier math, hand-computed (the independent oracle for our implementation) ------------


def test_ipcw_brier_no_censoring_reduces_to_plain_brier():
    durations = np.array([1.0, 2, 3, 4])
    events = np.array([1, 1, 1, 1])  # no censoring -> G == 1 -> plain Brier
    estimate = np.array([[0.9, 0.8], [0.7, 0.6], [0.5, 0.4], [0.3, 0.2]])
    times = np.array([1.5, 2.5])
    bs = _ipcw_brier_scores(durations, events, estimate, times)
    exp0 = (0.9**2 + (1 - 0.7) ** 2 + (1 - 0.5) ** 2 + (1 - 0.3) ** 2) / 4
    exp1 = (0.8**2 + 0.6**2 + (1 - 0.4) ** 2 + (1 - 0.2) ** 2) / 4
    assert np.allclose(bs, [exp0, exp1])


def test_ipcw_brier_perfect_is_zero_and_half_is_quarter():
    durations = np.array([1.0, 2, 3, 4])
    events = np.array([1, 1, 1, 1])
    times = np.array([2.5])
    perfect = np.array([[0.0], [0.0], [1.0], [1.0]])  # S=0 for the dead-by-2.5, S=1 for survivors
    assert np.isclose(_ipcw_brier_scores(durations, events, perfect, times)[0], 0.0)
    half = np.full((4, 1), 0.5)
    assert np.isclose(_ipcw_brier_scores(durations, events, half, times)[0], 0.25)


def test_ipcw_brier_with_censoring_hand_computed():
    # subj1 censored at t=2 -> censoring KM G(s>=2) = 2/3, G(s<2) = 1.
    durations = np.array([1.0, 2, 3, 4])
    events = np.array([1, 0, 1, 1])
    estimate = np.array([[0.9], [0.8], [0.5], [0.3]])
    times = np.array([2.5])
    bs = _ipcw_brier_scores(durations, events, estimate, times)
    # dead-by-2.5: subj0 (weight 1/G(1)=1). at-risk: subj2, subj3 (weight 1/G(2.5)=1.5).
    # censored subj1 contributes 0.
    expected = (0.9**2 * 1 + ((1 - 0.5) ** 2 + (1 - 0.3) ** 2) * 1.5) / 4
    assert np.isclose(bs[0], expected)


# --- conditional prediction + end-to-end ------------------------------------------------------


def _scored_design(n: int = 800, seed: int = 0) -> StudyDesign:
    rng = np.random.default_rng(seed)
    o0 = pd.Timestamp("2022-01-01")
    active = pd.Timestamp("2024-01-01")
    rows = []
    for i in range(n):
        signup = o0 + pd.Timedelta(days=int(rng.integers(0, 200)))
        x = float(rng.normal())
        t = float(rng.exponential(300.0) * np.exp(-0.6 * x))
        churn = signup + pd.Timedelta(days=t)
        rows.append(
            {"cid": i, "signup": signup, "churn": churn if churn <= active else pd.NaT, "x": x}
        )
    return StudyDesign.from_event_dates(
        pd.DataFrame(rows),
        id_col="cid",
        origin_col="signup",
        churn_date_col="churn",
        active_as_of=active,
        covariate_cols=["x"],
    )


def _scored_intervals(n: int = 400, seed: int = 1) -> StudyDesign:
    rng = np.random.default_rng(seed)
    o0 = pd.Timestamp("2022-01-01")
    rows = []
    for i in range(n):
        signup = o0 + pd.Timedelta(days=int(rng.integers(0, 150)))
        x = float(rng.normal())
        t = float(rng.exponential(300.0) * np.exp(-0.6 * x))
        end_t = min(t, 700.0)
        churned = t <= 700
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


def test_cox_conditional_matrix_matches_independent_fitter():
    design = _scored_design()
    train, test = tenure.temporal_holdout(design, "2022-10-01")
    cox = tenure.CoxPH().fit(train)
    times = np.array([20.0, 50.0])
    est, _beyond = _conditional_survival_matrix(cox, test, times)

    encoded = cox.design.encode_covariates(test.table).reindex(
        columns=cox.fitter.params_.index, fill_value=0.0
    )
    surv = cox.fitter.predict_survival_function(encoded)
    ti = surv.index.to_numpy(dtype=float)
    s_mat = surv.to_numpy()
    c = test.table["eval_start"].to_numpy(dtype=float)

    def step(col, q):
        idx = int(np.searchsorted(ti, q, side="right")) - 1
        return col[idx] if idx >= 0 else 1.0

    for i in (0, 5, 25):
        s0 = step(s_mat[:, i], c[i])
        for j, t in enumerate(times):
            expected = (step(s_mat[:, i], c[i] + t) / s0) if s0 > 0 else 0.0
            assert np.isclose(est[i, j], np.clip(expected, 0.0, 1.0))


def test_brier_and_ibs_end_to_end_cox_beats_overall_km():
    design = _scored_design()
    train, test = tenure.temporal_holdout(design, "2022-10-01")
    cox = tenure.CoxPH().fit(train)
    km = tenure.KaplanMeier().fit(train)
    times = [30, 60, 90, 120]

    cox_brier = tenure.brier(cox, test, times)
    assert list(cox_brier.table.columns) == ["time", "brier"]
    assert cox_brier.table["brier"].between(0.0, 0.3).all()

    cox_ibs = tenure.integrated_brier(cox, test, times).estimate
    km_ibs = tenure.integrated_brier(km, test, times).estimate
    assert 0.0 < cox_ibs < 0.25
    assert cox_ibs < km_ibs  # the informative covariate improves over the cohort curve


def test_brier_warns_val002_beyond_support():
    design = _scored_design()
    train, test = tenure.temporal_holdout(design, "2022-10-01")
    cox = tenure.CoxPH().fit(train)
    max_d = float(test.table["eval_duration"].max())
    with pytest.warns(UserWarning, match="VAL002"):
        result = tenure.brier(cox, test, [30, max_d + 100])
    assert result.metadata["warnings"] == ["VAL002_HORIZON_SUPPORT"]


def test_brier_unsupported_model_raises():
    design = _scored_intervals()
    train, test = tenure.temporal_holdout(design, "2022-06-01")
    tvc = tenure.TimeVaryingCox().fit(train)
    with pytest.raises(TenureValidationError, match="not yet supported"):
        tenure.brier(tvc, test, [30, 60])


def test_integrated_brier_needs_two_times():
    design = _scored_design()
    train, test = tenure.temporal_holdout(design, "2022-10-01")
    cox = tenure.CoxPH().fit(train)
    with pytest.raises(TenureValidationError, match="2 time"):
        tenure.integrated_brier(cox, test, [60])


def test_brier_rejects_nonpositive_times():
    design = _scored_design()
    train, test = tenure.temporal_holdout(design, "2022-10-01")
    cox = tenure.CoxPH().fit(train)
    with pytest.raises(TenureValidationError, match="> 0"):
        tenure.brier(cox, test, [-1, 30])


def test_integrated_brier_rejects_unsorted_or_duplicate_times():
    # An unsorted or duplicate grid would silently change the trapezoid IBS -- reject it.
    design = _scored_design()
    train, test = tenure.temporal_holdout(design, "2022-10-01")
    cox = tenure.CoxPH().fit(train)
    with pytest.raises(TenureValidationError, match="increasing"):
        tenure.integrated_brier(cox, test, [90, 30, 60])
    with pytest.raises(TenureValidationError, match="increasing"):
        tenure.integrated_brier(cox, test, [30, 30, 60])


def test_brier_warns_val002_beyond_model_support():
    # Times within the test follow-up but, for the longest-tenure subjects, beyond the fitted
    # model's tenure support after conditioning (eval_start + t) -> flat extrapolation -> VAL002.
    design = _scored_design()
    train, test = tenure.temporal_holdout(design, "2022-10-01")
    cox = tenure.CoxPH().fit(train)
    with pytest.warns(UserWarning, match="VAL002"):
        result = tenure.brier(cox, test, [30, 60])
    assert "VAL002_HORIZON_SUPPORT" in result.metadata["warnings"]
