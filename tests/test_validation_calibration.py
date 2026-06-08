from __future__ import annotations

import warnings

import matplotlib

matplotlib.use("Agg")  # headless backend for CI

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

import tenure  # noqa: E402
from tenure import StudyDesign, TenureValidationError  # noqa: E402


def _scored_design(n: int = 1000, seed: int = 0) -> StudyDesign:
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


def _fit_cox():
    train, test = tenure.temporal_holdout(_scored_design(), "2022-10-01")
    return tenure.CoxPH().fit(train), test


def test_calibration_table_and_metadata():
    # A short horizon extrapolates only a little (< 20%), so this stays quiet but still RECORDS it.
    cox, test = _fit_cox()
    result = tenure.calibration(cox, test, horizon=30, n_bins=10)
    assert list(result.table.columns) == ["bin", "mean_predicted", "observed", "n"]
    assert 2 <= len(result.table) <= 10
    assert result.table["mean_predicted"].between(0.0, 1.0).all()
    assert result.table["observed"].between(0.0, 1.0).all()
    assert int(result.table["n"].sum()) == test.n
    assert result.metadata["metric"] == "calibration"
    assert result.metadata["horizon"] == 30.0
    assert result.metadata["model_type"] == "CoxPH"
    assert 0.0 <= result.metadata["calibration_error"] <= 1.0
    assert result.metadata["n_extrapolated"] >= 0  # support info is surfaced...
    assert result.metadata["pct_extrapolated"] < 0.20  # ...but below the warn threshold here
    assert result.metadata["warnings"] == []


def test_well_specified_cox_is_reasonably_calibrated():
    # Train and test come from the same generating process, so the Cox model should track the
    # diagonal closely -- a small support-weighted predicted-vs-observed gap.
    cox, test = _fit_cox()
    assert tenure.calibration(cox, test, horizon=30).metadata["calibration_error"] < 0.05


def test_calibration_records_and_warns_on_material_extrapolation():
    # A far horizon extrapolates beyond model support for a material fraction -> VAL002, so the
    # diagram cannot look cleaner than the predictions actually are.
    cox, test = _fit_cox()
    with pytest.warns(UserWarning, match="VAL002"):
        result = tenure.calibration(cox, test, horizon=200)
    assert result.metadata["pct_extrapolated"] >= 0.20
    assert result.metadata["warnings"] == ["VAL002_HORIZON_SUPPORT"]


def test_calibration_rejects_nonpositive_horizon():
    cox, test = _fit_cox()
    with pytest.raises(TenureValidationError, match="> 0"):
        tenure.calibration(cox, test, horizon=0)


def test_calibration_rejects_small_n_bins():
    cox, test = _fit_cox()
    with pytest.raises(TenureValidationError, match=">= 2"):
        tenure.calibration(cox, test, horizon=30, n_bins=1)


def test_plot_calibration_returns_axes():
    cox, test = _fit_cox()
    result = tenure.calibration(cox, test, horizon=30)
    ax = tenure.plot_calibration(result)
    assert ax is not None
    assert len(ax.collections) >= 1  # the bin scatter
    plt.close("all")


def test_constant_predictions_raise():
    # Every subject signs up on the same day -> identical eval_start -> an overall KM predicts the
    # same survival for all -> no spread to bin -> a clear error rather than a cryptic one.
    o = pd.Timestamp("2022-01-01")
    active = pd.Timestamp("2024-01-01")
    rng = np.random.default_rng(0)
    rows = []
    for i in range(200):
        t = float(rng.exponential(300.0))
        churn = o + pd.Timedelta(days=t)
        rows.append({"cid": i, "signup": o, "churn": churn if churn <= active else pd.NaT})
    design = StudyDesign.from_event_dates(
        pd.DataFrame(rows),
        id_col="cid",
        origin_col="signup",
        churn_date_col="churn",
        active_as_of=active,
    )
    train, test = tenure.temporal_holdout(design, "2022-10-01")
    km = tenure.KaplanMeier().fit(train)
    # (these identical-tenure subjects also all extrapolate -> a material VAL002 fires before the
    # bin check; silence it since this test is about the distinct-values error.)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(TenureValidationError, match="distinct"):
            tenure.calibration(km, test, horizon=90)
