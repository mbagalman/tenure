from __future__ import annotations

import warnings

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

import tenure  # noqa: E402
from tenure import CoxPH, StudyDesign  # noqa: E402

_ACTIVE_AS_OF = "2026-12-31"
_SIGNUP = pd.Timestamp("2023-01-01")


def teardown_function():
    plt.close("all")


def _ph_respecting_df(n=600, seed=1):
    """Exponential lifetimes with a group-constant hazard ratio -> PH holds."""
    rng = np.random.default_rng(seed)
    grp = rng.choice(["a", "b"], size=n)
    life = rng.exponential(np.where(grp == "b", 320.0, 160.0))
    churn = pd.Series(_SIGNUP + pd.to_timedelta(life, unit="D"))
    return pd.DataFrame(
        {
            "cid": [f"c{i}" for i in range(n)],
            "start": _SIGNUP,
            "churn": churn.where(churn <= pd.Timestamp(_ACTIVE_AS_OF)),
            "grp": grp,
        }
    )


def _ph_violation_df(n=800, seed=2):
    """Opposite Weibull hazard shapes per group (decreasing vs increasing) -> the hazard ratio
    changes strongly over time, a textbook proportional-hazards violation."""
    rng = np.random.default_rng(seed)
    grp = rng.choice(["a", "b"], size=n)
    life = np.empty(n)
    is_a = grp == "a"
    life[is_a] = 200.0 * rng.weibull(0.6, int(is_a.sum()))  # decreasing hazard
    life[~is_a] = 200.0 * rng.weibull(2.5, int((~is_a).sum()))  # increasing hazard
    life = np.clip(life, 1.0, None)
    churn = pd.Series(_SIGNUP + pd.to_timedelta(life, unit="D"))
    return pd.DataFrame(
        {
            "cid": [f"c{i}" for i in range(n)],
            "start": _SIGNUP,
            "churn": churn.where(churn <= pd.Timestamp(_ACTIVE_AS_OF)),
            "grp": grp,
        }
    )


def _fit(df):
    design = StudyDesign.from_event_dates(
        df,
        id_col="cid",
        origin_col="start",
        churn_date_col="churn",
        active_as_of=_ACTIVE_AS_OF,
        covariate_cols=["grp"],
    )
    return CoxPH().fit(design)


def test_report_columns_and_structure():
    report = _fit(_ph_respecting_df()).proportional_hazards_test(warn=False)
    assert list(report.table.columns) == ["covariate", "test_statistic", "p_value", "status"]
    assert report.table["status"].isin(["pass", "fail"]).all()


def test_ph_respecting_passes():
    report = _fit(_ph_respecting_df()).proportional_hazards_test(warn=False)
    assert report.ok
    assert report.violations == []


def test_ph_violation_flagged():
    report = _fit(_ph_violation_df()).proportional_hazards_test(warn=False)
    assert not report.ok
    assert any(v.startswith("grp") for v in report.violations)


def test_warning_emitted_on_violation():
    cox = _fit(_ph_violation_df())
    with pytest.warns(UserWarning, match="Proportional-hazards"):
        cox.proportional_hazards_test()  # warn=True by default


def test_no_warning_when_ph_holds():
    cox = _fit(_ph_respecting_df())
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        report = cox.proportional_hazards_test()
    assert report.ok
    assert not any("Proportional-hazards" in str(w.message) for w in caught)


def test_log_log_plot_has_a_line_per_group():
    design = StudyDesign.from_event_dates(
        _ph_respecting_df(),
        id_col="cid",
        origin_col="start",
        churn_date_col="churn",
        active_as_of=_ACTIVE_AS_OF,
        group_cols=["grp"],
    )
    km = tenure.KaplanMeier().fit(design, by="grp")
    ax = tenure.plot_log_log_survival(km)
    assert len(ax.lines) == 2
    assert "log(-log" in ax.get_ylabel()
