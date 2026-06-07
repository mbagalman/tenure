from __future__ import annotations

import warnings

import matplotlib

matplotlib.use("Agg")  # headless backend for CI

import matplotlib.pyplot as plt  # noqa: E402
import pytest  # noqa: E402

import tenure  # noqa: E402


def _km_by_plan():
    df = tenure.load_svod_demo(with_left_truncation=False, seed=0)
    design = tenure.StudyDesign.from_event_dates(
        df,
        id_col="customer_id",
        origin_col="signup_date",
        churn_date_col="churn_date",
        active_as_of="2026-05-31",
        group_cols=["plan"],
    )
    return tenure.KaplanMeier().fit(design, by="plan")


def teardown_function():
    plt.close("all")


def test_plot_returns_axes_with_a_line_per_group():
    ax = tenure.plot_survival(_km_by_plan(), at_risk=False)
    assert len(ax.lines) == 3  # basic / standard / premium


def test_at_risk_builds_a_second_panel():
    ax = tenure.plot_survival(_km_by_plan(), at_risk=True)
    fig = ax.figure
    assert len(fig.axes) == 2  # curves + number-at-risk table
    table_ax = fig.axes[1]
    # The table row labels are the groups.
    assert set(t.get_text() for t in table_ax.get_yticklabels()) == {"basic", "standard", "premium"}
    # Number-at-risk text cells were drawn.
    assert len(table_ax.texts) > 0


def test_at_risk_requires_a_fresh_figure():
    _, ax = plt.subplots()
    with pytest.raises(ValueError):
        tenure.plot_survival(_km_by_plan(), at_risk=True, ax=ax)


def test_ci_band_drawn_when_requested():
    ax = tenure.plot_survival(_km_by_plan(), at_risk=False, ci=True)
    assert len(ax.collections) >= 3  # one fill_between per group


def test_no_caveat_on_clean_design():
    ax = tenure.plot_survival(_km_by_plan(), at_risk=False)
    assert not any("Caveat" in t.get_text() for t in ax.figure.texts)


def test_caveat_stamp_on_bypassed_warnings():
    df = tenure.load_svod_demo(with_left_truncation=True, seed=2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = tenure.RetentionStudy.from_event_dates(
            df,
            id_col="customer_id",
            origin_col="signup_date",
            churn_date_col="churn_date",
            active_as_of="2026-05-31",
            analysis_start="2024-01-01",
            includes_pre_entry_churners=False,
            strictness="warn",
        ).run()
    ax = result.plot(at_risk=False)  # via RetentionResult.plot delegation
    caveats = [t.get_text() for t in ax.figure.texts if "Caveat" in t.get_text()]
    assert any("TNR001" in c for c in caveats)
