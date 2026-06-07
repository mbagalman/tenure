from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from lifelines import NelsonAalenFitter  # noqa: E402

import tenure  # noqa: E402
from tenure._frame import as_estimator_frame  # noqa: E402


def teardown_function():
    plt.close("all")


def _design(*, with_lt=False, group=False, **kwargs):
    df = tenure.load_svod_demo(with_left_truncation=with_lt, seed=0)
    return tenure.StudyDesign.from_event_dates(
        df,
        id_col="customer_id",
        origin_col="signup_date",
        churn_date_col="churn_date",
        active_as_of="2026-05-31",
        group_cols=["plan"] if group else None,
        **kwargs,
    )


def _lifelines_reference(design):
    ef = as_estimator_frame(design.derive())
    return NelsonAalenFitter().fit(durations=ef.duration, event_observed=ef.event, entry=ef.entry)


def test_cumulative_hazard_matches_lifelines():
    design = _design()
    na = tenure.NelsonAalen().fit(design)
    naf = _lifelines_reference(design)
    times = np.array([10.0, 50.0, 100.0, 365.0])
    ours = (
        na.cumulative_hazard_at(times, group="overall")
        .sort_values("time")["cumulative_hazard"]
        .to_numpy()
    )
    assert np.allclose(ours, naf.cumulative_hazard_at_times(times).to_numpy(), atol=1e-9)


def test_delayed_entry_matches_lifelines():
    design = _design(with_lt=True, event_observed_from="2024-01-01")
    na = tenure.NelsonAalen().fit(design)
    naf = _lifelines_reference(design)
    times = np.array([100.0, 365.0, 700.0])
    ours = (
        na.cumulative_hazard_at(times, group="overall")
        .sort_values("time")["cumulative_hazard"]
        .to_numpy()
    )
    assert np.allclose(ours, naf.cumulative_hazard_at_times(times).to_numpy(), atol=1e-9)


def test_groups_tidy_and_monotone():
    na = tenure.NelsonAalen().fit(_design(group=True), by="plan")
    assert set(na.cumulative_hazard_.groups) == {"basic", "standard", "premium"}
    frame = na.cumulative_hazard_at([30, 90, 365], group=None)
    assert list(frame.columns) == ["group", "time", "cumulative_hazard", "ci_lower", "ci_upper"]
    hazard = frame[frame["group"] == "basic"].sort_values("time")["cumulative_hazard"]
    assert hazard.is_monotonic_increasing  # cumulative hazard only rises


def test_not_fitted_raises():
    with pytest.raises(RuntimeError):
        tenure.NelsonAalen().cumulative_hazard_at([1.0])


def test_plot_cumulative_hazard():
    na = tenure.NelsonAalen().fit(_design(group=True), by="plan")
    ax = tenure.plot_cumulative_hazard(na)
    assert len(ax.lines) == 3
    assert "Cumulative hazard" in ax.get_ylabel()
