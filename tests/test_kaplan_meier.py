from __future__ import annotations

import numpy as np
import pytest
from lifelines import KaplanMeierFitter

import tenure
from tenure._frame import as_estimator_frame


def _design(*, with_left_truncation=False, seed=0, **kwargs):
    df = tenure.load_svod_demo(with_left_truncation=with_left_truncation, seed=seed)
    return tenure.StudyDesign.from_event_dates(
        df,
        id_col="customer_id",
        origin_col="signup_date",
        churn_date_col="churn_date",
        active_as_of="2026-05-31",
        group_cols=["plan"],
        **kwargs,
    )


def _lifelines_reference(design):
    ef = as_estimator_frame(design.derive())
    kmf = KaplanMeierFitter()
    kmf.fit(durations=ef.duration, event_observed=ef.event, entry=ef.entry)
    return kmf


def _ours_at(km, times, group):
    frame = km.survival_at(times, group=group).sort_values("time")
    return frame["survival"].to_numpy()


def test_km_survival_matches_lifelines():  # AC-1
    design = _design(seed=0)
    km = tenure.KaplanMeier().fit(design)
    kmf = _lifelines_reference(design)
    times = np.array([10.0, 50.0, 100.0, 200.0, 365.0])
    ours = _ours_at(km, times, "overall")
    theirs = kmf.survival_function_at_times(times).to_numpy()
    assert np.allclose(ours, theirs, atol=1e-9)


def test_km_confidence_interval_matches_lifelines():  # AC-1 (CI)
    design = _design(seed=0)
    km = tenure.KaplanMeier().fit(design)
    kmf = _lifelines_reference(design)
    # Compare at the curve's own jump points (exact, no interpolation ambiguity).
    sf = kmf.survival_function_
    ci = kmf.confidence_interval_
    times = sf.index.to_numpy(dtype=float)
    frame = km.survival_at(times, group="overall").sort_values("time")
    assert np.allclose(frame["survival"].to_numpy(), sf.iloc[:, 0].to_numpy(), atol=1e-9)
    assert np.allclose(frame["ci_lower"].to_numpy(), ci.iloc[:, 0].to_numpy(), atol=1e-9)
    assert np.allclose(frame["ci_upper"].to_numpy(), ci.iloc[:, 1].to_numpy(), atol=1e-9)


def test_km_median_matches_lifelines():
    design = _design(seed=0)
    km = tenure.KaplanMeier().fit(design)
    kmf = _lifelines_reference(design)
    ours = km.median_survival(group="overall")["median"].iloc[0]
    theirs = kmf.median_survival_time_
    assert (np.isinf(ours) and np.isinf(theirs)) or np.isclose(ours, theirs, atol=1e-9)


def test_km_delayed_entry_matches_lifelines():  # AC-2
    design = _design(with_left_truncation=True, seed=2, event_observed_from="2024-01-01")
    assert design.entry_modeled
    km = tenure.KaplanMeier().fit(design)
    kmf = _lifelines_reference(design)  # reference also fed entry=...
    times = np.array([100.0, 365.0, 700.0])
    ours = _ours_at(km, times, "overall")
    theirs = kmf.survival_function_at_times(times).to_numpy()
    assert np.allclose(ours, theirs, atol=1e-9)


def test_km_multigroup_tidy_contract():  # D-S1
    design = _design(seed=0)
    km = tenure.KaplanMeier().fit(design, by="plan")
    assert set(km.survival_.groups) == {"basic", "standard", "premium"}

    frame = km.survival_at([30, 90, 365], group=None)
    assert list(frame.columns) == ["group", "time", "survival", "ci_lower", "ci_upper"]
    assert set(frame["group"].unique()) == {"basic", "standard", "premium"}
    assert isinstance(frame.index, type(frame.reset_index(drop=True).index))  # default RangeIndex

    one = km.survival_at([30], group="basic")
    assert (one["group"] == "basic").all()
    assert len(one) == 1

    # Survival is a probability and non-increasing in time within a group.
    g = frame[frame["group"] == "basic"].sort_values("time")
    assert ((g["survival"] >= 0) & (g["survival"] <= 1)).all()
    assert g["survival"].is_monotonic_decreasing


def test_km_ungrouped_labels_overall():
    design = _design(seed=1)
    km = tenure.KaplanMeier().fit(design)  # by=None
    assert km.survival_.groups == ["overall"]
    med = km.median_survival()  # group=None
    assert list(med.columns) == ["group", "median"]
    assert med["group"].iloc[0] == "overall"


def test_km_unknown_group_raises():
    km = tenure.KaplanMeier().fit(_design(seed=0), by="plan")
    with pytest.raises(KeyError):
        km.survival_at([30], group="enterprise")


def test_km_not_fitted_raises():
    with pytest.raises(RuntimeError):
        tenure.KaplanMeier().survival_at([1.0])


def test_km_bad_by_column_raises():
    with pytest.raises(tenure.TenureValidationError):
        tenure.KaplanMeier().fit(_design(seed=0), by="nonexistent")
