"""Nelson-Aalen cumulative hazard estimator (wraps lifelines; deferred v0.1 FR-RC-5).

A hazard analogue of the survival side: a multi-group `CumulativeHazardFunction` queried with a
tidy long frame, so it composes with `plot_cumulative_hazard` the same way KM composes with
`plot_survival`. Delayed entry flows through.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from lifelines import NelsonAalenFitter

from tenure._frame import as_estimator_frame
from tenure.estimators.kaplan_meier import _group_labels

_QUERY_COLUMNS = ["group", "time", "cumulative_hazard", "ci_lower", "ci_upper"]


@dataclass(frozen=True)
class HazardCurve:
    """A single group's right-continuous Nelson-Aalen step curve, anchored at (t=0, H=0)."""

    times: np.ndarray
    cumulative_hazard: np.ndarray
    ci_lower: np.ndarray
    ci_upper: np.ndarray

    def at(self, t: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        idx = np.searchsorted(self.times, t, side="right") - 1
        idx = np.clip(idx, 0, len(self.times) - 1)
        return self.cumulative_hazard[idx], self.ci_lower[idx], self.ci_upper[idx]


class CumulativeHazardFunction:
    """Per-group cumulative-hazard curves with a backend-neutral tidy query API."""

    def __init__(self, curves: dict[str, HazardCurve], time_unit: str = "day") -> None:
        self._curves = dict(curves)
        self.time_unit = time_unit

    @property
    def groups(self) -> list[str]:
        return list(self._curves)

    def curve(self, group: str) -> HazardCurve:
        if group not in self._curves:
            raise KeyError(f"Unknown group {group!r}; known groups: {self.groups}")
        return self._curves[group]

    def cumulative_hazard_at(self, times, group=None) -> pd.DataFrame:
        t = np.atleast_1d(np.asarray(times, dtype=float))
        selection = self._curves.items() if group is None else [(group, self.curve(group))]
        frames = []
        for label, curve in selection:
            hazard, ci_lower, ci_upper = curve.at(t)
            frames.append(
                pd.DataFrame(
                    {
                        "group": label,
                        "time": t,
                        "cumulative_hazard": hazard,
                        "ci_lower": ci_lower,
                        "ci_upper": ci_upper,
                    }
                )
            )
        return pd.concat(frames, ignore_index=True)[_QUERY_COLUMNS]


class NelsonAalen:
    """Fit per-group Nelson-Aalen cumulative-hazard curves."""

    def __init__(self, alpha: float = 0.05) -> None:
        self.alpha = alpha
        self._hazard: CumulativeHazardFunction | None = None

    def fit(self, data, *, by=None) -> NelsonAalen:
        table = data.derive() if hasattr(data, "derive") else data
        time_unit = getattr(data, "time_unit", "day")
        labels, order = _group_labels(table, by)
        curves: dict[str, HazardCurve] = {}
        for label in order:
            mask = (labels == label).to_numpy()
            curves[label] = self._fit_one(as_estimator_frame(table.loc[mask]))
        self._hazard = CumulativeHazardFunction(curves, time_unit=time_unit)
        return self

    def _fit_one(self, ef) -> HazardCurve:
        naf = NelsonAalenFitter(alpha=self.alpha)
        naf.fit(durations=ef.duration, event_observed=ef.event, entry=ef.entry)
        hazard_frame = naf.cumulative_hazard_
        ci = naf.confidence_interval_
        times = hazard_frame.index.to_numpy(dtype=float)
        hazard = hazard_frame.iloc[:, 0].to_numpy(dtype=float)
        ci_lower = ci.iloc[:, 0].to_numpy(dtype=float)
        ci_upper = ci.iloc[:, 1].to_numpy(dtype=float)
        if times[0] > 0.0:
            times = np.insert(times, 0, 0.0)
            hazard = np.insert(hazard, 0, 0.0)
            ci_lower = np.insert(ci_lower, 0, 0.0)
            ci_upper = np.insert(ci_upper, 0, 0.0)
        return HazardCurve(times, hazard, ci_lower, ci_upper)

    def _require_fitted(self) -> CumulativeHazardFunction:
        if self._hazard is None:
            raise RuntimeError("NelsonAalen is not fitted yet; call .fit(...) first.")
        return self._hazard

    @property
    def cumulative_hazard_(self) -> CumulativeHazardFunction:
        return self._require_fitted()

    def cumulative_hazard_at(self, times, group=None) -> pd.DataFrame:
        return self._require_fitted().cumulative_hazard_at(times, group=group)
