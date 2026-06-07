"""The multi-group survival function -- the interface the business layer consumes (A3/A6).

Any estimator that can produce per-group step curves (Kaplan-Meier now; Cox, parametric, or
time-varying later) builds one of these. Queries return tidy long frames keyed by a string
``group`` label, never pandas-index-keyed objects (D-S1).

A ``GroupCurve`` also carries the support information the business layer needs -- the last
observed event time and the number-at-risk over time -- so RMST/LTV can compute a per-group
effective horizon (FR-BO-2, TNR005) and integrate the curve exactly.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

_QUERY_COLUMNS = ["group", "time", "survival", "ci_lower", "ci_upper"]


@dataclass(frozen=True)
class GroupCurve:
    """A single group's right-continuous KM step curve, anchored at (t=0, S=1)."""

    times: np.ndarray  # ascending jump points; times[0] == 0.0
    survival: np.ndarray
    ci_lower: np.ndarray
    ci_upper: np.ndarray
    median: float  # may be inf when the median survival time is not reached
    # Support information (from the estimator's risk/event table):
    risk_times: np.ndarray = field(default_factory=lambda: np.array([0.0]))
    n_at_risk: np.ndarray = field(default_factory=lambda: np.array([0.0]))
    last_event_time: float = 0.0

    def at(self, t: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Step lookup of survival (+ CI): value at the last jump time <= t (right-continuous)."""
        idx = np.searchsorted(self.times, t, side="right") - 1
        idx = np.clip(idx, 0, len(self.times) - 1)
        return self.survival[idx], self.ci_lower[idx], self.ci_upper[idx]

    def n_at_risk_at(self, t) -> np.ndarray:
        """Number at risk at tenure ``t`` (the conventional at-risk-table convention)."""
        t = np.atleast_1d(np.asarray(t, dtype=float))
        idx = np.searchsorted(self.risk_times, t, side="left")
        in_range = idx < len(self.risk_times)
        safe = np.clip(idx, 0, len(self.risk_times) - 1)
        return np.where(in_range, self.n_at_risk[safe], 0.0)

    def integral(self, a: float, b: float) -> float:
        """Exact integral of the step survival function over [a, b] (area under S)."""
        if b <= a:
            return 0.0
        times = self.times
        surv = self.survival
        n = len(times)
        total = 0.0
        for i in range(n):
            seg_start = times[i]
            seg_end = times[i + 1] if i + 1 < n else np.inf
            lo = max(a, seg_start)
            hi = min(b, seg_end)
            if hi > lo:
                total += surv[i] * (hi - lo)
            if seg_end >= b:
                break
        return float(total)

    def effective_horizon(self, requested: float, min_at_risk: int) -> float:
        """Largest horizon with adequate support (FR-BO-2): min of the requested horizon, the
        last observed event time, and the last tenure where at-risk >= ``min_at_risk``.

        If the whole cohort never reaches ``min_at_risk`` (small sample), the at-risk criterion
        does not bind -- only the last event time caps the horizon.
        """
        h = min(float(requested), float(self.last_event_time))
        if len(self.n_at_risk) and self.n_at_risk.max() >= min_at_risk:
            ok = self.risk_times[self.n_at_risk >= min_at_risk]
            h = min(h, float(ok.max())) if len(ok) else 0.0
        return max(h, 0.0)


class SurvivalFunction:
    """A collection of per-group survival curves with a backend-neutral query API."""

    def __init__(self, curves: dict[str, GroupCurve], time_unit: str = "day") -> None:
        self._curves = dict(curves)
        self.time_unit = time_unit

    @property
    def groups(self) -> list[str]:
        return list(self._curves)

    def curve(self, group: str) -> GroupCurve:
        """Access a single group's curve (used by the business-output layer)."""
        if group not in self._curves:
            raise KeyError(f"Unknown group {group!r}; known groups: {self.groups}")
        return self._curves[group]

    def _select(self, group) -> list[tuple[str, GroupCurve]]:
        if group is None:
            return list(self._curves.items())
        return [(group, self.curve(group))]

    def survival_at(self, times, group=None) -> pd.DataFrame:
        """Survival (+ CI) at ``times`` as a tidy frame [group, time, survival, ci_lower, ci_upper].

        ``group=None`` returns every fitted group; ``group=<label>`` filters to one.
        """
        t = np.atleast_1d(np.asarray(times, dtype=float))
        frames = []
        for label, curve in self._select(group):
            survival, ci_lower, ci_upper = curve.at(t)
            frames.append(
                pd.DataFrame(
                    {
                        "group": label,
                        "time": t,
                        "survival": survival,
                        "ci_lower": ci_lower,
                        "ci_upper": ci_upper,
                    }
                )
            )
        return pd.concat(frames, ignore_index=True)[_QUERY_COLUMNS]

    def median_survival(self, group=None) -> pd.DataFrame:
        """Median survival time per group as a tidy frame [group, median]."""
        rows = [{"group": label, "median": curve.median} for label, curve in self._select(group)]
        return pd.DataFrame(rows, columns=["group", "median"])
