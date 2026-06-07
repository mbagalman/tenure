"""The multi-group survival function -- the interface the business layer consumes (A3/A6).

Any estimator that can produce per-group step curves (Kaplan-Meier now; Cox, parametric, or
time-varying later) builds one of these. Queries return tidy long frames keyed by a string
``group`` label, never pandas-index-keyed objects (D-S1).
"""

from __future__ import annotations

from dataclasses import dataclass

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

    def at(self, t: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Step lookup: value at the last jump time <= t (right-continuous KM)."""
        idx = np.searchsorted(self.times, t, side="right") - 1
        idx = np.clip(idx, 0, len(self.times) - 1)
        return self.survival[idx], self.ci_lower[idx], self.ci_upper[idx]


class SurvivalFunction:
    """A collection of per-group survival curves with a backend-neutral query API."""

    def __init__(self, curves: dict[str, GroupCurve]) -> None:
        self._curves = dict(curves)

    @property
    def groups(self) -> list[str]:
        return list(self._curves)

    def _select(self, group) -> list[tuple[str, GroupCurve]]:
        if group is None:
            return list(self._curves.items())
        if group not in self._curves:
            raise KeyError(f"Unknown group {group!r}; known groups: {self.groups}")
        return [(group, self._curves[group])]

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
