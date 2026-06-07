"""Kaplan-Meier estimator (wraps lifelines; AD-2) producing a multi-group SurvivalFunction.

Delayed entry (left truncation) flows through from the canonical table's ``entry_tenure``,
so curves are correct for an existing customer base, not just clean acquisition cohorts.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter

from tenure._frame import as_estimator_frame, ensure_estimable
from tenure.estimators.survival import GroupCurve, SurvivalFunction
from tenure.exceptions import TenureValidationError


def _group_labels(table: pd.DataFrame, by) -> tuple[pd.Series, list[str]]:
    """Return a per-row string group label and the groups in first-seen order."""
    if by is None:
        return pd.Series("overall", index=table.index), ["overall"]

    cols = [by] if isinstance(by, str) else list(by)
    missing = [c for c in cols if c not in table.columns]
    if missing:
        raise TenureValidationError(
            f"by column(s) not in the design's table: {missing}. "
            "Declare them via group_cols when building the StudyDesign."
        )

    if len(cols) == 1:
        labels = table[cols[0]].astype(str)
    else:
        labels = cols[0] + "=" + table[cols[0]].astype(str)
        for col in cols[1:]:
            labels = labels + "|" + col + "=" + table[col].astype(str)

    return labels, list(pd.unique(labels))


class KaplanMeier:
    """Fit per-group Kaplan-Meier curves and expose them as a SurvivalFunction.

    ``data`` may be a :class:`~tenure.study_design.StudyDesign` or its derived canonical
    table. ``by`` selects grouping column(s) (which must be present on the table, i.e.
    declared via ``group_cols``); ``by=None`` fits a single curve labeled ``"overall"``.
    """

    def __init__(self, alpha: float = 0.05) -> None:
        self.alpha = alpha
        self._survival: SurvivalFunction | None = None

    def fit(self, data, *, by=None) -> KaplanMeier:
        ensure_estimable(data)
        table = data.derive() if hasattr(data, "derive") else data
        time_unit = getattr(data, "time_unit", "day")
        labels, order = _group_labels(table, by)
        curves: dict[str, GroupCurve] = {}
        for label in order:
            mask = (labels == label).to_numpy()
            curves[label] = self._fit_one(as_estimator_frame(table.loc[mask]))
        self._survival = SurvivalFunction(curves, time_unit=time_unit)
        return self

    def _fit_one(self, ef) -> GroupCurve:
        kmf = KaplanMeierFitter(alpha=self.alpha)
        kmf.fit(durations=ef.duration, event_observed=ef.event, entry=ef.entry)

        sf = kmf.survival_function_
        ci = kmf.confidence_interval_
        times = sf.index.to_numpy(dtype=float)
        survival = sf.iloc[:, 0].to_numpy(dtype=float)
        ci_lower = ci.iloc[:, 0].to_numpy(dtype=float)
        ci_upper = ci.iloc[:, 1].to_numpy(dtype=float)

        # Anchor at (t=0, S=1) so queries before the first event return 1.0.
        if times[0] > 0.0:
            times = np.insert(times, 0, 0.0)
            survival = np.insert(survival, 0, 1.0)
            ci_lower = np.insert(ci_lower, 0, 1.0)
            ci_upper = np.insert(ci_upper, 0, 1.0)

        # Support information from the risk/event table.
        event_table = kmf.event_table
        risk_times = event_table.index.to_numpy(dtype=float)
        n_at_risk = event_table["at_risk"].to_numpy(dtype=float)
        observed = event_table["observed"].to_numpy(dtype=float)
        events = risk_times[observed > 0]
        last_event_time = float(events.max()) if events.size else 0.0

        return GroupCurve(
            times=times,
            survival=survival,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            median=float(kmf.median_survival_time_),
            risk_times=risk_times,
            n_at_risk=n_at_risk,
            last_event_time=last_event_time,
        )

    def _require_fitted(self) -> SurvivalFunction:
        if self._survival is None:
            raise RuntimeError("KaplanMeier is not fitted yet; call .fit(...) first.")
        return self._survival

    @property
    def survival_(self) -> SurvivalFunction:
        """The fitted multi-group SurvivalFunction (what the business layer consumes)."""
        return self._require_fitted()

    def survival_at(self, times, group=None) -> pd.DataFrame:
        return self._require_fitted().survival_at(times, group=group)

    def median_survival(self, group=None) -> pd.DataFrame:
        return self._require_fitted().median_survival(group=group)
