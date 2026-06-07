"""RetentionStudy: the high-level guided workflow (Persona B hero path).

`RetentionStudy.from_*().run()` builds a StudyDesign, runs the audit FIRST (raising under
``strictness='block'`` before any number is computed; warning under ``'warn'`` and proceeding),
fits Kaplan-Meier, and bundles everything into a `RetentionResult`. The workflow calls the same
primitives it exposes, so its numbers are identical to the low-level path (NFR-API-1).
"""

from __future__ import annotations

import warnings

from tenure.audit import audit
from tenure.estimators import KaplanMeier
from tenure.outputs import retention_at, rmst, summarize, survival_weighted_ltv
from tenure.outputs._common import DEFAULT_HORIZONS, DEFAULT_MIN_AT_RISK
from tenure.study_design import StudyDesign


class RetentionResult:
    """The bundled outcome of a RetentionStudy: the audit, the curves, and output helpers.

    Numbers are never produced without the audit being available on ``.audit`` (and, under
    ``strictness='warn'``, surfaced via a Python warning at run time).
    """

    def __init__(self, *, audit_report, estimator) -> None:
        self._audit = audit_report
        self._estimator = estimator

    @property
    def audit(self):
        return self._audit

    @property
    def curves(self):
        """The fitted multi-group SurvivalFunction."""
        return self._estimator.survival_

    def retention(self, horizons=DEFAULT_HORIZONS, *, min_at_risk: int = DEFAULT_MIN_AT_RISK):
        return retention_at(self._estimator, horizons, min_at_risk=min_at_risk)

    def rmst(self, *, horizon, min_at_risk: int = DEFAULT_MIN_AT_RISK):
        return rmst(self._estimator, horizon=horizon, min_at_risk=min_at_risk)

    def ltv(
        self,
        *,
        period_margin,
        horizon,
        discount_rate: float = 0.0,
        period: str = "month",
        min_at_risk: int = DEFAULT_MIN_AT_RISK,
    ):
        return survival_weighted_ltv(
            self._estimator,
            period_margin=period_margin,
            horizon=horizon,
            discount_rate=discount_rate,
            period=period,
            min_at_risk=min_at_risk,
        )

    def summary(self, *, period_margin, ltv_horizon, **kwargs):
        """A SummaryReport, with this run's audit verdict attached for provenance (FR-BO-5)."""
        return summarize(
            self._estimator,
            period_margin=period_margin,
            ltv_horizon=ltv_horizon,
            audit_report=self._audit,
            **kwargs,
        )

    def plot(self, **kwargs):
        raise NotImplementedError(
            "RetentionResult.plot lands in the plotting slice (v0.1 Slice 7); "
            "use .curves / .summary(...) for now."
        )

    def __repr__(self) -> str:
        return f"RetentionResult(groups={self.curves.groups}, audit={self._audit!r})"


class RetentionStudy:
    """A guided retention study: design -> audit -> Kaplan-Meier -> bundled result."""

    def __init__(self, design: StudyDesign, *, by, strictness: str) -> None:
        self._design = design
        self._by = by
        self._strictness = strictness

    @classmethod
    def from_event_dates(
        cls, df, *, strictness: str = "block", by=None, group_cols=None, **design_kwargs
    ) -> RetentionStudy:
        """Configure a study from the event-date schema (see StudyDesign.from_event_dates)."""
        design = StudyDesign.from_event_dates(df, group_cols=group_cols, **design_kwargs)
        return cls(design, by=by if by is not None else (group_cols or None), strictness=strictness)

    @classmethod
    def from_status(
        cls, df, *, strictness: str = "block", by=None, group_cols=None, **design_kwargs
    ) -> RetentionStudy:
        """Configure a study from the status-label schema (see StudyDesign.from_status)."""
        design = StudyDesign.from_status(df, group_cols=group_cols, **design_kwargs)
        return cls(design, by=by if by is not None else (group_cols or None), strictness=strictness)

    @property
    def design(self) -> StudyDesign:
        return self._design

    def run(self) -> RetentionResult:
        """Audit first (raise/warn per strictness), then fit the curves and bundle the result."""
        report = audit(self._design, strictness=self._strictness)  # raises on a blocking design
        if report.warnings:
            names = "; ".join(f"{r.check_id}: {r.title}" for r in report.warnings)
            warnings.warn(
                f"Study-design audit raised {len(report.warnings)} warning(s) -- {names}. "
                "Numbers were still computed; inspect result.audit.",
                stacklevel=2,
            )
        estimator = KaplanMeier().fit(self._design, by=self._by)
        return RetentionResult(audit_report=report, estimator=estimator)
