"""Cox Proportional Hazards (wraps lifelines) producing curves via the SurvivalFunction interface.

Fit on a StudyDesign carrying ``covariate_cols``; predict survival at covariate profiles. The
predicted curves are returned as a `SurvivalFunction`, so the v0.1 business-output and plotting
layers consume Cox exactly like Kaplan-Meier (A3/A8) -- no rework. Delayed entry flows through.

Cox curves carry point estimates only (no CI band) in v0.2; the GroupCurve CI bounds are set to
the point estimate. Their support (last event time / at-risk) is taken from the training cohort,
so RMST/LTV truncate-and-relabel where the training data thins out.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import proportional_hazard_test

from tenure._frame import ENTRY, EVENT, EXIT, as_estimator_frame
from tenure.estimators.survival import GroupCurve, SurvivalFunction
from tenure.exceptions import TenureValidationError

_DURATION = "__duration__"
_EVENT = "__event__"
_ENTRY = "__entry__"


@dataclass
class CoxDiagnosticReport:
    """Proportional-hazards diagnostic: a tidy per-covariate table plus pass/fail helpers."""

    table: pd.DataFrame  # columns: covariate, test_statistic, p_value, status
    threshold: float = 0.05

    @property
    def ok(self) -> bool:
        return not (self.table["status"] == "fail").any()

    @property
    def violations(self) -> list[str]:
        return self.table.loc[self.table["status"] == "fail", "covariate"].tolist()

    def __repr__(self) -> str:
        return (
            f"CoxDiagnosticReport(ok={self.ok}, violations={self.violations}, "
            f"threshold={self.threshold})"
        )


class CoxPH:
    """Cox PH estimator. Fit a design with covariates, then ``predict_survival(profiles)``."""

    def __init__(self, alpha: float = 0.05, penalizer: float = 0.0) -> None:
        self.alpha = alpha
        self.penalizer = penalizer
        self._fitter: CoxPHFitter | None = None
        self._design = None
        self._support: tuple | None = None
        self._training_frame: pd.DataFrame | None = None

    def fit(self, design) -> CoxPH:
        if not getattr(design, "covariate_cols", None):
            raise TenureValidationError(
                "CoxPH requires covariate_cols on the StudyDesign "
                "(build it with covariate_cols=[...])."
            )
        table = design.derive()
        frame = design.encode_covariates(table)
        frame[_DURATION] = table[EXIT].to_numpy(dtype=float)
        frame[_EVENT] = table[EVENT].to_numpy(dtype=int)
        frame[_ENTRY] = table[ENTRY].to_numpy(dtype=float)

        fitter = CoxPHFitter(alpha=self.alpha, penalizer=self.penalizer)
        fitter.fit(frame, duration_col=_DURATION, event_col=_EVENT, entry_col=_ENTRY)
        self._fitter = fitter
        self._design = design
        self._support = self._training_support(table)
        self._training_frame = frame
        return self

    @staticmethod
    def _training_support(table: pd.DataFrame) -> tuple:
        ef = as_estimator_frame(table)
        kmf = KaplanMeierFitter().fit(
            durations=ef.duration, event_observed=ef.event, entry=ef.entry
        )
        event_table = kmf.event_table
        risk_times = event_table.index.to_numpy(dtype=float)
        n_at_risk = event_table["at_risk"].to_numpy(dtype=float)
        events = risk_times[event_table["observed"].to_numpy(dtype=float) > 0]
        last_event_time = float(events.max()) if events.size else 0.0
        return risk_times, n_at_risk, last_event_time

    def _require_fitted(self) -> None:
        if self._fitter is None:
            raise RuntimeError("CoxPH is not fitted yet; call .fit(design) first.")

    @property
    def fitter(self) -> CoxPHFitter:
        """The underlying fitted lifelines CoxPHFitter (coefficients, summary, etc.)."""
        self._require_fitted()
        return self._fitter

    @property
    def design(self):
        """The StudyDesign this model was fit on."""
        self._require_fitted()
        return self._design

    def encode_for_prediction(self, design) -> pd.DataFrame:
        """Encode a design's covariates, aligned to the fitted model's columns (missing -> 0)."""
        self._require_fitted()
        encoded = design.encode_covariates(design.derive())
        return encoded.reindex(columns=self._fitter.params_.index, fill_value=0.0)

    def predict_survival(self, profiles) -> SurvivalFunction:
        """Predicted survival per covariate profile (raw labels) as a SurvivalFunction.

        ``profiles`` is a DataFrame of raw covariate values (one row per profile). Curves are
        labeled by the frame's index, or by a stringified row when the index is a default range.
        """
        self._require_fitted()
        profiles = self._as_frame(profiles)
        encoded = self._design.encode_covariates(profiles)
        predicted = self._fitter.predict_survival_function(encoded)
        labels = self._profile_labels(profiles)
        risk_times, n_at_risk, last_event_time = self._support

        curves: dict[str, GroupCurve] = {}
        for i, label in enumerate(labels):
            times = predicted.index.to_numpy(dtype=float)
            survival = predicted.iloc[:, i].to_numpy(dtype=float)
            if times[0] > 0.0:
                times = np.insert(times, 0, 0.0)
                survival = np.insert(survival, 0, 1.0)
            curves[label] = GroupCurve(
                times=times,
                survival=survival,
                ci_lower=survival.copy(),
                ci_upper=survival.copy(),
                median=self._median(times, survival),
                risk_times=risk_times,
                n_at_risk=n_at_risk,
                last_event_time=last_event_time,
            )
        return SurvivalFunction(curves, time_unit=self._design.time_unit)

    def profile_grid(self, vary: str) -> pd.DataFrame:
        """Profiles varying one categorical covariate over its levels, others at reference/mean.

        Index = the varied levels, so ``predict_survival(profile_grid("plan"))`` yields one curve
        per plan level -- a Cox analogue of grouped Kaplan-Meier comparison.
        """
        self._require_fitted()
        mappings = self._design.covariate_mappings
        if vary not in mappings:
            raise TenureValidationError(f"{vary!r} is not a covariate_col; got {list(mappings)}.")
        if mappings[vary]["kind"] != "categorical":
            raise TenureValidationError(
                f"profile_grid varies categorical covariates only; {vary!r} is numeric."
            )
        table = self._design.derive()
        base = {
            col: (
                float(pd.to_numeric(table[col]).mean()) if m["kind"] == "numeric" else m["baseline"]
            )
            for col, m in mappings.items()
        }
        levels = mappings[vary]["levels"]
        rows = [{**base, vary: level} for level in levels]
        return pd.DataFrame(rows, index=[str(level) for level in levels])

    def proportional_hazards_test(
        self, *, time_transform: str = "rank", threshold: float = 0.05, warn: bool = True
    ) -> CoxDiagnosticReport:
        """Schoenfeld-residual test of the PH assumption per covariate (matches lifelines).

        Returns a CoxDiagnosticReport (status pass/fail at ``threshold``) and emits a warning when
        any covariate fails (unless ``warn=False``). A fitted-model diagnostic, not a study-design
        audit check (DV2-4).

        Note: lifelines cannot compute Schoenfeld residuals for left-truncated (entry) fits, so the
        test refits without the entry column -- identical when there is no delayed entry, a close
        approximation otherwise (the only way lifelines exposes the test).
        """
        self._require_fitted()
        frame = self._training_frame.drop(columns=[_ENTRY])
        fitter = CoxPHFitter(penalizer=self.penalizer)
        fitter.fit(frame, duration_col=_DURATION, event_col=_EVENT)
        result = proportional_hazard_test(fitter, frame, time_transform=time_transform)
        summary = result.summary
        covariates = [str(i[0]) if isinstance(i, tuple) else str(i) for i in summary.index]
        p_value = summary["p"].to_numpy(dtype=float)
        table = pd.DataFrame(
            {
                "covariate": covariates,
                "test_statistic": summary["test_statistic"].to_numpy(dtype=float),
                "p_value": p_value,
                "status": np.where(p_value < threshold, "fail", "pass"),
            }
        )
        report = CoxDiagnosticReport(table=table, threshold=threshold)
        if warn and not report.ok:
            warnings.warn(
                f"Proportional-hazards assumption may be violated for {report.violations} "
                f"(Schoenfeld p < {threshold}). Consider a stratified Cox or a time-varying "
                "model (v0.3).",
                stacklevel=2,
            )
        return report

    @staticmethod
    def _as_frame(profiles) -> pd.DataFrame:
        if isinstance(profiles, pd.DataFrame):
            return profiles
        if isinstance(profiles, dict):
            return pd.DataFrame([profiles])
        return pd.DataFrame(profiles)

    def _profile_labels(self, profiles: pd.DataFrame) -> list[str]:
        if isinstance(profiles.index, pd.RangeIndex):
            cov = self._design.covariate_cols
            return [
                "|".join(f"{c}={profiles.iloc[i][c]}" for c in cov) for i in range(len(profiles))
            ]
        return [str(x) for x in profiles.index]

    @staticmethod
    def _median(times: np.ndarray, survival: np.ndarray) -> float:
        below = np.where(survival <= 0.5)[0]
        return float(times[below[0]]) if below.size else float("inf")
