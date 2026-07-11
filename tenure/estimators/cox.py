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

from tenure._frame import ENTRY, EVENT, EXIT, as_estimator_frame, ensure_estimable
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
    """Cox PH estimator. Fit a design with covariates, then ``predict_survival(profiles)``.

    ``strata`` names categorical covariates to stratify on instead of estimating a coefficient:
    each stratum gets its own baseline hazard while the remaining covariates share one set of
    coefficients. This is the standard remedy when ``proportional_hazards_test`` flags a
    covariate -- refit the SAME design with ``CoxPH(strata=["plan"])`` and the offending
    covariate leaves the PH assumption (and the test) entirely. Strata must be a subset of the
    design's categorical ``covariate_cols``; at least one covariate must remain unstratified.
    """

    def __init__(self, alpha: float = 0.05, penalizer: float = 0.0, strata=None) -> None:
        self.alpha = alpha
        self.penalizer = penalizer
        self.strata = [strata] if isinstance(strata, str) else list(strata or [])
        self._fitter: CoxPHFitter | None = None
        self._design = None
        self._support: tuple | None = None
        self._training_frame: pd.DataFrame | None = None

    def _validate_strata(self, design) -> None:
        mappings = design.covariate_mappings
        for col in self.strata:
            if col not in mappings:
                raise TenureValidationError(
                    f"Stratum {col!r} is not a covariate_col on the design (got "
                    f"{list(mappings)}). Strata are drawn from covariate_cols so the "
                    "detect-then-stratify remedy reuses the same StudyDesign."
                )
            if mappings[col]["kind"] != "categorical":
                raise TenureValidationError(
                    f"Stratum {col!r} is numeric; stratification needs discrete levels. "
                    "Bin it into a categorical column first if you mean to stratify on it."
                )
        if len(self.strata) >= len(mappings):
            raise TenureValidationError(
                "All covariates are stratified away; at least one must remain to estimate "
                "coefficients. Drop a stratum or add a covariate."
            )

    def _encode_with_strata(self, raw: pd.DataFrame) -> pd.DataFrame:
        """Encode covariates, replacing each stratum's one-hot columns with its raw labels.

        ``encode_covariates`` runs first on everything so its unknown-level guard still protects
        strata; the stratum's dummy columns are then swapped for the raw column lifelines'
        ``strata=`` machinery consumes.
        """
        encoded = self._design.encode_covariates(raw)
        mappings = self._design.covariate_mappings
        for col in self.strata:
            dummies = [f"{col}_{level}" for level in mappings[col]["levels"][1:]]
            encoded = encoded.drop(columns=[d for d in dummies if d in encoded.columns])
            encoded[col] = raw[col].astype(str).to_numpy()
        return encoded

    def fit(self, design) -> CoxPH:
        ensure_estimable(design)
        if not getattr(design, "covariate_cols", None):
            raise TenureValidationError(
                "CoxPH requires covariate_cols on the StudyDesign "
                "(build it with covariate_cols=[...])."
            )
        self._design = design
        if self.strata:
            self._validate_strata(design)
        table = design.derive()
        frame = self._encode_with_strata(table) if self.strata else design.encode_covariates(table)
        frame[_DURATION] = table[EXIT].to_numpy(dtype=float)
        frame[_EVENT] = table[EVENT].to_numpy(dtype=int)
        frame[_ENTRY] = table[ENTRY].to_numpy(dtype=float)

        fitter = CoxPHFitter(alpha=self.alpha, penalizer=self.penalizer)
        fitter.fit(
            frame,
            duration_col=_DURATION,
            event_col=_EVENT,
            entry_col=_ENTRY,
            strata=self.strata or None,
        )
        self._fitter = fitter
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
        """Encode a design's covariates, aligned to the fitted model's columns (missing -> 0).

        For a stratified model the strata columns ride along as raw labels -- lifelines needs
        them to pick each row's baseline hazard.
        """
        self._require_fitted()
        table = design.derive()
        encoded = design.encode_covariates(table)
        aligned = encoded.reindex(columns=self._fitter.params_.index, fill_value=0.0)
        for col in self.strata:
            aligned[col] = table[col].astype(str).to_numpy()
        return aligned

    def predict_survival(self, profiles) -> SurvivalFunction:
        """Predicted survival per covariate profile (raw labels) as a SurvivalFunction.

        ``profiles`` is a DataFrame of raw covariate values (one row per profile). Curves are
        labeled by the frame's index, or by a stringified row when the index is a default range.
        For a stratified model each profile's curve uses its own stratum's baseline hazard (the
        support arrays remain the pooled training cohort's, as for the unstratified model).
        """
        self._require_fitted()
        profiles = self._as_frame(profiles)
        missing = [c for c in self.strata if c not in profiles.columns]
        if missing:
            raise TenureValidationError(
                f"Stratified model: profiles must include the strata column(s) {missing} so each "
                "row selects its baseline hazard."
            )
        encoded = (
            self._encode_with_strata(profiles)
            if self.strata
            else self._design.encode_covariates(profiles)
        )
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

        On a stratified model the test covers only the estimated (non-strata) covariates:
        a stratified covariate has no coefficient and no PH assumption to violate -- which is
        exactly why stratifying is the remedy when this test flags it.
        """
        self._require_fitted()
        frame = self._training_frame.drop(columns=[_ENTRY])
        fitter = CoxPHFitter(penalizer=self.penalizer)
        fitter.fit(frame, duration_col=_DURATION, event_col=_EVENT, strata=self.strata or None)
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
            raw = sorted({self._raw_covariate(v) for v in report.violations})
            categorical = [
                c for c in raw if self._design.covariate_mappings[c]["kind"] == "categorical"
            ]
            remedy = (
                f"refit stratified on the offending categorical covariate "
                f"(CoxPH(strata={categorical!r})) or move to a time-varying model"
                if categorical
                else "move to a time-varying model (or bin and stratify the numeric covariate)"
            )
            warnings.warn(
                f"Proportional-hazards assumption may be violated for {report.violations} "
                f"(Schoenfeld p < {threshold}). Remedies: {remedy}.",
                stacklevel=2,
            )
        return report

    def _raw_covariate(self, encoded_name: str) -> str:
        """Map an encoded column (e.g. ``plan_premium``) back to its raw covariate (``plan``)."""
        for col, mapping in self._design.covariate_mappings.items():
            if encoded_name == col:
                return col
            if mapping["kind"] == "categorical" and any(
                encoded_name == f"{col}_{level}" for level in mapping["levels"][1:]
            ):
                return col
        return encoded_name

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
