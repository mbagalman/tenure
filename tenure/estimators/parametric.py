"""Parametric survival models (wraps lifelines) that can extrapolate past observed support.

Kaplan-Meier is non-parametric: it says nothing beyond the last observed time, so Tenure's RMST/LTV
truncate-and-relabel there (never a silent flat tail). A *parametric* fit is different -- a fitted
distribution is defined at every tenure, so it gives a principled projection beyond the data window
(e.g. a 3-year LTV from 1 year of history) and a readable hazard shape (Weibull ``rho > 1`` =>
rising churn hazard, ``< 1`` => falling, ``== 1`` => memoryless/exponential).

This estimator presents the SAME multi-group ``SurvivalFunction`` interface as Kaplan-Meier (A3), so
``retention_at`` / ``rmst`` / ``survival_weighted_ltv`` / ``plot_survival`` consume it unchanged --
the only difference is that its ``effective_horizon`` is the *requested* horizon (the model
extrapolates by design) rather than the KM support cap.

Because extrapolation IS the point, the caller is opting into model-based projection: the curve
records ``last_event_time`` (the data/model boundary) so downstream code and docs can be explicit
about where empirical evidence ends and the model takes over. Curves carry point estimates only
(no CI band) in this release, matching Cox.

The survival functions are evaluated in closed form from the fitted parameters (backend-neutral
floats, not a live fitter handle) and reference-matched to lifelines in the tests.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from lifelines import (
    ExponentialFitter,
    LogLogisticFitter,
    LogNormalFitter,
    WeibullFitter,
)
from scipy.integrate import quad
from scipy.stats import norm

from tenure._frame import as_estimator_frame, ensure_estimable
from tenure.estimators.kaplan_meier import _group_labels
from tenure.estimators.survival import GroupCurve, SurvivalFunction
from tenure.exceptions import TenureValidationError

# distribution -> (lifelines fitter class, ordered lifelines attribute names for the parameters)
_DISTRIBUTIONS: dict[str, tuple] = {
    "weibull": (WeibullFitter, ("lambda_", "rho_")),
    "exponential": (ExponentialFitter, ("lambda_",)),
    "lognormal": (LogNormalFitter, ("mu_", "sigma_")),
    "loglogistic": (LogLogisticFitter, ("alpha_", "beta_")),
}
# human-facing parameter names (parallel to the lifelines attributes above).
_PARAM_NAMES: dict[str, tuple] = {
    "weibull": ("scale", "shape"),
    "exponential": ("scale",),
    "lognormal": ("mu", "sigma"),
    "loglogistic": ("scale", "shape"),
}


def _survival_fn(distribution: str, params: tuple, t) -> np.ndarray:
    """Closed-form survival S(t) for a fitted distribution (S(0)=1). Vectorized over ``t``.

    Parameterizations match lifelines' fitters (pinned by reference-match tests):
    weibull ``exp(-(t/scale)**shape)``; exponential ``exp(-t/scale)``;
    lognormal ``1 - Phi((ln t - mu)/sigma)``; loglogistic ``1/(1 + (t/scale)**shape)``.
    """
    t = np.asarray(t, dtype=float)
    positive = t > 0
    is_nan = np.isnan(t)  # NaN > 0 is False, so without this a NaN query would coerce to S=1.0
    safe = np.where(positive, t, 1.0)  # avoid log(0) / 0**negative in the off-support branch
    if distribution == "weibull":
        scale, shape = params
        s = np.exp(-np.power(safe / scale, shape))
    elif distribution == "exponential":
        (scale,) = params
        s = np.exp(-safe / scale)
    elif distribution == "lognormal":
        mu, sigma = params
        s = norm.sf((np.log(safe) - mu) / sigma)
    elif distribution == "loglogistic":
        scale, shape = params
        s = 1.0 / (1.0 + np.power(safe / scale, shape))
    else:  # pragma: no cover - guarded at construction
        raise TenureValidationError(f"Unknown distribution {distribution!r}.")
    return np.where(is_nan, np.nan, np.where(positive, s, 1.0))  # propagate NaN (review fix)


def _empirical_at_risk(ef) -> tuple[np.ndarray, np.ndarray]:
    """Number at risk (entry < t <= exit) at each distinct exit time -- for the at-risk table."""
    times = np.unique(ef.duration)
    n = np.array([np.sum((ef.entry < t) & (ef.duration >= t)) for t in times], dtype=float)
    return times, n


@dataclass(frozen=True)
class ParametricGroupCurve(GroupCurve):
    """A single group's fitted parametric survival curve.

    Overrides ``at`` / ``integral`` / ``effective_horizon`` to evaluate the fitted distribution
    analytically, so queries are exact and extrapolate past the data. The inherited ``times`` /
    ``survival`` arrays are a dense grid over the observed range, used only for plotting.
    """

    distribution: str = "weibull"
    params: tuple = ()

    def at(self, t):  # noqa: D102 -- inherited contract
        s = _survival_fn(self.distribution, self.params, np.atleast_1d(np.asarray(t, dtype=float)))
        return s, s, s  # CI == point estimate (no band in this release)

    def integral(self, a: float, b: float) -> float:
        """Exact area under the parametric S(t) over [a, b] via adaptive quadrature."""
        if b <= a:
            return 0.0
        value, _ = quad(
            lambda x: float(_survival_fn(self.distribution, self.params, x)), a, b, limit=200
        )
        return float(value)

    def effective_horizon(self, requested: float, min_at_risk: int) -> float:
        """The full requested horizon: a parametric model is defined everywhere, so it extrapolates
        by design (unlike KM's truncate-and-relabel). ``last_event_time`` marks where data ended.
        """
        return max(float(requested), 0.0)


class ParametricSurvival:
    """Fit a parametric survival distribution per group and expose it as a SurvivalFunction.

    ``distribution`` is one of ``"weibull"`` (default), ``"exponential"``, ``"lognormal"``, or
    ``"loglogistic"``. ``data`` may be a :class:`~tenure.study_design.StudyDesign` or its derived
    canonical table; ``by`` selects grouping column(s) (declared via ``group_cols``), ``by=None``
    fits a single ``"overall"`` curve. Delayed entry (left truncation) flows through.
    """

    def __init__(self, distribution: str = "weibull", alpha: float = 0.05) -> None:
        if distribution not in _DISTRIBUTIONS:
            raise TenureValidationError(
                f"Unknown distribution {distribution!r}; choose from {sorted(_DISTRIBUTIONS)}."
            )
        self.distribution = distribution
        self.alpha = alpha
        self._survival: SurvivalFunction | None = None

    def fit(self, data, *, by=None) -> ParametricSurvival:
        ensure_estimable(data)
        table = data.derive() if hasattr(data, "derive") else data
        time_unit = getattr(data, "time_unit", "day")
        labels, order = _group_labels(table, by)
        curves: dict[str, ParametricGroupCurve] = {}
        for label in order:
            mask = (labels == label).to_numpy()
            curves[label] = self._fit_one(as_estimator_frame(table.loc[mask]))
        self._survival = SurvivalFunction(curves, time_unit=time_unit)
        return self

    def _fit_one(self, ef) -> ParametricGroupCurve:
        fitter_cls, attrs = _DISTRIBUTIONS[self.distribution]
        fitter = fitter_cls(alpha=self.alpha)
        fitter.fit(durations=ef.duration, event_observed=ef.event, entry=ef.entry)
        params = tuple(float(getattr(fitter, a)) for a in attrs)

        # Dense grid over the observed range, for plotting only (queries use the closed form).
        t_max = float(ef.duration.max()) if ef.duration.size else 0.0
        grid = np.linspace(0.0, t_max, 200) if t_max > 0.0 else np.array([0.0])
        surv = _survival_fn(self.distribution, params, grid)

        risk_times, n_at_risk = _empirical_at_risk(ef)
        events = ef.duration[ef.event == 1]
        last_event_time = float(events.max()) if events.size else 0.0

        return ParametricGroupCurve(
            times=grid,
            survival=surv,
            ci_lower=surv,
            ci_upper=surv,
            median=float(fitter.median_survival_time_),
            risk_times=risk_times,
            n_at_risk=n_at_risk,
            last_event_time=last_event_time,
            distribution=self.distribution,
            params=params,
        )

    def _require_fitted(self) -> SurvivalFunction:
        if self._survival is None:
            raise RuntimeError("ParametricSurvival is not fitted yet; call .fit(...) first.")
        return self._survival

    @property
    def survival_(self) -> SurvivalFunction:
        """The fitted multi-group SurvivalFunction (what the business layer consumes)."""
        return self._require_fitted()

    def survival_at(self, times, group=None) -> pd.DataFrame:
        return self._require_fitted().survival_at(times, group=group)

    def median_survival(self, group=None) -> pd.DataFrame:
        return self._require_fitted().median_survival(group=group)

    @property
    def params_(self) -> pd.DataFrame:
        """Tidy fitted parameters per group: [group, distribution, parameter, value].

        For Weibull, ``shape`` > 1 means churn hazard rises with tenure, < 1 means it falls, and
        == 1 reduces to the memoryless exponential.
        """
        survival = self._require_fitted()
        names = _PARAM_NAMES[self.distribution]
        rows = []
        for group in survival.groups:
            curve = survival.curve(group)
            for name, value in zip(names, curve.params, strict=True):
                rows.append(
                    {
                        "group": group,
                        "distribution": self.distribution,
                        "parameter": name,
                        "value": value,
                    }
                )
        return pd.DataFrame(rows, columns=["group", "distribution", "parameter", "value"])
