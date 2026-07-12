"""Hybrid (spliced) survival curves: empirical where you have data, modeled beyond (A12).

Kaplan-Meier is the honest choice inside your data window; a fitted model is the principled way
past it. A hybrid curve gives you both: the EMPIRICAL curve up to its supported horizon (the
splice boundary), then the MODELED curve's shape beyond, rescaled so the two segments meet with
no jump. The result is the richer alternative to truncate-and-relabel -- long-horizon RMST/LTV
that still uses every observed event, with the data/model boundary recorded on the curve (and
drawn on plots) so nobody mistakes projection for evidence.

Splice construction: with boundary ``b`` per group,

    S_hybrid(t) = S_emp(t)                                for t <= b
    S_hybrid(t) = S_emp(b) * S_mod(t) / S_mod(b)          for t >  b

The tail is the model's CONDITIONAL survival past ``b``, anchored at the empirical value -- the
model contributes shape, the data contributes level. Confidence intervals exist only on the
empirical segment; beyond the boundary the CI collapses to the point estimate (a model tail has
no sampling band here, matching Cox/parametric curves).

The tail model's own support still applies: a parametric tail extends to any horizon, while a
step-curve tail (e.g. a Cox profile curve, whose baseline ends with the training data) caps the
hybrid's effective horizon where it flattens -- splicing does not launder a flat tail into
extrapolation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from tenure.estimators.survival import GroupCurve, SurvivalFunction
from tenure.exceptions import TenureValidationError

# Matches outputs._common.DEFAULT_MIN_AT_RISK (not imported: estimators must not depend on the
# business-output layer above them).
_DEFAULT_MIN_AT_RISK = 10


def _as_survival(estimator) -> SurvivalFunction:
    if isinstance(estimator, SurvivalFunction):
        return estimator
    survival = getattr(estimator, "survival_", None)
    if isinstance(survival, SurvivalFunction):
        return survival
    raise TypeError(
        "Expected a SurvivalFunction or a fitted estimator exposing .survival_, "
        f"got {type(estimator).__name__}."
    )


@dataclass(frozen=True)
class HybridGroupCurve(GroupCurve):
    """One group's spliced curve: empirical to ``boundary``, conditional model tail beyond.

    Carries the per-segment provenance A12 requires: ``boundary`` (the splice tenure),
    ``empirical`` and ``modeled`` (the source curves), and ``scale`` (the continuity anchor
    ``S_emp(boundary) / S_mod(boundary)`` applied to the tail). The inherited array fields are a
    materialized rendering for plotting; queries evaluate the segments directly.
    """

    boundary: float = 0.0
    empirical: GroupCurve | None = None
    modeled: GroupCurve | None = None
    scale: float = 1.0

    def at(self, t):  # noqa: D102 -- inherited contract
        t = np.atleast_1d(np.asarray(t, dtype=float))
        emp_s, emp_lo, emp_hi = self.empirical.at(t)
        mod_s, _, _ = self.modeled.at(t)
        tail = self.scale * mod_s
        in_data = t <= self.boundary
        s = np.where(in_data, emp_s, tail)
        # CI only where there is data; the model tail is a point estimate.
        return s, np.where(in_data, emp_lo, tail), np.where(in_data, emp_hi, tail)

    def integral(self, a: float, b: float) -> float:
        """Exact piecewise integral: empirical up to the boundary, scaled model tail beyond."""
        if b <= a:
            return 0.0
        total = 0.0
        if a < self.boundary:
            total += self.empirical.integral(a, min(b, self.boundary))
        if b > self.boundary:
            total += self.scale * self.modeled.integral(max(a, self.boundary), b)
        return float(total)

    def effective_horizon(self, requested: float, min_at_risk: int) -> float:
        """The tail model's reach, never less than the splice boundary.

        A parametric tail supports any horizon (its effective horizon is the requested one); a
        step-curve tail caps the hybrid where its own support ends, so a flat Cox tail is not
        silently integrated as if it were a projection.
        """
        tail_reach = self.modeled.effective_horizon(requested, min_at_risk)
        return min(float(requested), max(self.boundary, tail_reach))


def _s_at(curve: GroupCurve, t: float) -> float:
    """Scalar survival lookup (GroupCurve.at expects array-like)."""
    return float(curve.at(np.array([float(t)]))[0][0])


def _tail_median(scale: float, modeled: GroupCurve, boundary: float) -> float:
    """Tenure where ``scale * S_mod(t)`` crosses 0.5, searched past ``boundary`` (inf if never)."""
    lo = boundary
    hi = max(boundary, 1.0)
    for _ in range(60):  # expanding bracket, then bisection
        hi *= 2.0
        if scale * _s_at(modeled, hi) <= 0.5:
            break
    else:
        return float("inf")
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if scale * _s_at(modeled, mid) > 0.5:
            lo = mid
        else:
            hi = mid
    return float(hi)


def hybrid_survival(
    empirical: Any,
    modeled: Any,
    *,
    horizon: float | None = None,
    min_at_risk: int = _DEFAULT_MIN_AT_RISK,
) -> SurvivalFunction:
    """Splice an empirical survival function with a modeled tail into one SurvivalFunction.

    Per group, the curve is the empirical estimate up to its supported horizon (the splice
    boundary: last event time, capped where the risk set thins below ``min_at_risk``) and the
    model's conditional survival beyond, rescaled to meet the empirical value exactly at the
    boundary. Business outputs consume the result like any other curve; RMST/LTV integrate
    seamlessly across both segments, and ``plot_survival`` marks the boundary.

    Args:
        empirical: A fitted ``KaplanMeier`` (or any SurvivalFunction) -- the data segment.
        modeled: A fitted ``ParametricSurvival`` (the intended tail source) or any
            SurvivalFunction with the same group labels -- e.g. fit both with the same ``by=``.
            A step-curve tail (Cox profile curves) is accepted but caps the hybrid's effective
            horizon at its own support.
        horizon: How far past the boundary to materialize the tail in the stored plotting arrays
            (queries are exact at any tenure regardless). Defaults to twice each boundary.
        min_at_risk: Risk-set floor defining each group's splice boundary.

    Returns:
        A ``SurvivalFunction`` of ``HybridGroupCurve``s carrying the splice boundary and the
        source curves as provenance.

    Raises:
        TenureValidationError: If the curves' time units differ, the group labels differ, or the
            model assigns zero survival at a splice boundary (no tail can be anchored there).
    """
    emp = _as_survival(empirical)
    mod = _as_survival(modeled)
    if emp.time_unit != mod.time_unit:
        raise TenureValidationError(
            f"empirical and modeled curves must share a time_unit; got "
            f"{emp.time_unit!r} vs {mod.time_unit!r}. Splicing a daily curve with a monthly "
            "tail would silently misalign every tenure (review fix)."
        )
    if set(emp.groups) != set(mod.groups):
        raise TenureValidationError(
            f"Group labels differ: empirical has {sorted(emp.groups)}, modeled has "
            f"{sorted(mod.groups)}. Fit both estimators with the same grouping (same by= / "
            "matching profile labels) so each empirical curve has a tail."
        )

    curves: dict[str, HybridGroupCurve] = {}
    for group in emp.groups:
        e = emp.curve(group)
        m = mod.curve(group)
        boundary = e.effective_horizon(np.inf, min_at_risk)
        s_emp_b = _s_at(e, boundary)
        s_mod_b = _s_at(m, boundary)
        if s_mod_b <= 0.0:
            raise TenureValidationError(
                f"Modeled survival for group {group!r} is 0 at the splice boundary "
                f"({boundary:g}); a tail cannot be anchored there. Check the model fit."
            )
        scale = s_emp_b / s_mod_b

        # Materialized arrays for plotting: empirical steps on [0, b] + a dense tail grid.
        plot_end = float(horizon) if horizon is not None else 2.0 * boundary
        keep = e.times <= boundary
        tail_grid = (
            np.linspace(boundary, plot_end, 100) if plot_end > boundary else np.array([boundary])
        )
        tail_surv = scale * m.at(tail_grid)[0]
        times = np.concatenate([e.times[keep], tail_grid])
        survival = np.concatenate([e.survival[keep], tail_surv])
        ci_lower = np.concatenate([e.ci_lower[keep], tail_surv])
        ci_upper = np.concatenate([e.ci_upper[keep], tail_surv])

        if np.isfinite(e.median) and e.median <= boundary:
            median = float(e.median)
        else:
            median = _tail_median(scale, m, boundary)

        curves[group] = HybridGroupCurve(
            times=times,
            survival=survival,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            median=median,
            risk_times=e.risk_times,
            n_at_risk=e.n_at_risk,
            last_event_time=e.last_event_time,
            boundary=float(boundary),
            empirical=e,
            modeled=m,
            scale=float(scale),
        )
    return SurvivalFunction(curves, time_unit=emp.time_unit)
