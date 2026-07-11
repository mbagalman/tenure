"""Period-correct survival-weighted LTV (FR-BO-3, formula D-S2).

Split the (effective) horizon into periods of length ``L`` (the margin's period expressed in
the curve's time unit). Each period contributes ``M * S_bar_m * (1+d)^(-m) * width/L`` where
``S_bar_m`` is the average survival over the period (exact KM integral / width). With no
discounting this reduces exactly to ``(M/L) * RMST(H_eff)``.
"""

from __future__ import annotations

import math
from typing import Any

import pandas as pd

from tenure.outputs._common import DEFAULT_MIN_AT_RISK, as_survival, period_length_in_units

_LTV_COLUMNS = ["group", "requested_horizon", "effective_horizon", "ltv", "period", "truncated"]


def survival_weighted_ltv(
    estimator: Any,
    *,
    period_margin: float,
    horizon: float,
    discount_rate: float = 0.0,
    period: str = "month",
    min_at_risk: int = DEFAULT_MIN_AT_RISK,
) -> pd.DataFrame:
    """Survival-weighted LTV in margin units per group, truncated at the supported horizon.

    Each period's contribution margin is weighted by the probability the subject is still
    subscribed during that period (the exact Kaplan-Meier integral over the period, not the
    endpoint). The horizon is split into periods of the margin's length; with no discounting the
    result reduces exactly to ``(period_margin / period_length) * RMST(effective_horizon)``.

    Period-correct by construction (FR-BO-3): the margin's ``period`` is reconciled against the
    curve's ``time_unit``, so a daily survival curve and a monthly margin are aligned for you --
    you cannot silently multiply a daily survival probability by a monthly margin.

    Never extrapolates: integration stops at each group's effective (supported) horizon rather
    than reading a flat KM tail. When that horizon is shorter than requested the row is flagged
    ``truncated=True`` and ``effective_horizon`` reports where it stopped (the TNR005 guard).

    Args:
        estimator: Any fitted estimator exposing the survival-function interface (KaplanMeier,
            a CoxPH profile curve, a TimeVaryingCox path curve, ...). LTV consumes the survival
            abstraction, never estimator internals, so all of these are interchangeable here.
        period_margin: Contribution margin ``M`` earned per ``period`` a subject is retained,
            in your currency.
        horizon: LTV horizon, in the curve's ``time_unit`` (e.g. days).
        discount_rate: Optional per-period discount rate for NPV. Period ``m`` is discounted by
            ``(1 + discount_rate) ** (-m)``. Default ``0.0`` (no discounting).
        period: The margin's period -- ``"day"``, ``"week"``, ``"month"``, or ``"year"`` --
            reconciled against the curve's time unit. Default ``"month"``.
        min_at_risk: Minimum at-risk count for a horizon to count as supported; drives the
            effective-horizon truncation.

    Returns:
        A tidy DataFrame with one row per group and columns ``group``, ``requested_horizon``,
        ``effective_horizon``, ``ltv``, ``period``, and ``truncated``.
    """
    survival = as_survival(estimator)
    requested = float(horizon)
    length = period_length_in_units(period, survival.time_unit)
    rows = []
    for group in survival.groups:
        curve = survival.curve(group)
        h_eff = curve.effective_horizon(requested, min_at_risk)
        n_periods = int(math.ceil(h_eff / length)) if h_eff > 0 else 0
        ltv = 0.0
        for m in range(n_periods):
            t0 = m * length
            t1 = min((m + 1) * length, h_eff)
            if t1 <= t0:
                continue
            avg_survival = curve.integral(t0, t1) / (t1 - t0)
            ltv += period_margin * avg_survival * (1.0 + discount_rate) ** (-m) * (t1 - t0) / length
        rows.append(
            {
                "group": group,
                "requested_horizon": requested,
                "effective_horizon": h_eff,
                "ltv": float(ltv),
                "period": period,
                "truncated": h_eff < requested - 1e-9,
            }
        )
    return pd.DataFrame(rows, columns=_LTV_COLUMNS)
