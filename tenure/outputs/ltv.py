"""Period-correct survival-weighted LTV (FR-BO-3, formula D-S2).

Split the (effective) horizon into periods of length ``L`` (the margin's period expressed in
the curve's time unit). Each period contributes ``M * S_bar_m * (1+d)^(-m) * width/L`` where
``S_bar_m`` is the average survival over the period (exact KM integral / width). With no
discounting this reduces exactly to ``(M/L) * RMST(H_eff)``.
"""

from __future__ import annotations

import math

import pandas as pd

from tenure.outputs._common import DEFAULT_MIN_AT_RISK, as_survival, period_length_in_units

_LTV_COLUMNS = ["group", "requested_horizon", "effective_horizon", "ltv", "period", "truncated"]


def survival_weighted_ltv(
    estimator,
    *,
    period_margin: float,
    horizon: float,
    discount_rate: float = 0.0,
    period: str = "month",
    min_at_risk: int = DEFAULT_MIN_AT_RISK,
) -> pd.DataFrame:
    """Survival-weighted LTV in margin units per group, truncated at the supported horizon."""
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
