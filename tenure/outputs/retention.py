"""Retention at horizons and Restricted Mean Survival Time (RMST)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from tenure.outputs._common import DEFAULT_HORIZONS, DEFAULT_MIN_AT_RISK, as_survival

_RETENTION_COLUMNS = ["group", "horizon", "retention", "ci_lower", "ci_upper", "supported"]
_RMST_COLUMNS = ["group", "requested_horizon", "effective_horizon", "rmst", "truncated"]


def retention_at(
    estimator,
    horizons=DEFAULT_HORIZONS,
    *,
    min_at_risk: int = DEFAULT_MIN_AT_RISK,
) -> pd.DataFrame:
    """Retention (survival) at each horizon, per group, with a support flag (TNR005).

    ``supported`` is False when the horizon exceeds the group's supported horizon -- the
    retention there is read off the flat KM tail and should be treated with caution.
    """
    survival = as_survival(estimator)
    # Order-preserving dedupe: a repeated horizon (e.g. [30, 30]) would duplicate the time index
    # and crash the per-horizon lookup (review fix); asking twice means asking once.
    horizons = list(dict.fromkeys(float(h) for h in np.atleast_1d(horizons)))
    rows = []
    for group in survival.groups:
        curve = survival.curve(group)
        at = survival.survival_at(horizons, group=group).set_index("time")
        for h in horizons:
            point = at.loc[h]
            rows.append(
                {
                    "group": group,
                    "horizon": h,
                    "retention": float(point["survival"]),
                    "ci_lower": float(point["ci_lower"]),
                    "ci_upper": float(point["ci_upper"]),
                    "supported": h <= curve.effective_horizon(h, min_at_risk) + 1e-9,
                }
            )
    return pd.DataFrame(rows, columns=_RETENTION_COLUMNS)


def rmst(estimator, *, horizon: float, min_at_risk: int = DEFAULT_MIN_AT_RISK) -> pd.DataFrame:
    """Restricted Mean Survival Time through ``horizon``, per group.

    No flat extrapolation past support: the integral runs to a per-group effective horizon
    (FR-BO-2). ``truncated`` is True and ``effective_horizon`` < ``horizon`` when that bound bit.
    """
    survival = as_survival(estimator)
    requested = float(horizon)
    rows = []
    for group in survival.groups:
        curve = survival.curve(group)
        h_eff = curve.effective_horizon(requested, min_at_risk)
        rows.append(
            {
                "group": group,
                "requested_horizon": requested,
                "effective_horizon": h_eff,
                "rmst": curve.integral(0.0, h_eff),
                "truncated": h_eff < requested - 1e-9,
            }
        )
    return pd.DataFrame(rows, columns=_RMST_COLUMNS)
