"""Shared helpers for the business-output layer."""

from __future__ import annotations

from tenure.estimators.survival import SurvivalFunction

DEFAULT_HORIZONS = (30, 60, 90, 180, 365)
DEFAULT_MIN_AT_RISK = 10

# Period length in days, used to reconcile a margin's period with the curve's time unit.
PERIOD_DAYS = {"day": 1.0, "week": 7.0, "month": 30.4375}
UNIT_DAYS = {"day": 1.0, "week": 7.0, "month": 30.4375}


def as_survival(estimator) -> SurvivalFunction:
    """Accept a SurvivalFunction or any fitted estimator exposing ``.survival_``."""
    if isinstance(estimator, SurvivalFunction):
        return estimator
    survival = getattr(estimator, "survival_", None)
    if isinstance(survival, SurvivalFunction):
        return survival
    raise TypeError(
        "Expected a SurvivalFunction or a fitted estimator exposing .survival_, "
        f"got {type(estimator).__name__}."
    )


def period_length_in_units(period: str, time_unit: str) -> float:
    """Length of one ``period`` expressed in the curve's ``time_unit``."""
    try:
        return PERIOD_DAYS[period] / UNIT_DAYS[time_unit]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported period/time_unit ({period!r}/{time_unit!r}); "
            f"choose from {sorted(PERIOD_DAYS)}."
        ) from exc
