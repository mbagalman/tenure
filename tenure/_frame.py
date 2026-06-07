"""Canonical-table contract and the estimator-edge seam (AD-1, AD-4).

This module centralizes two things so a future polars/narwhals migration touches one place:

1. The canonical survival table schema -- the single internal representation every layer
   (audit, estimators, business outputs) consumes.
2. ``as_estimator_frame`` -- materialization to the plain numpy arrays survival estimators
   (lifelines, later) require.

It also owns the date -> tenure conversion so temporal correctness has one tested home.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from tenure.exceptions import TenureValidationError

# Canonical column names (the lingua franca between layers).
ID = "id"
ORIGIN = "origin"
ENTRY = "entry_tenure"
EXIT = "exit_tenure"
EVENT = "event"
STATUS = "status"
CANONICAL_COLUMNS = [ID, ORIGIN, ENTRY, EXIT, EVENT, STATUS]

_UNIT_DAYS = {"day": 1.0, "week": 7.0, "month": 30.4375}


def unit_factor(time_unit: str) -> float:
    """Days per one unit of ``time_unit``."""
    try:
        return _UNIT_DAYS[time_unit]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported time_unit {time_unit!r}; choose from {sorted(_UNIT_DAYS)}."
        ) from exc


def to_tenure(end, origin, time_unit: str = "day") -> np.ndarray:
    """Convert calendar ``end`` (scalar or array-like) minus ``origin`` to tenure.

    Returns a float numpy array, aligned by position to ``origin``. A scalar ``end`` is
    broadcast across all origins (used for snapshot / observation-start dates).
    """
    factor = unit_factor(time_unit)
    origin = pd.to_datetime(pd.Series(np.asarray(origin)).reset_index(drop=True))
    n = len(origin)
    if np.ndim(end) == 0:
        end = pd.Series([pd.Timestamp(end)] * n)
    else:
        end = pd.to_datetime(pd.Series(np.asarray(end)).reset_index(drop=True))
    delta_days = (end - origin).dt.total_seconds().to_numpy() / 86400.0
    return delta_days / factor


@dataclass(frozen=True)
class EstimatorFrame:
    """Plain arrays at the estimator edge (delayed-entry aware).

    ``entry`` and ``duration`` are tenures; ``event`` is 0/1. This is the shape lifelines'
    ``fit(durations, event_observed, entry=...)`` consumes -- the seam where ``to_native``
    will live once a non-pandas backend is supported.
    """

    entry: np.ndarray
    duration: np.ndarray
    event: np.ndarray


def ensure_estimable(design) -> None:
    """Refuse to fit a design that silently dropped unmapped-status rows until it has been audited.

    ``from_status`` drops rows whose status is absent from ``status_map`` (counting them in
    ``n_unmapped``) and relies on TNR003 to block. A low-level caller that fits directly would
    otherwise compute curves on a silently-reduced cohort -- so estimators call this first.
    """
    if getattr(design, "n_unmapped", 0) and not getattr(design, "audited", False):
        raise TenureValidationError(
            f"This design dropped {design.n_unmapped} row(s) whose status was not in status_map, "
            "so the cohort is silently incomplete. Run audit(design) first (it reports this as "
            "TNR003), or extend status_map to cover every status."
        )


def as_estimator_frame(canonical: pd.DataFrame) -> EstimatorFrame:
    """Materialize the canonical table to estimator-ready numpy arrays."""
    return EstimatorFrame(
        entry=canonical[ENTRY].to_numpy(dtype=float),
        duration=canonical[EXIT].to_numpy(dtype=float),
        event=canonical[EVENT].to_numpy(dtype=int),
    )
