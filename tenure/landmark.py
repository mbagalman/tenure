"""Landmarking (DV3-5, A5): build a static landmark dataset from an interval design.

Landmark analysis is the *static-model* way to avoid immortal-time bias: pick a landmark tenure
``L``, keep only subjects still at risk at ``L``, fix each subject's covariates to their value AS OF
``L``, and analyze survival from ``L`` onward (delayed entry at ``L``, so the clock is unchanged).
The result is a single-interval ``StudyDesign`` that Kaplan-Meier / CoxPH consume unchanged -- a
complement to the time-varying Cox, useful when a one-number-per-subject covariate is wanted.

This keeps the dual clock honest (A5): the landmark is on the *tenure* clock, and any calendar- or
behavior-derived covariate simply rides the interval rows and is read off as of the landmark.
"""

from __future__ import annotations

import pandas as pd

from tenure._frame import ENTRY, EVENT, EXIT, ID, ORIGIN, unit_factor
from tenure.exceptions import TenureValidationError
from tenure.study_design import StudyDesign


def landmark(design, landmark_time: float) -> StudyDesign:
    """Return a single-interval ``StudyDesign`` of subjects at risk at ``landmark_time``.

    Each kept subject contributes one interval ``[landmark_time, final_exit]`` (delayed entry at the
    landmark) carrying its covariate values as of the landmark and its terminal event. Subjects who
    churned or were censored at or before ``landmark_time`` are dropped (they were not at risk).
    """
    if not getattr(design, "interval", False):
        raise TenureValidationError(
            "landmark() expects an interval design built with StudyDesign.from_intervals."
        )
    covariate_cols = list(getattr(design, "covariate_cols", []) or [])
    table = design.canonical
    factor = unit_factor(design.time_unit)

    rows = []
    for cid, group in table.groupby(ID, sort=False):
        final_exit = float(group[EXIT].max())
        if final_exit <= landmark_time:
            continue  # already churned or censored by the landmark -> not in the risk set
        covering = group[(group[ENTRY] <= landmark_time) & (group[EXIT] > landmark_time)]
        if covering.empty:
            continue  # a gap in coverage at the landmark (defensive; contiguous intervals fill it)
        as_of = covering.iloc[0]
        origin = group[ORIGIN].iloc[0]
        rows.append(
            {
                ID: cid,
                ORIGIN: origin,
                "start": origin + pd.to_timedelta(landmark_time * factor, unit="D"),
                "end": origin + pd.to_timedelta(final_exit * factor, unit="D"),
                "event": int(group[EVENT].max()),
                **{col: as_of[col] for col in covariate_cols},
            }
        )

    if not rows:
        raise TenureValidationError(
            f"No subjects are still at risk at landmark_time={landmark_time}; "
            "choose an earlier landmark."
        )

    return StudyDesign.from_intervals(
        pd.DataFrame(rows),
        id_col=ID,
        origin_col=ORIGIN,
        interval_start_col="start",
        interval_end_col="end",
        event_col="event",
        covariate_cols=covariate_cols,
        time_unit=design.time_unit,
    )
