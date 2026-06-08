"""Out-of-time (temporal) holdout -- the thesis-defining v0.4 split (DV4-2/3, A5, A9).

``temporal_holdout(design, cutoff)`` partitions a study at a calendar cutoff into:
- a TRAIN design with ALL observation administratively censored at the cutoff (no post-cutoff event
  can leak into training -- the panel/leakage guarantee, DV4-7), and
- a TEST cohort of subjects still at risk at the cutoff, evaluated on the post-cutoff clock (the
  prediction target is survival CONDITIONAL on having reached the cutoff, DV4-3).

It is the calendar-clock sibling of the v0.3 ``landmark`` helper: each subject is landmarked at its
own tenure at the cutoff (``c_i = (cutoff - origin_i)``). Single-spell and interval designs are
handled uniformly -- a row crossing the cutoff is split exactly at the cutoff.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

from tenure._frame import ENTRY, EVENT, EXIT, ID, ORIGIN, ensure_estimable, unit_factor
from tenure.exceptions import TenureValidationError
from tenure.study_design import StudyDesign
from tenure.validation.result import VAL001_RANDOM_SPLIT


@dataclass(frozen=True)
class TestCohort:
    """The out-of-time test set from ``temporal_holdout`` (DV4-3).

    Evaluation runs on the post-cutoff clock; the prediction target is survival conditional on
    reaching the cutoff.
    - ``table``: one row per subject -- ``id``, ``eval_start`` (the subject's tenure at the cutoff),
      ``eval_duration`` (time from cutoff to event/censoring), ``eval_event`` (1 if churned after
      the cutoff), plus each covariate's value AS OF the cutoff (for static prediction).
    - ``paths``: the per-subject post-cutoff interval panel (the crossing interval trimmed to start
      at the cutoff), so a time-varying model can predict along the covariate trajectory.
    """

    table: pd.DataFrame
    paths: pd.DataFrame
    prediction_time: pd.Timestamp
    time_unit: str
    covariate_cols: list

    @property
    def n(self) -> int:
        return len(self.table)

    @property
    def ids(self) -> list:
        return self.table[ID].tolist()


def _to_date(origin, tenure, factor):
    return origin + pd.to_timedelta(float(tenure) * factor, unit="D")


def temporal_holdout(design, cutoff):
    """Split ``design`` at a calendar ``cutoff`` into ``(train_design, TestCohort)`` (DV4-2/3).

    Train is the design with everything censored at the cutoff (events on or before the cutoff
    are preserved). Test is the at-risk-at-cutoff cohort with post-cutoff outcomes. Subjects whose
    observation begins at or after the cutoff are in neither (not at risk at the cutoff).
    """
    ensure_estimable(design)
    cutoff_ts = pd.Timestamp(cutoff)
    table = design.canonical
    time_unit = design.time_unit
    factor = unit_factor(time_unit)
    covariate_cols = list(getattr(design, "covariate_cols", []) or [])
    group_cols = list(getattr(design, "group_cols", []) or [])
    carry = covariate_cols + [c for c in group_cols if c not in covariate_cols]

    train_rows: list[dict] = []
    test_table_rows: list[dict] = []
    test_path_rows: list[dict] = []

    for cid, raw in table.groupby(ID, sort=False):
        g = raw.sort_values(ENTRY).reset_index(drop=True)
        origin = g[ORIGIN].iloc[0]
        c = (cutoff_ts - origin) / pd.Timedelta(days=1) / factor  # cutoff in tenure units
        entries = g[ENTRY].to_numpy(dtype=float)
        exits = g[EXIT].to_numpy(dtype=float)
        events = g[EVENT].to_numpy(dtype=int)
        first_entry, last_exit, terminal_event = (
            float(entries[0]),
            float(exits[-1]),
            int(events[-1]),
        )

        if first_entry >= c:
            continue  # entered at/after the cutoff -> in neither train nor test (DV4-3 edge)

        # --- train: rows up to the cutoff; the crossing row is truncated and censored at it ---
        for i in range(len(g)):
            entry, exit_, ev = entries[i], exits[i], int(events[i])
            if entry >= c:
                break  # this and later rows are entirely post-cutoff
            row = {
                ID: cid,
                ORIGIN: origin,
                "start": _to_date(origin, entry, factor),
                "end": _to_date(origin, min(exit_, c), factor),
                "event": ev if exit_ <= c else 0,  # an event after the cutoff cannot leak in
            }
            for col in carry:
                row[col] = g[col].iloc[i]
            train_rows.append(row)

        # --- test: subjects at risk at the cutoff (survived to it) ---
        if last_exit > c:
            cover = np.where((entries <= c) & (exits > c))[0]
            cover_i = int(cover[0]) if cover.size else len(g) - 1
            test_row = {
                ID: cid,
                "eval_start": c,
                "eval_duration": last_exit - c,
                "eval_event": terminal_event,
            }
            for col in carry:
                test_row[col] = g[col].iloc[cover_i]
            test_table_rows.append(test_row)

            for i in range(len(g)):
                if exits[i] <= c:
                    continue  # entirely pre-cutoff
                start_tenure = max(entries[i], c)  # crossing interval starts at the cutoff
                prow = {
                    ID: cid,
                    ORIGIN: origin,
                    "start": _to_date(origin, start_tenure, factor),
                    "end": _to_date(origin, exits[i], factor),
                    "event": int(events[i]),
                }
                for col in carry:
                    prow[col] = g[col].iloc[i]
                test_path_rows.append(prow)

    if not train_rows:
        raise TenureValidationError(
            f"cutoff {cutoff_ts.date()} precedes every subject's entry; nothing to train on."
        )
    if not test_table_rows:
        raise TenureValidationError(
            f"no subjects are at risk at cutoff {cutoff_ts.date()}; choose an earlier cutoff."
        )

    train_design = StudyDesign.from_intervals(
        pd.DataFrame(train_rows),
        id_col=ID,
        origin_col=ORIGIN,
        interval_start_col="start",
        interval_end_col="end",
        event_col="event",
        covariate_cols=covariate_cols,
        group_cols=group_cols,
        time_unit=time_unit,
    )

    ordered = [ID, "eval_start", "eval_duration", "eval_event", *carry]
    test_table = pd.DataFrame(test_table_rows)[ordered]
    test_cohort = TestCohort(
        table=test_table,
        paths=pd.DataFrame(test_path_rows),
        prediction_time=cutoff_ts,
        time_unit=time_unit,
        covariate_cols=covariate_cols,
    )
    return train_design, test_cohort


def random_split(design, *, test_fraction: float = 0.25, seed: int = 0):
    """Partition subject ids into (train, test) at random -- a DISCOURAGED footgun (warns VAL001).

    Random splitting is not forward-in-time: a customer's future outcome can leak into training.
    Prefer ``temporal_holdout``. Returns ``(train_ids, test_ids)`` numpy arrays; the warning is the
    point.
    """
    warnings.warn(
        f"{VAL001_RANDOM_SPLIT}: random splitting is not forward-in-time, so a customer's future "
        "can leak into training. Prefer temporal_holdout(design, cutoff) for honest out-of-time "
        "validation.",
        UserWarning,
        stacklevel=2,
    )
    ids = pd.unique(design.canonical[ID])
    rng = np.random.default_rng(seed)
    is_test = rng.random(len(ids)) < test_fraction
    return ids[~is_test], ids[is_test]
