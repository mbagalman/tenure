"""StudyDesign: the explicit-semantics ingress layer (the first AD-1 seam).

Two input schemas:
- ``from_event_dates`` -- origin + churn-date columns (null churn date = active).
- ``from_status`` -- origin + exit + status columns, with a ``status_map`` declaring each
  status's intent ({event, censored, exclude}).

The interval/counting-process schema arrives in a later (v0.3) slice.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tenure._frame import ENTRY, EVENT, EXIT, ID, ORIGIN, STATUS, to_tenure
from tenure.exceptions import TenureValidationError

_VALID_INTENTS = ("event", "censored", "exclude")


def _coerce_ts(value, name: str) -> pd.Timestamp | None:
    if value is None:
        return None
    ts = pd.Timestamp(value)
    if pd.isna(ts):
        raise TenureValidationError(f"{name} is not a valid date: {value!r}")
    return ts


def _check_columns(df: pd.DataFrame, required: list[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise TenureValidationError(f"Missing required column(s): {missing}")


def _check_unique_ids(ids: pd.Series, id_col: str) -> None:
    if ids.duplicated().any():
        n_dup = int(ids.duplicated().sum())
        raise TenureValidationError(
            f"{n_dup} duplicate value(s) in id_col={id_col!r}; each row must be one analysis "
            "unit. Supply a per-spell unique key (e.g. subscription_id) as id_col, or dedup with "
            "an explicit policy -- keep-first is unbiased; keep-most-recent introduces win-back "
            "selection bias."
        )


class StudyDesign:
    """An explicit, validated churn study design and its derived canonical table.

    The audit and estimators read the derived canonical table plus named design attributes
    (origin, ``analysis_start``, ``event_observed_from``, ``entry_modeled``,
    ``includes_pre_entry_churners``) -- kept deliberately distinct (the product's spine). Status
    designs additionally carry ``status_map``, ``n_excluded``, ``n_unmapped``,
    ``unmapped_statuses``, and ``informative_censoring_statuses``.
    """

    def __init__(
        self,
        *,
        canonical: pd.DataFrame,
        analysis_start: pd.Timestamp | None,
        event_observed_from: pd.Timestamp | None,
        entry_modeled: bool,
        includes_pre_entry_churners: bool | None,
        group_cols: list[str],
        time_unit: str,
        status_map: dict | None = None,
        n_excluded: int = 0,
        n_unmapped: int = 0,
        unmapped_statuses: list | None = None,
        informative_censoring_statuses: list | None = None,
    ) -> None:
        self.canonical = canonical
        self.analysis_start = analysis_start
        self.event_observed_from = event_observed_from
        self.entry_modeled = entry_modeled
        self.includes_pre_entry_churners = includes_pre_entry_churners
        self.group_cols = list(group_cols)
        self.time_unit = time_unit
        self.status_map = status_map
        self.n_excluded = n_excluded
        self.n_unmapped = n_unmapped
        self.unmapped_statuses = list(unmapped_statuses or [])
        self.informative_censoring_statuses = list(informative_censoring_statuses or [])

    @property
    def origin(self) -> pd.Series:
        """Per-subject origin (customer birth / t=0) as a datetime Series."""
        return self.canonical[ORIGIN]

    @property
    def n(self) -> int:
        return len(self.canonical)

    def derive(self) -> pd.DataFrame:
        """Return a copy of the inspectable canonical survival table."""
        return self.canonical.copy()

    def __repr__(self) -> str:
        return (
            f"StudyDesign(n={self.n}, time_unit={self.time_unit!r}, "
            f"entry_modeled={self.entry_modeled}, n_excluded={self.n_excluded})"
        )

    @classmethod
    def _assemble(
        cls,
        data: pd.DataFrame,
        *,
        ids,
        origin,
        exit_date,
        event,
        status_label,
        group_cols: list[str],
        analysis_start_ts,
        event_observed_from_ts,
        entry_col: str | None,
        includes_pre_entry_churners,
        time_unit: str,
        status_map=None,
        n_excluded: int = 0,
        n_unmapped: int = 0,
        unmapped_statuses=None,
        informative_censoring_statuses=None,
    ) -> StudyDesign:
        """Shared finalizer: validate, derive tenures, and build the canonical table."""
        origin = pd.to_datetime(origin).reset_index(drop=True)
        exit_date = pd.to_datetime(exit_date).reset_index(drop=True)

        if (origin > exit_date).any():
            bad = int((origin > exit_date).sum())
            raise TenureValidationError(
                f"{bad} row(s) have origin after their exit date; check origin/exit/active_as_of."
            )

        exit_tenure = pd.Series(to_tenure(exit_date, origin, time_unit)).clip(lower=0.0)

        if entry_col is not None:
            entry_date = pd.to_datetime(data[entry_col], errors="raise")
            entry_tenure = pd.Series(to_tenure(entry_date, origin, time_unit)).clip(lower=0.0)
        elif event_observed_from_ts is not None:
            entry_tenure = pd.Series(to_tenure(event_observed_from_ts, origin, time_unit)).clip(
                lower=0.0
            )
        else:
            entry_tenure = pd.Series(0.0, index=range(len(origin)))

        entry_modeled = (entry_col is not None) or (event_observed_from_ts is not None)

        if entry_modeled and (entry_tenure >= exit_tenure).any():
            bad = int((entry_tenure >= exit_tenure).sum())
            raise TenureValidationError(
                f"{bad} row(s) have entry tenure >= exit tenure -- these subjects exited before "
                "observation began (pre-entry churners) and are unobservable under this design. "
                "Exclude them before analysis."
            )

        canonical = pd.DataFrame(
            {
                ID: np.asarray(ids),
                ORIGIN: origin.to_numpy(),
                ENTRY: entry_tenure.to_numpy(dtype=float),
                EXIT: exit_tenure.to_numpy(dtype=float),
                EVENT: np.asarray(event, dtype=int),
                STATUS: np.asarray(status_label),
            }
        )
        for col in group_cols:
            canonical[col] = data[col].to_numpy()

        return cls(
            canonical=canonical,
            analysis_start=analysis_start_ts,
            event_observed_from=event_observed_from_ts,
            entry_modeled=entry_modeled,
            includes_pre_entry_churners=includes_pre_entry_churners,
            group_cols=group_cols,
            time_unit=time_unit,
            status_map=status_map,
            n_excluded=n_excluded,
            n_unmapped=n_unmapped,
            unmapped_statuses=unmapped_statuses,
            informative_censoring_statuses=informative_censoring_statuses,
        )

    @classmethod
    def from_event_dates(
        cls,
        df: pd.DataFrame,
        *,
        id_col: str,
        origin_col: str,
        churn_date_col: str,
        active_as_of,
        analysis_start=None,
        entry_col: str | None = None,
        event_observed_from=None,
        includes_pre_entry_churners: bool | None = None,
        group_cols: list[str] | None = None,
        time_unit: str = "day",
    ) -> StudyDesign:
        """Build a design from origin + churn-date columns (null churn date = active)."""
        group_cols = list(group_cols or [])
        required = [id_col, origin_col, churn_date_col, *group_cols]
        if entry_col is not None:
            required.append(entry_col)
        _check_columns(df, required)

        data = df.reset_index(drop=True)
        _check_unique_ids(data[id_col], id_col)

        active_as_of_ts = _coerce_ts(active_as_of, "active_as_of")
        if active_as_of_ts is None:
            raise TenureValidationError("active_as_of is required for the event-date schema.")
        analysis_start_ts = _coerce_ts(analysis_start, "analysis_start")
        event_observed_from_ts = _coerce_ts(event_observed_from, "event_observed_from")

        origin = pd.to_datetime(data[origin_col], errors="raise")
        churn = pd.to_datetime(data[churn_date_col], errors="coerce")  # NaT -> active
        event = churn.notna() & (churn <= active_as_of_ts)
        exit_date = churn.where(event, active_as_of_ts)
        status_label = np.where(event, "churn", "active")

        return cls._assemble(
            data,
            ids=data[id_col],
            origin=origin,
            exit_date=exit_date,
            event=event.astype(int),
            status_label=status_label,
            group_cols=group_cols,
            analysis_start_ts=analysis_start_ts,
            event_observed_from_ts=event_observed_from_ts,
            entry_col=entry_col,
            includes_pre_entry_churners=includes_pre_entry_churners,
            time_unit=time_unit,
        )

    @classmethod
    def from_status(
        cls,
        df: pd.DataFrame,
        *,
        id_col: str,
        origin_col: str,
        exit_col: str,
        status_col: str,
        status_map: dict,
        active_as_of,
        analysis_start=None,
        entry_col: str | None = None,
        event_observed_from=None,
        includes_pre_entry_churners: bool | None = None,
        group_cols: list[str] | None = None,
        time_unit: str = "day",
    ) -> StudyDesign:
        """Build a design from origin + exit + status columns via an explicit ``status_map``.

        Each status maps to one of {event, censored, exclude}. Excluded rows are dropped and
        counted (``n_excluded``); statuses absent from the map are also dropped and counted
        (``n_unmapped``) and flagged by the audit (TNR003) -- they are never silently coerced
        into an outcome.
        """
        group_cols = list(group_cols or [])
        required = [id_col, origin_col, exit_col, status_col, *group_cols]
        if entry_col is not None:
            required.append(entry_col)
        _check_columns(df, required)

        bad_intents = set(status_map.values()) - set(_VALID_INTENTS)
        if bad_intents:
            raise TenureValidationError(
                f"status_map values must be in {_VALID_INTENTS}; got {sorted(bad_intents)}."
            )

        data = df.reset_index(drop=True)
        _check_unique_ids(data[id_col], id_col)

        active_as_of_ts = _coerce_ts(active_as_of, "active_as_of")
        if active_as_of_ts is None:
            raise TenureValidationError("active_as_of is required for the status schema.")
        analysis_start_ts = _coerce_ts(analysis_start, "analysis_start")
        event_observed_from_ts = _coerce_ts(event_observed_from, "event_observed_from")

        status = data[status_col]
        intent = status.map(status_map)
        present = list(pd.unique(status))
        unmapped_statuses = [s for s in present if s not in status_map]
        n_unmapped = int(intent.isna().sum())
        n_excluded = int((intent == "exclude").sum())

        keep = intent.isin(("event", "censored")).to_numpy()
        kept = data.loc[keep].reset_index(drop=True)
        kept_intent = intent[keep].reset_index(drop=True)
        origin = pd.to_datetime(kept[origin_col], errors="raise")
        exit_date = pd.to_datetime(kept[exit_col], errors="raise")
        event = (kept_intent == "event").astype(int)
        status_label = kept[status_col].reset_index(drop=True)

        # Informative censoring: censored rows that exit before the snapshot are not ordinary
        # administrative right-censoring (TNR003 warns).
        censored_early = (event == 0) & (exit_date < active_as_of_ts)
        informative = sorted({s for s in status_label[censored_early.to_numpy()]})

        return cls._assemble(
            kept,
            ids=kept[id_col],
            origin=origin,
            exit_date=exit_date,
            event=event,
            status_label=status_label,
            group_cols=group_cols,
            analysis_start_ts=analysis_start_ts,
            event_observed_from_ts=event_observed_from_ts,
            entry_col=entry_col,
            includes_pre_entry_churners=includes_pre_entry_churners,
            time_unit=time_unit,
            status_map=dict(status_map),
            n_excluded=n_excluded,
            n_unmapped=n_unmapped,
            unmapped_statuses=unmapped_statuses,
            informative_censoring_statuses=informative,
        )
