"""StudyDesign: the explicit-semantics ingress layer (the first AD-1 seam).

Phase 0 ships the event-date schema (``from_event_dates``). The status-label schema and
the interval/counting-process schema arrive in later v0.1 / v0.3 slices.
"""

from __future__ import annotations

import pandas as pd

from tenure._frame import ENTRY, EVENT, EXIT, ID, ORIGIN, STATUS, to_tenure
from tenure.exceptions import TenureValidationError


def _coerce_ts(value, name: str) -> pd.Timestamp | None:
    if value is None:
        return None
    ts = pd.Timestamp(value)
    if pd.isna(ts):
        raise TenureValidationError(f"{name} is not a valid date: {value!r}")
    return ts


class StudyDesign:
    """An explicit, validated churn study design and its derived canonical table.

    Construct via a schema classmethod (currently :meth:`from_event_dates`). The audit and
    estimators read the derived canonical table plus the named design attributes -- origin,
    ``analysis_start``, ``event_observed_from``, ``entry_modeled``,
    ``includes_pre_entry_churners`` -- which are kept deliberately distinct (these concepts
    are the product's spine).
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
    ) -> None:
        self.canonical = canonical
        self.analysis_start = analysis_start
        self.event_observed_from = event_observed_from
        self.entry_modeled = entry_modeled
        self.includes_pre_entry_churners = includes_pre_entry_churners
        self.group_cols = list(group_cols)
        self.time_unit = time_unit

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
            f"entry_modeled={self.entry_modeled}, "
            f"includes_pre_entry_churners={self.includes_pre_entry_churners!r})"
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
        """Build a design from origin + churn-date columns (null churn date = active).

        A subject is censored at ``active_as_of`` when their churn date is null or later
        than the snapshot. Provide ``event_observed_from`` (the date event recording becomes
        reliable) or ``entry_col`` to model delayed entry for a Window-Cut study.
        """
        group_cols = list(group_cols or [])

        required = [id_col, origin_col, churn_date_col, *group_cols]
        if entry_col is not None:
            required.append(entry_col)
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise TenureValidationError(f"Missing required column(s): {missing}")

        data = df.reset_index(drop=True)

        ids = data[id_col]
        if ids.duplicated().any():
            n_dup = int(ids.duplicated().sum())
            raise TenureValidationError(
                f"{n_dup} duplicate value(s) in id_col={id_col!r}; each row must be one "
                "analysis unit. Supply a per-spell unique key (e.g. subscription_id) as "
                "id_col, or dedup with an explicit policy -- keep-first is unbiased; "
                "keep-most-recent introduces win-back selection bias."
            )

        active_as_of_ts = _coerce_ts(active_as_of, "active_as_of")
        if active_as_of_ts is None:
            raise TenureValidationError("active_as_of is required for the event-date schema.")
        analysis_start_ts = _coerce_ts(analysis_start, "analysis_start")
        eof_ts = _coerce_ts(event_observed_from, "event_observed_from")

        origin = pd.to_datetime(data[origin_col], errors="raise")
        churn = pd.to_datetime(data[churn_date_col], errors="coerce")  # NaT -> active

        event = churn.notna() & (churn <= active_as_of_ts)
        exit_date = churn.where(event, active_as_of_ts)

        if (origin > exit_date).any():
            bad = int((origin > exit_date).sum())
            raise TenureValidationError(
                f"{bad} row(s) have origin after their exit date; check origin/active_as_of."
            )

        exit_tenure = pd.Series(to_tenure(exit_date, origin, time_unit)).clip(lower=0.0)

        if entry_col is not None:
            entry_date = pd.to_datetime(data[entry_col], errors="raise")
            entry_tenure = pd.Series(to_tenure(entry_date, origin, time_unit)).clip(lower=0.0)
        elif eof_ts is not None:
            entry_tenure = pd.Series(to_tenure(eof_ts, origin, time_unit)).clip(lower=0.0)
        else:
            entry_tenure = pd.Series(0.0, index=data.index)

        entry_modeled = (entry_col is not None) or (eof_ts is not None)

        if entry_modeled and (entry_tenure >= exit_tenure).any():
            bad = int((entry_tenure >= exit_tenure).sum())
            raise TenureValidationError(
                f"{bad} row(s) have entry tenure >= exit tenure -- these subjects exited "
                "before observation began (pre-entry churners) and are unobservable under "
                "this design. Exclude them before analysis."
            )

        canonical = pd.DataFrame(
            {
                ID: ids.to_numpy(),
                ORIGIN: origin.to_numpy(),
                ENTRY: entry_tenure.to_numpy(dtype=float),
                EXIT: exit_tenure.to_numpy(dtype=float),
                EVENT: event.to_numpy(dtype=int),
                STATUS: ["churn" if e else "active" for e in event],
            }
        )
        for col in group_cols:
            canonical[col] = data[col].to_numpy()

        return cls(
            canonical=canonical,
            analysis_start=analysis_start_ts,
            event_observed_from=eof_ts,
            entry_modeled=entry_modeled,
            includes_pre_entry_churners=includes_pre_entry_churners,
            group_cols=group_cols,
            time_unit=time_unit,
        )

    @classmethod
    def from_status(cls, *args, **kwargs):  # noqa: D401 - placeholder
        """Status-label schema -- lands in a later v0.1 slice."""
        raise NotImplementedError(
            "StudyDesign.from_status (status-label schema) is not implemented yet; "
            "use from_event_dates for now."
        )
