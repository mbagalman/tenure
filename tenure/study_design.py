"""StudyDesign: the explicit-semantics ingress layer (the first AD-1 seam).

Two input schemas:
- ``from_event_dates`` -- origin + churn-date columns (null churn date = active).
- ``from_status`` -- origin + exit + status columns, with a ``status_map`` declaring each
  status's intent ({event, censored, exclude}).

The interval/counting-process schema arrives in a later (v0.3) slice.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from tenure._frame import ENTRY, EVENT, EXIT, ID, ORIGIN, STATUS, to_tenure
from tenure.exceptions import TenureValidationError

_VALID_INTENTS = ("event", "censored", "exclude")
_DEDUP_POLICIES = ("error", "keep-first", "keep-most-recent")


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


def _resolve_duplicates(data: pd.DataFrame, id_col: str, dedup_policy: str) -> pd.DataFrame:
    """Apply the duplicate-id policy. Default 'error' raises (no silent data mutation).

    'keep-first' is the unbiased dedup; 'keep-most-recent' warns (it discards won-back
    customers' earlier spells -- a selection bias that inflates survival).
    """
    if dedup_policy not in _DEDUP_POLICIES:
        raise TenureValidationError(
            f"Unknown dedup_policy {dedup_policy!r}; use one of {_DEDUP_POLICIES}."
        )
    if not data[id_col].duplicated().any():
        return data
    n_dup = int(data[id_col].duplicated().sum())
    if dedup_policy == "error":
        raise TenureValidationError(
            f"{n_dup} duplicate value(s) in id_col={id_col!r}. Each row must be one analysis "
            "unit. Pass a per-spell unique key (e.g. subscription_id) as id_col, or choose a "
            "dedup_policy: 'keep-first' (unbiased) or 'keep-most-recent' (warns -- win-back "
            "selection bias)."
        )
    if dedup_policy == "keep-most-recent":
        warnings.warn(
            f"dedup_policy='keep-most-recent' kept the last of {n_dup} duplicate id row(s); this "
            "discards earlier spells of won-back customers and biases survival upward. Prefer "
            "'keep-first' or a per-spell subscription_id.",
            stacklevel=3,
        )
    keep = "last" if dedup_policy == "keep-most-recent" else "first"
    return data.drop_duplicates(subset=id_col, keep=keep).reset_index(drop=True)


def _build_covariate_mappings(data: pd.DataFrame, covariate_cols) -> dict:
    """Classify each covariate numeric vs categorical and reject nulls (Cox needs complete data)."""
    mappings: dict = {}
    for col in covariate_cols:
        series = data[col]
        if series.isna().any():
            raise TenureValidationError(
                f"covariate_cols column {col!r} has null value(s); regression models require "
                "complete-case data. Impute or drop those rows before building the design."
            )
        if pd.api.types.is_bool_dtype(series) or not pd.api.types.is_numeric_dtype(series):
            levels = sorted(pd.unique(series.astype(str)).tolist())
            mappings[col] = {"kind": "categorical", "levels": levels, "baseline": levels[0]}
        else:
            mappings[col] = {"kind": "numeric"}
    return mappings


def _validate_intervals(ids, origin, start, end, event) -> None:
    """Per-subject interval checks: contiguous & non-overlapping, terminal-only event, start<end."""
    work = pd.DataFrame(
        {
            "id": np.asarray(ids),
            "origin": pd.to_datetime(np.asarray(origin)),
            "start": pd.to_datetime(np.asarray(start)),
            "end": pd.to_datetime(np.asarray(end)),
            "event": np.asarray(event, dtype=int),
        }
    )
    if not set(np.unique(work["event"].to_numpy())).issubset({0, 1}):
        raise TenureValidationError("event_col must contain only 0/1 (or boolean) values.")
    if (work["start"] >= work["end"]).any():
        raise TenureValidationError("Every interval needs interval_start < interval_end.")
    if (work.groupby("id")["origin"].nunique() > 1).any():
        raise TenureValidationError(
            "origin varies within an id; each subject must have a single origin across intervals."
        )

    ordered = work.sort_values(["id", "start"]).reset_index(drop=True)
    same_next = ordered["id"].to_numpy()[1:] == ordered["id"].to_numpy()[:-1]
    contiguous = ordered["start"].to_numpy()[1:] == ordered["end"].to_numpy()[:-1]
    if (same_next & ~contiguous).any():
        raise TenureValidationError(
            "Intervals for a subject must be contiguous and non-overlapping "
            "(each interval_start must equal the previous interval_end)."
        )
    is_last = np.append(~same_next, True)  # last row of each id
    if (ordered["event"].to_numpy()[~is_last] == 1).any():
        raise TenureValidationError(
            "event=1 is only allowed on a subject's terminal (last) interval."
        )
    if (ordered.groupby("id")["event"].sum() > 1).any():
        raise TenureValidationError("A subject may have at most one event.")


class StudyDesign:
    """An explicit, validated churn study design and its derived canonical table.

    The audit and estimators read the derived canonical table plus named design attributes --
    origin, ``analysis_start``, ``event_observed_from``, ``entry_modeled``,
    ``includes_pre_entry_churners``, the attestations, and (status schema) ``status_map`` and
    the excluded/unmapped/informative-censoring tracking.
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
        covariate_cols: list | None = None,
        covariate_mappings: dict | None = None,
        interval: bool = False,
        attest_origin_correct: bool | None = None,
        attest_invariant_covariates: list | None = None,
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
        self.covariate_cols = list(covariate_cols or [])
        self.covariate_mappings = dict(covariate_mappings or {})
        self.interval = interval
        self.attest_origin_correct = attest_origin_correct
        self.attest_invariant_covariates = list(attest_invariant_covariates or [])
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

    def encode_covariates(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Encode covariates to a numeric design matrix (numeric as-is; categorical one-hot,
        drop-first) using the stored ``covariate_mappings`` -- so callers query with raw labels.
        """
        columns: dict = {}
        for col, mapping in self.covariate_mappings.items():
            if mapping["kind"] == "numeric":
                columns[col] = pd.to_numeric(frame[col]).astype(float).to_numpy()
            else:
                for level in mapping["levels"][1:]:  # drop the baseline level
                    name = f"{col}_{level}"
                    columns[name] = (frame[col].astype(str) == level).astype(float).to_numpy()
        if not columns:
            return pd.DataFrame(index=frame.index)
        return pd.DataFrame(columns, index=frame.index)

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
        covariate_cols=(),
        interval: bool = False,
        analysis_start_ts,
        event_observed_from_ts,
        entry_col: str | None,
        includes_pre_entry_churners,
        time_unit: str,
        attest_origin_correct=None,
        attest_invariant_covariates=None,
        status_map=None,
        n_excluded: int = 0,
        n_unmapped: int = 0,
        unmapped_statuses=None,
        informative_censoring_statuses=None,
    ) -> StudyDesign:
        """Shared finalizer: validate, derive tenures, and build the canonical table."""
        origin = pd.to_datetime(origin).reset_index(drop=True)
        exit_date = pd.to_datetime(exit_date).reset_index(drop=True)

        if len(origin) == 0:
            raise TenureValidationError(
                "Study design has zero rows (after dedup / exclusions / unmapped drops); "
                "nothing to analyze."
            )
        if origin.isna().any() or exit_date.isna().any():
            raise TenureValidationError(
                "origin/exit contain null dates; every analysis row needs a valid origin and "
                "exit (active customers exit at the snapshot date)."
            )

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
        for col in dict.fromkeys([*group_cols, *covariate_cols]):
            canonical[col] = data[col].to_numpy()

        covariate_mappings = _build_covariate_mappings(data, covariate_cols)

        return cls(
            canonical=canonical,
            analysis_start=analysis_start_ts,
            event_observed_from=event_observed_from_ts,
            entry_modeled=entry_modeled,
            includes_pre_entry_churners=includes_pre_entry_churners,
            group_cols=group_cols,
            time_unit=time_unit,
            covariate_cols=list(covariate_cols),
            covariate_mappings=covariate_mappings,
            interval=interval,
            attest_origin_correct=attest_origin_correct,
            attest_invariant_covariates=attest_invariant_covariates,
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
        covariate_cols: list[str] | None = None,
        time_unit: str = "day",
        dedup_policy: str = "error",
        attest_origin_correct: bool | None = None,
        attest_invariant_covariates: list[str] | None = None,
    ) -> StudyDesign:
        """Build a design from origin + churn-date columns (null churn date = active)."""
        group_cols = list(group_cols or [])
        covariate_cols = list(covariate_cols or [])
        required = [id_col, origin_col, churn_date_col, *group_cols, *covariate_cols]
        if entry_col is not None:
            required.append(entry_col)
        _check_columns(df, required)

        data = _resolve_duplicates(df.reset_index(drop=True), id_col, dedup_policy)

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
            covariate_cols=covariate_cols,
            analysis_start_ts=analysis_start_ts,
            event_observed_from_ts=event_observed_from_ts,
            entry_col=entry_col,
            includes_pre_entry_churners=includes_pre_entry_churners,
            time_unit=time_unit,
            attest_origin_correct=attest_origin_correct,
            attest_invariant_covariates=attest_invariant_covariates,
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
        covariate_cols: list[str] | None = None,
        time_unit: str = "day",
        dedup_policy: str = "error",
        attest_origin_correct: bool | None = None,
        attest_invariant_covariates: list[str] | None = None,
    ) -> StudyDesign:
        """Build a design from origin + exit + status columns via an explicit ``status_map``.

        Each status maps to one of {event, censored, exclude}. Excluded rows are dropped and
        counted (``n_excluded``); statuses absent from the map are also dropped and counted
        (``n_unmapped``) and flagged by the audit (TNR003) -- never silently coerced.
        """
        group_cols = list(group_cols or [])
        covariate_cols = list(covariate_cols or [])
        required = [id_col, origin_col, exit_col, status_col, *group_cols, *covariate_cols]
        if entry_col is not None:
            required.append(entry_col)
        _check_columns(df, required)

        bad_intents = set(status_map.values()) - set(_VALID_INTENTS)
        if bad_intents:
            raise TenureValidationError(
                f"status_map values must be in {_VALID_INTENTS}; got {sorted(bad_intents)}."
            )

        data = _resolve_duplicates(df.reset_index(drop=True), id_col, dedup_policy)

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
            covariate_cols=covariate_cols,
            analysis_start_ts=analysis_start_ts,
            event_observed_from_ts=event_observed_from_ts,
            entry_col=entry_col,
            includes_pre_entry_churners=includes_pre_entry_churners,
            time_unit=time_unit,
            attest_origin_correct=attest_origin_correct,
            attest_invariant_covariates=attest_invariant_covariates,
            status_map=dict(status_map),
            n_excluded=n_excluded,
            n_unmapped=n_unmapped,
            unmapped_statuses=unmapped_statuses,
            informative_censoring_statuses=informative,
        )

    @classmethod
    def from_intervals(
        cls,
        df: pd.DataFrame,
        *,
        id_col: str,
        origin_col: str,
        interval_start_col: str,
        interval_end_col: str,
        event_col: str,
        covariate_cols: list[str] | None = None,
        group_cols: list[str] | None = None,
        time_unit: str = "day",
    ) -> StudyDesign:
        """Build a counting-process (start-stop) design with time-varying covariates.

        One row per (subject, interval); a subject's covariates may change between intervals.
        ``event_col`` is 1 only on the subject's terminal interval (if they churned). This reuses
        the canonical columns -- ``entry_tenure`` is the interval start, ``exit_tenure`` the
        interval stop -- so KM and the business outputs consume it unchanged (DV3-1 / A1).
        Repeated ``id_col`` values are expected here (id-uniqueness is not enforced).
        """
        covariate_cols = list(covariate_cols or [])
        group_cols = list(group_cols or [])
        required = [
            id_col,
            origin_col,
            interval_start_col,
            interval_end_col,
            event_col,
            *covariate_cols,
            *group_cols,
        ]
        _check_columns(df, required)

        data = df.reset_index(drop=True)
        origin = pd.to_datetime(data[origin_col], errors="raise")
        start = pd.to_datetime(data[interval_start_col], errors="raise")
        end = pd.to_datetime(data[interval_end_col], errors="raise")
        event = data[event_col].astype(int)

        _validate_intervals(data[id_col], origin, start, end, event)

        status_label = np.where(event.to_numpy() == 1, "churn", "active")

        return cls._assemble(
            data,
            ids=data[id_col],
            origin=origin,
            exit_date=end,
            event=event,
            status_label=status_label,
            group_cols=group_cols,
            covariate_cols=covariate_cols,
            interval=True,
            analysis_start_ts=None,
            event_observed_from_ts=None,
            entry_col=interval_start_col,  # entry_tenure = interval_start - origin
            includes_pre_entry_churners=None,
            time_unit=time_unit,
        )
