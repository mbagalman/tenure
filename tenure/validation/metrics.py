"""Validation metrics: concordance (C-index) and the time-dependent Brier score / IBS.

``concordance`` wraps lifelines' ``concordance_index`` (wrap-don't-reimplement) and evaluates a
model's per-subject risk against a ``TestCohort``'s post-cutoff outcomes (the eval clock). The
per-subject risk is obtained through a single dispatch (``_subject_risk``) so the metric stays
model-agnostic:

- Cox-family (CoxPH / TimeVaryingCox): the partial hazard ``exp(beta^T x)`` on each subject's
  covariates AS OF the cutoff -- a horizon-free, proportional-hazards risk score.
- A ``SurvivalFunction`` / KaplanMeier: ``1 - S(horizon)``. Overall KM is constant across subjects,
  so its C-index is ~0.5 by construction (it predicts a cohort curve, not individual risk).
- A raw per-subject risk array (advanced / testing).

WHY the Brier score / IBS below are hand-rolled rather than wrapped from ``scikit-survival`` (the
usual reference for IPCW Brier/IBS): we keep Tenure's core dependency-light (numpy + lifelines) with
no compiled/solver extras. scikit-survival pulls a heavier stack including compiled solver
dependencies (e.g. ``ecos``/``osqp``); on this project's environment (Python 3.14) that install
failed building ``ecos`` from source, which also meant we could not run it locally as a test oracle.
Wheel availability for new Python versions changes over time, so this is a "stay light / don't
require compiled extras" choice, not a permanent installability claim. The IPCW Brier score and IBS
are implemented directly here and validated against hand-computed references and known properties
(no-censoring reduces to the plain Brier; a perfect model scores 0; a constant 0.5 predictor scores
0.25). If a scikit-survival cross-check is wanted, it can be run out-of-band in an environment where
it installs.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines.utils import concordance_index

from tenure._frame import ID
from tenure.estimators.survival import SurvivalFunction
from tenure.exceptions import TenureValidationError
from tenure.outputs._common import as_survival
from tenure.validation.result import VAL002_HORIZON_SUPPORT, ValidationResult


def _subject_risk(model, test_cohort, horizon) -> np.ndarray:
    """Per-subject risk (higher => churns sooner), dispatched by model type (A3)."""
    if isinstance(model, np.ndarray | list | pd.Series):
        return np.asarray(model, dtype=float)

    fitter = getattr(model, "fitter", None)
    design = getattr(model, "design", None)
    if fitter is not None and getattr(design, "covariate_cols", None):
        # Cox-family: partial hazard on covariates as of the cutoff (horizon-free under PH).
        encoded = design.encode_covariates(test_cohort.table).reindex(
            columns=fitter.params_.index, fill_value=0.0
        )
        return fitter.predict_partial_hazard(encoded).to_numpy(dtype=float)

    survival = as_survival(model)
    if horizon is None:
        raise TenureValidationError(
            "concordance on a survival-function / KaplanMeier model needs horizon=... "
            "(risk = 1 - S(horizon))."
        )
    groups = survival.groups
    if len(groups) == 1:
        s = float(survival.survival_at([float(horizon)], group=groups[0])["survival"].iloc[0])
        return np.full(test_cohort.n, 1.0 - s)
    raise TenureValidationError(
        "C-index for a grouped survival function needs a per-subject group mapping (not yet "
        "supported); use a Cox model for covariate-specific risk."
    )


def concordance(model, test_cohort, *, horizon: float | None = None) -> ValidationResult:
    """Harrell's concordance index (C-index) of ``model`` on a held-out ``TestCohort`` (DV4-6).

    ``model`` is a fitted Cox-family estimator, a ``SurvivalFunction`` / KaplanMeier (with a
    ``horizon``), or a raw per-subject risk array. Higher risk should mean shorter survival; the
    metric is computed on the cohort's eval clock (``eval_duration`` / ``eval_event``). 0.5 is
    random, 1.0 is perfect concordance. Returns a ``ValidationResult``.
    """
    risk = _subject_risk(model, test_cohort, horizon)
    table = test_cohort.table
    durations = table["eval_duration"].to_numpy(dtype=float)
    events = table["eval_event"].to_numpy(dtype=int)
    if len(risk) != len(durations):
        raise TenureValidationError(
            f"risk length ({len(risk)}) != test cohort size ({len(durations)})."
        )
    if not np.isfinite(risk).all():
        raise TenureValidationError(
            "risk scores must be finite (got NaN/inf); a covariate value as of the cutoff may be "
            "missing, or the model failed to converge."
        )

    # lifelines scores concordance as higher-predicted => longer survival, so feed -risk.
    try:
        estimate = float(concordance_index(durations, -risk, events))
    except ZeroDivisionError as exc:
        raise TenureValidationError(
            "C-index is undefined: the test cohort has no admissible (comparable) event pairs -- "
            "e.g. it is all-censored after the cutoff. Choose an earlier cutoff or a cohort that "
            "has post-cutoff churn."
        ) from exc

    train_design = getattr(model, "design", None)
    n_train_rows = int(train_design.n) if train_design is not None else None
    n_train_subjects = (
        int(train_design.canonical[ID].nunique()) if train_design is not None else None
    )
    metadata = {
        "metric": "c_index",
        "estimate": estimate,
        "horizon": horizon,
        "prediction_time": test_cohort.prediction_time,
        # Harrell's C handles right-censoring via admissible (comparable) pairs -- not ignored.
        "censoring_method": "right_censored_harrell",
        "model_type": type(model).__name__,
        "n_train_rows": n_train_rows,  # canonical rows (intervals for time-varying designs)
        "n_train_subjects": n_train_subjects,  # distinct customers
        "n_test": int(test_cohort.n),
        "warnings": [],
    }
    result_table = pd.DataFrame(
        [{"metric": "c_index", "estimate": estimate, "n_test": int(test_cohort.n)}]
    )
    return ValidationResult(table=result_table, metadata=metadata)


# --- Time-dependent Brier score + IBS (IPCW, hand-rolled -- see the module docstring for why) ---
#
# The IPCW Brier score (Graf et al. 1999) weights each test subject by the inverse probability of
# remaining uncensored, using a Kaplan-Meier estimate of the CENSORING distribution G. With no
# censoring G == 1 and it reduces to the plain Brier score. Predictions are CONDITIONAL survival on
# the eval clock (survival from each subject's tenure at the cutoff), so KM/Cox curves on the tenure
# clock are conditioned per subject before scoring.


def _step(col: np.ndarray, tindex: np.ndarray, q: float) -> float:
    """Right-continuous step lookup of a survival column at query time ``q`` (S=1 before t0)."""
    idx = int(np.searchsorted(tindex, q, side="right")) - 1
    return float(col[idx]) if idx >= 0 else 1.0


def _condition_columns(S, tindex, starts, times) -> tuple[np.ndarray, bool]:
    """Per-subject conditional survival S(start_i + t)/S(start_i) on the eval clock.

    Also reports ``beyond_support`` = whether any query ``start_i + t`` exceeded the fitted curve's
    last time, where the step lookup holds the last survival value flat (extrapolation, VAL002).
    """
    n, m = S.shape[1], len(times)
    max_support = float(tindex[-1]) if len(tindex) else 0.0
    est = np.empty((n, m))
    beyond_support = False
    for i in range(n):
        col = S[:, i]
        s_start = _step(col, tindex, starts[i])
        for j in range(m):
            q = starts[i] + times[j]
            if q > max_support:
                beyond_support = True
            est[i, j] = (_step(col, tindex, q) / s_start) if s_start > 0 else 0.0
    return np.clip(est, 0.0, 1.0), beyond_support


def _conditional_survival_matrix(model, test_cohort, times) -> tuple[np.ndarray, bool]:
    """(n_test, n_times) predicted survival on the eval clock, dispatched by model type (A3)."""
    starts = test_cohort.table["eval_start"].to_numpy(dtype=float)
    fitter = getattr(model, "fitter", None)
    design = getattr(model, "design", None)
    if (
        fitter is not None
        and hasattr(fitter, "predict_survival_function")
        and getattr(design, "covariate_cols", None)
    ):
        # Cox PH: per-subject survival on the tenure clock, then conditioned at the cutoff.
        encoded = design.encode_covariates(test_cohort.table).reindex(
            columns=fitter.params_.index, fill_value=0.0
        )
        surv = fitter.predict_survival_function(encoded)
        return _condition_columns(surv.to_numpy(), surv.index.to_numpy(dtype=float), starts, times)

    survival = model if isinstance(model, SurvivalFunction) else getattr(model, "survival_", None)
    if isinstance(survival, SurvivalFunction):
        if len(survival.groups) != 1:
            raise TenureValidationError(
                "Brier/IBS for a grouped survival function needs a per-subject group mapping; "
                "not yet supported."
            )
        curve = survival.curve(survival.groups[0])
        cohort = np.tile(curve.survival.reshape(-1, 1), (1, len(starts)))
        return _condition_columns(cohort, np.asarray(curve.times, dtype=float), starts, times)

    raise TenureValidationError(
        f"Brier/IBS support CoxPH and overall survival functions; got {type(model).__name__}. "
        "Time-varying / grouped Brier needs the full pre-cutoff covariate path and is not yet "
        "supported."
    )


_MODEL_SUPPORT_WARNING = (
    f"{VAL002_HORIZON_SUPPORT}: some predictions fall beyond the fitted model's tenure support "
    "(eval_start + time); survival is held flat past the last fitted time."
)


def _validate_times(times) -> np.ndarray:
    """Reject time grids that would silently corrupt the score: non-finite, non-positive, or not
    strictly increasing (which also rejects duplicates and unsorted grids -- the trapezoid IBS and
    the per-time scores assume a clean, ordered eval-clock grid)."""
    times = np.atleast_1d(np.asarray(times, dtype=float))
    if not np.isfinite(times).all():
        raise TenureValidationError("Brier times must be finite.")
    if (times <= 0.0).any():
        raise TenureValidationError(
            "Brier times must be > 0 (they are post-cutoff eval-clock durations)."
        )
    if times.size > 1 and not np.all(np.diff(times) > 0):
        raise TenureValidationError(
            "Brier times must be strictly increasing (sorted, with no duplicates)."
        )
    return times


def _supported_times(times: np.ndarray, durations: np.ndarray) -> tuple[np.ndarray, bool]:
    """Keep only eval times within the cohort's follow-up; warn VAL002 for any dropped."""
    max_follow = float(durations.max())
    valid = times[times < max_follow]
    dropped = len(valid) < len(times)
    if dropped:
        warnings.warn(
            f"{VAL002_HORIZON_SUPPORT}: requested Brier time(s) >= the supported follow-up "
            f"({max_follow:.1f}); those points were dropped.",
            UserWarning,
            stacklevel=3,
        )
    if valid.size == 0:
        raise TenureValidationError(
            f"all requested Brier times exceed the test cohort's follow-up ({max_follow:.1f}); "
            "choose earlier times."
        )
    return valid, dropped


def _ipcw_brier_scores(durations, events, estimate, times) -> np.ndarray:
    """Time-dependent IPCW Brier score at each ``times`` point (Graf et al. 1999)."""
    durations = np.asarray(durations, dtype=float)
    events = np.asarray(events, dtype=int)
    n = len(durations)
    g = KaplanMeierFitter().fit(durations, event_observed=(1 - events))  # KM of censoring
    # lifelines predict() returns a Series for many times but a scalar for one -- normalize both.
    g_dur = np.atleast_1d(np.asarray(g.predict(durations), dtype=float))  # G(T_i)
    g_t = np.atleast_1d(np.asarray(g.predict(times), dtype=float))  # G(t_j)
    w_event = np.where(g_dur > 0, 1.0 / np.where(g_dur > 0, g_dur, 1.0), 0.0)

    scores = np.empty(len(times))
    for j, t in enumerate(times):
        s_t = estimate[:, j]
        had_event_by_t = (durations <= t) & (events == 1)
        at_risk_after_t = durations > t
        gt = g_t[j]
        term_dead = (s_t**2) * had_event_by_t * w_event
        term_alive = ((1.0 - s_t) ** 2) * at_risk_after_t * (1.0 / gt if gt > 0 else 0.0)
        scores[j] = float((term_dead + term_alive).sum() / n)
    return scores


def _eval_arrays(test_cohort):
    table = test_cohort.table
    return (
        table["eval_duration"].to_numpy(dtype=float),
        table["eval_event"].to_numpy(dtype=int),
    )


def _brier_metadata(model, test_cohort, times, support_warning) -> dict:
    train_design = getattr(model, "design", None)
    return {
        "prediction_time": test_cohort.prediction_time,
        "censoring_method": "ipcw",
        "model_type": type(model).__name__,
        "times": [float(t) for t in times],
        "n_train_subjects": (
            int(train_design.canonical[ID].nunique()) if train_design is not None else None
        ),
        "n_test": int(test_cohort.n),
        "warnings": [VAL002_HORIZON_SUPPORT] if support_warning else [],
    }


def brier(model, test_cohort, times) -> ValidationResult:
    """Time-dependent IPCW Brier score of ``model`` on a held-out ``TestCohort`` at each time.

    Lower is better (0 = perfect). Predictions are conditional survival on the eval clock; censoring
    is handled by inverse-probability-of-censoring weighting. ``times`` must be finite, > 0, and
    strictly increasing. Times beyond the cohort's follow-up -- or beyond the fitted model's tenure
    support -- get a VAL002 warning. Returns a ``ValidationResult`` (``.table`` = [time, brier]).
    """
    times = _validate_times(times)
    durations, events = _eval_arrays(test_cohort)
    times, dropped = _supported_times(times, durations)
    estimate, beyond = _conditional_survival_matrix(model, test_cohort, times)
    if beyond:
        warnings.warn(_MODEL_SUPPORT_WARNING, UserWarning, stacklevel=2)
    scores = _ipcw_brier_scores(durations, events, estimate, times)
    metadata = {"metric": "brier", **_brier_metadata(model, test_cohort, times, dropped or beyond)}
    return ValidationResult(table=pd.DataFrame({"time": times, "brier": scores}), metadata=metadata)


def integrated_brier(model, test_cohort, times) -> ValidationResult:
    """Integrated Brier Score (IBS): the Brier score averaged over ``times`` (trapezoidal).

    A single-number summary of calibration+discrimination over the horizon; lower is better.
    Requires at least two supported time points. Returns a ``ValidationResult`` (``.estimate`` is
    the IBS).
    """
    times = _validate_times(times)
    if times.size < 2:
        raise TenureValidationError("integrated_brier needs at least 2 time points.")
    durations, events = _eval_arrays(test_cohort)
    times, dropped = _supported_times(times, durations)
    if times.size < 2:
        raise TenureValidationError("fewer than 2 supported time points remain for the IBS.")
    estimate, beyond = _conditional_survival_matrix(model, test_cohort, times)
    if beyond:
        warnings.warn(_MODEL_SUPPORT_WARNING, UserWarning, stacklevel=2)
    scores = _ipcw_brier_scores(durations, events, estimate, times)
    span = float(times[-1] - times[0])
    ibs = float(np.sum(np.diff(times) * (scores[:-1] + scores[1:]) / 2.0) / span)

    metadata = {
        "metric": "ibs",
        "estimate": ibs,
        **_brier_metadata(model, test_cohort, times, dropped or beyond),
    }
    table = pd.DataFrame(
        [{"metric": "ibs", "estimate": ibs, "t_min": float(times[0]), "t_max": float(times[-1])}]
    )
    return ValidationResult(table=table, metadata=metadata)
