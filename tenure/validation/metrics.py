"""Discrimination metrics for the validation layer (v0.4 Slice 2).

``concordance`` wraps lifelines' ``concordance_index`` (DV4-6, wrap-don't-reimplement) and evaluates
a model's per-subject risk against a ``TestCohort``'s post-cutoff outcomes (the eval clock). The
per-subject risk is obtained through a single dispatch (``_subject_risk``) so the metric stays
model-agnostic (A3):

- Cox-family (CoxPH / TimeVaryingCox): the partial hazard ``exp(beta^T x)`` on each subject's
  covariates AS OF the cutoff -- a horizon-free, proportional-hazards risk score.
- A ``SurvivalFunction`` / KaplanMeier: ``1 - S(horizon)``. Overall KM is constant across subjects,
  so its C-index is ~0.5 by construction (it predicts a cohort curve, not individual risk -- DV4-5).
- A raw per-subject risk array (advanced / testing).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from lifelines.utils import concordance_index

from tenure.exceptions import TenureValidationError
from tenure.outputs._common import as_survival
from tenure.validation.result import ValidationResult


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

    # lifelines scores concordance as higher-predicted => longer survival, so feed -risk.
    estimate = float(concordance_index(durations, -risk, events))

    train_design = getattr(model, "design", None)
    metadata = {
        "metric": "c_index",
        "estimate": estimate,
        "horizon": horizon,
        "prediction_time": test_cohort.prediction_time,
        "censoring_method": "none",
        "model_type": type(model).__name__,
        "n_train": int(train_design.n) if train_design is not None else None,
        "n_test": int(test_cohort.n),
        "warnings": [],
    }
    result_table = pd.DataFrame(
        [{"metric": "c_index", "estimate": estimate, "n_test": int(test_cohort.n)}]
    )
    return ValidationResult(table=result_table, metadata=metadata)
