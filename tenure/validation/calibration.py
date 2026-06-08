"""Calibration (reliability) of predicted survival at a horizon (v0.4 Slice 4, A10/DV4-9).

For a chosen ``horizon`` on the eval clock, each test subject gets a predicted survival probability;
subjects are grouped into quantile bins of that prediction, and within each bin the OBSERVED
survival is the Kaplan-Meier estimate at the horizon (so right-censoring is handled, not ignored). A
well-calibrated model lands on the diagonal -- predicted == observed. Returns a ``ValidationResult``
the reporting layer plots; compute stays separate from the plot (A10).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter

from tenure.exceptions import TenureValidationError
from tenure.validation.metrics import _conditional_survival_matrix
from tenure.validation.result import ValidationResult


def calibration(model, test_cohort, *, horizon: float, n_bins: int = 10) -> ValidationResult:
    """Predicted-vs-observed survival at ``horizon``, binned by predicted survival.

    ``.table`` has one row per bin: ``bin``, ``mean_predicted``, ``observed`` (KM at the horizon),
    and ``n``. ``.metadata['calibration_error']`` is the support-weighted mean
    ``|mean_predicted - observed|`` (0 = perfectly calibrated).
    """
    horizon = float(horizon)
    estimate, _n_extrap = _conditional_survival_matrix(model, test_cohort, np.array([horizon]))
    predicted = estimate[:, 0]
    durations = test_cohort.table["eval_duration"].to_numpy(dtype=float)
    events = test_cohort.table["eval_event"].to_numpy(dtype=int)

    frame = pd.DataFrame({"predicted": predicted, "duration": durations, "event": events})
    if frame["predicted"].nunique() < 2:
        # e.g. an overall KM where every subject shares an eval_start -> identical predictions.
        raise TenureValidationError(
            "predicted survival has too few distinct values to form calibration bins; use a model "
            "with varying predictions (e.g. CoxPH with covariates)."
        )
    bins = pd.qcut(frame["predicted"], q=n_bins, duplicates="drop")

    rows = []
    for k, (_label, group) in enumerate(frame.groupby(bins, observed=True)):
        kmf = KaplanMeierFitter().fit(
            group["duration"].to_numpy(), event_observed=group["event"].to_numpy()
        )
        observed = float(np.atleast_1d(np.asarray(kmf.predict(horizon)))[0])
        rows.append(
            {
                "bin": k,
                "mean_predicted": float(group["predicted"].mean()),
                "observed": observed,
                "n": int(len(group)),
            }
        )

    table = pd.DataFrame(rows)
    calibration_error = float(
        np.average(np.abs(table["mean_predicted"] - table["observed"]), weights=table["n"])
    )
    metadata = {
        "metric": "calibration",
        "horizon": horizon,
        "calibration_error": calibration_error,
        "model_type": type(model).__name__,
        "n_bins": int(len(table)),
        "prediction_time": test_cohort.prediction_time,
        "n_test": int(test_cohort.n),
        "warnings": [],
    }
    return ValidationResult(table=table, metadata=metadata)
