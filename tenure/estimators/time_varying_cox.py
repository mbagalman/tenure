"""Time-varying Cox (wraps lifelines ``CoxTimeVaryingFitter``) for counting-process designs.

Fit on a ``StudyDesign.from_intervals`` carrying ``covariate_cols`` whose values may change between
a subject's intervals. The model reuses the canonical ``entry_tenure``/``exit_tenure`` columns as
each interval's (start, stop), so the v0.3 interval schema plugs straight in -- no new boundary.

Two things this buys beyond static Cox:
- coefficients + inference for covariates measured *as of each interval* (lifelines summary), and
- a per-interval, time-varying risk score (the partial hazard ratio ``exp(beta^T x)`` for that
  interval), so a subject's relative risk can rise and fall along its observed path.

Because a future-looking attribute can only enter a subject's design matrix on the interval where it
actually becomes true, this is how v0.3 *prevents* immortal-time bias structurally rather than only
warning about it (the demo proving that lands in Slice 3).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from lifelines import CoxTimeVaryingFitter

from tenure._frame import ENTRY, EVENT, EXIT, ID
from tenure.exceptions import TenureValidationError

_ID = "__id__"
_START = "__start__"
_STOP = "__stop__"
_EVENT = "__event__"


class TimeVaryingCox:
    """Time-varying Cox. Fit an interval design, then read ``summary`` / ``risk_scores``."""

    def __init__(self, penalizer: float = 0.0, l1_ratio: float = 0.0) -> None:
        self.penalizer = penalizer
        self.l1_ratio = l1_ratio
        self._fitter: CoxTimeVaryingFitter | None = None
        self._design = None

    def fit(self, design) -> TimeVaryingCox:
        if not getattr(design, "interval", False):
            raise TenureValidationError(
                "TimeVaryingCox requires an interval (counting-process) design; build it with "
                "StudyDesign.from_intervals(...)."
            )
        if not getattr(design, "covariate_cols", None):
            raise TenureValidationError(
                "TimeVaryingCox requires covariate_cols on the StudyDesign "
                "(build it with covariate_cols=[...])."
            )
        table = design.derive()
        frame = design.encode_covariates(table)
        frame[_ID] = table[ID].to_numpy()
        frame[_START] = table[ENTRY].to_numpy(dtype=float)
        frame[_STOP] = table[EXIT].to_numpy(dtype=float)
        frame[_EVENT] = table[EVENT].to_numpy(dtype=int)

        fitter = CoxTimeVaryingFitter(penalizer=self.penalizer, l1_ratio=self.l1_ratio)
        fitter.fit(frame, id_col=_ID, event_col=_EVENT, start_col=_START, stop_col=_STOP)
        self._fitter = fitter
        self._design = design
        return self

    def _require_fitted(self) -> None:
        if self._fitter is None:
            raise RuntimeError("TimeVaryingCox is not fitted yet; call .fit(design) first.")

    @property
    def fitter(self) -> CoxTimeVaryingFitter:
        """The underlying fitted lifelines CoxTimeVaryingFitter (params, summary, etc.)."""
        self._require_fitted()
        return self._fitter

    @property
    def design(self):
        """The StudyDesign this model was fit on."""
        self._require_fitted()
        return self._design

    @property
    def summary(self) -> pd.DataFrame:
        """Tidy coefficient table: covariate, coef, hazard_ratio (exp coef), and p_value."""
        self._require_fitted()
        s = self._fitter.summary
        return pd.DataFrame(
            {
                "covariate": [str(i) for i in s.index],
                "coef": s["coef"].to_numpy(dtype=float),
                "hazard_ratio": s["exp(coef)"].to_numpy(dtype=float),
                "p_value": s["p"].to_numpy(dtype=float),
            }
        )

    def encode_for_prediction(self, design) -> pd.DataFrame:
        """Encode a design's covariates, aligned to the fitted model's columns (missing -> 0)."""
        self._require_fitted()
        encoded = design.encode_covariates(design.derive())
        return encoded.reindex(columns=self._fitter.params_.index, fill_value=0.0)

    def risk_scores(self, design=None) -> pd.DataFrame:
        """Per-interval time-varying risk score (partial hazard ratio for each interval row).

        Returns one row per interval with its ``[interval_start, interval_stop)`` window and the
        partial hazard ratio ``exp(beta^T x)`` (relative to the training mean, matching lifelines).
        A subject's score can change across its intervals as its covariates change -- the
        time-varying analogue of ``CoxPH``'s single per-subject score.
        """
        self._require_fitted()
        design = design if design is not None else self._design
        table = design.derive()
        encoded = self.encode_for_prediction(design)
        log_ph = self._fitter.predict_log_partial_hazard(encoded).to_numpy(dtype=float)
        return pd.DataFrame(
            {
                "id": table[ID].to_numpy(),
                "interval_start": table[ENTRY].to_numpy(dtype=float),
                "interval_stop": table[EXIT].to_numpy(dtype=float),
                "risk_score": np.exp(log_ph),
            }
        )

    def __repr__(self) -> str:
        if self._fitter is None:
            return "TimeVaryingCox(unfitted)"
        return f"TimeVaryingCox(covariates={list(self._fitter.params_.index)})"
