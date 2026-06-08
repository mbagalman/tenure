"""Validation result object + stable validation-time footgun IDs (DV4-1, DV4-8).

``ValidationResult`` is the backend-neutral contract every metric returns -- a tidy ``.table`` plus
a ``.metadata`` dict, aligned with ``SummaryReport`` / ``RiskScores`` so validation outputs travel
with their provenance (A6/A7). The ``VAL00x`` ids are validation-time guards, kept OUT of the
design-audit registry (which is design-time, TNR001-004) so a pasted validation report carries a
stable, citable code.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

# Stable validation-time footgun ids (DV4-8). Not TNR ids -- the design-audit registry stays
# design-time-only; these fire during validation/evaluation.
VAL001_RANDOM_SPLIT = "VAL001_RANDOM_SPLIT"  # survival validation should be forward-in-time
VAL002_HORIZON_SUPPORT = "VAL002_HORIZON_SUPPORT"  # eval horizon beyond supported follow-up
VAL003_PANEL_LEAKAGE = "VAL003_PANEL_LEAKAGE"  # a customer split across train/test (or folds)


@dataclass(frozen=True)
class ValidationResult:
    """One validation metric's outcome: a tidy table plus provenance metadata.

    ``metadata`` carries (per DV4-1): ``metric`` name, ``estimate``, ``horizon``/``times``,
    ``prediction_time`` (the calendar cutoff), ``censoring_method``, ``model_type``, ``n_train``,
    ``n_test``, ``audit_verdict`` (from the train design), and ``warnings`` (any VAL00x that fired).
    """

    table: pd.DataFrame
    metadata: dict = field(default_factory=dict)

    @property
    def estimate(self) -> float | None:
        """Convenience accessor for a single headline number, when the metric has one."""
        return self.metadata.get("estimate")

    def __repr__(self) -> str:
        metric = self.metadata.get("metric", "?")
        return f"ValidationResult(metric={metric!r}, estimate={self.metadata.get('estimate')})"
