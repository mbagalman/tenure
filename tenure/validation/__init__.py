"""Validation layer (v0.4): honest out-of-time evaluation of Tenure's survival predictions.

A SEPARATE layer (A9) over a fitted model's predictions and a held-out design -- it never reaches
into estimator internals. ``temporal_holdout`` is the promoted default split (forward-in-time);
metrics (C-index, Brier/IBS) and calibration land in later v0.4 slices.
"""

from __future__ import annotations

from tenure.validation.calibration import calibration
from tenure.validation.metrics import brier, concordance, integrated_brier
from tenure.validation.result import (
    VAL001_RANDOM_SPLIT,
    VAL002_HORIZON_SUPPORT,
    VAL003_PANEL_LEAKAGE,
    ValidationResult,
)
from tenure.validation.split import TestCohort, random_split, temporal_holdout

__all__ = [
    "temporal_holdout",
    "random_split",
    "concordance",
    "brier",
    "integrated_brier",
    "calibration",
    "TestCohort",
    "ValidationResult",
    "VAL001_RANDOM_SPLIT",
    "VAL002_HORIZON_SUPPORT",
    "VAL003_PANEL_LEAKAGE",
]
