"""Tenure: audit-first survival analysis for B2C customer churn.

The public surface is intentionally small in this Phase 0 slice. It will grow across the
v0.1 milestones (Kaplan-Meier, retention/LTV outputs, more audit checks).
"""

from __future__ import annotations

from tenure.audit import AuditReport, audit
from tenure.audit.report import CheckResult, Status
from tenure.datasets import load_svod_demo, svod_demo_truth
from tenure.estimators import KaplanMeier, SurvivalFunction
from tenure.exceptions import (
    AuditBlockedError,
    TenureError,
    TenureValidationError,
)
from tenure.outputs import (
    SummaryReport,
    retention_at,
    rmst,
    summarize,
    survival_weighted_ltv,
)
from tenure.study_design import StudyDesign

__version__ = "0.1.0.dev0"

__all__ = [
    "StudyDesign",
    "KaplanMeier",
    "SurvivalFunction",
    "retention_at",
    "rmst",
    "survival_weighted_ltv",
    "summarize",
    "SummaryReport",
    "audit",
    "AuditReport",
    "CheckResult",
    "Status",
    "AuditBlockedError",
    "TenureError",
    "TenureValidationError",
    "load_svod_demo",
    "svod_demo_truth",
    "__version__",
]
