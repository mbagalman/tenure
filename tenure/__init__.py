"""Tenure: audit-first survival analysis for B2C customer churn.

The public surface is intentionally small in this Phase 0 slice. It will grow across the
v0.1 milestones (Kaplan-Meier, retention/LTV outputs, more audit checks).
"""

from __future__ import annotations

from tenure.audit import AuditReport, audit
from tenure.audit.report import CheckResult, Status
from tenure.datasets import load_svod_demo, svod_demo_truth
from tenure.demo import naive_vs_corrected_demo, naive_vs_corrected_immortal_demo
from tenure.estimators import (
    CoxDiagnosticReport,
    CoxPH,
    CumulativeHazardFunction,
    KaplanMeier,
    LogRankReport,
    NelsonAalen,
    ParametricSurvival,
    SurvivalFunction,
    TimeVaryingCox,
    logrank_test,
)
from tenure.exceptions import (
    AuditBlockedError,
    TenureError,
    TenureValidationError,
)
from tenure.landmark import landmark
from tenure.outputs import (
    RiskScores,
    SummaryReport,
    churn_risk_scores,
    retention_at,
    rmst,
    summarize,
    survival_weighted_ltv,
)
from tenure.plotting import (
    plot_calibration,
    plot_cumulative_hazard,
    plot_log_log_survival,
    plot_survival,
)
from tenure.study_design import StudyDesign
from tenure.validation import (
    TestCohort,
    ValidationResult,
    brier,
    calibration,
    concordance,
    integrated_brier,
    random_split,
    temporal_holdout,
)
from tenure.workflow import RetentionResult, RetentionStudy

__version__ = "0.4.0"

__all__ = [
    "StudyDesign",
    "RetentionStudy",
    "RetentionResult",
    "KaplanMeier",
    "NelsonAalen",
    "CoxPH",
    "CoxDiagnosticReport",
    "TimeVaryingCox",
    "ParametricSurvival",
    "SurvivalFunction",
    "CumulativeHazardFunction",
    "logrank_test",
    "LogRankReport",
    "retention_at",
    "rmst",
    "survival_weighted_ltv",
    "summarize",
    "SummaryReport",
    "churn_risk_scores",
    "RiskScores",
    "plot_survival",
    "plot_log_log_survival",
    "plot_cumulative_hazard",
    "audit",
    "AuditReport",
    "CheckResult",
    "Status",
    "AuditBlockedError",
    "TenureError",
    "TenureValidationError",
    "load_svod_demo",
    "svod_demo_truth",
    "naive_vs_corrected_demo",
    "naive_vs_corrected_immortal_demo",
    "landmark",
    "temporal_holdout",
    "random_split",
    "concordance",
    "brier",
    "integrated_brier",
    "calibration",
    "plot_calibration",
    "TestCohort",
    "ValidationResult",
    "__version__",
]
