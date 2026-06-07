"""Business outputs: retention at horizons, RMST, survival-weighted LTV, and a summary report.

All consume the backend-neutral SurvivalFunction (A8), never estimator internals.
"""

from __future__ import annotations

from tenure.outputs.ltv import survival_weighted_ltv
from tenure.outputs.retention import retention_at, rmst
from tenure.outputs.scoring import RiskScores, churn_risk_scores
from tenure.outputs.summary import SummaryReport, summarize

__all__ = [
    "retention_at",
    "rmst",
    "survival_weighted_ltv",
    "SummaryReport",
    "summarize",
    "churn_risk_scores",
    "RiskScores",
]
