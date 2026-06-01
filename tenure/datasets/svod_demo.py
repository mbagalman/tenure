"""Synthetic SVOD demo dataset with a known data-generating process.

Lifetimes are exponential with a fixed mean, so survival, RMST, and a simple LTV have exact
closed forms exposed via :func:`svod_demo_truth`. ``with_left_truncation=True`` simulates a
Window-Cut study (events observed only from ``ANALYSIS_START`` onward), seeding the
left-truncation trap that TNR001 catches. Fully offline and deterministic given ``seed``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

COHORT_START = pd.Timestamp("2022-01-01")
ANALYSIS_START = pd.Timestamp("2024-01-01")
ACTIVE_AS_OF = pd.Timestamp("2026-05-31")

MEAN_LIFETIME_DAYS = 365.0
_MONTH_DAYS = 30.4375


@dataclass(frozen=True)
class SvodTruth:
    """Closed-form ground truth for the demo's generative model (exponential lifetimes)."""

    mean_lifetime_days: float = MEAN_LIFETIME_DAYS

    def survival_at(self, days: float) -> float:
        return float(math.exp(-days / self.mean_lifetime_days))

    def median_tenure_days(self) -> float:
        return float(self.mean_lifetime_days * math.log(2.0))

    def rmst_days(self, horizon_days: float) -> float:
        """Restricted mean survival time through ``horizon_days`` (area under S)."""
        m = self.mean_lifetime_days
        return float(m * (1.0 - math.exp(-horizon_days / m)))

    def expected_lifetime_months(self, horizon_days: float) -> float:
        return self.rmst_days(horizon_days) / _MONTH_DAYS

    def ltv(self, monthly_margin: float, horizon_days: float = 365.0) -> float:
        """Simple survival-weighted LTV = monthly margin x expected active months."""
        return float(monthly_margin * self.expected_lifetime_months(horizon_days))


def svod_demo_truth() -> SvodTruth:
    """Ground-truth constants for :func:`load_svod_demo`."""
    return SvodTruth()


def load_svod_demo(
    *,
    with_left_truncation: bool = True,
    seed: int = 0,
    n: int = 4000,
) -> pd.DataFrame:
    """Return a synthetic SVOD customer table (one row per customer).

    Columns: ``customer_id``, ``signup_date``, ``churn_date`` (NaT = still active at the
    ``2026-05-31`` snapshot), ``plan``, ``channel``.
    """
    rng = np.random.default_rng(seed)
    span_days = int((ACTIVE_AS_OF - COHORT_START).days)

    signup_offset = rng.integers(0, span_days, size=n)
    signup = pd.Series(COHORT_START + pd.to_timedelta(signup_offset, unit="D"))
    lifetime = rng.exponential(MEAN_LIFETIME_DAYS, size=n)
    churn = signup + pd.Series(pd.to_timedelta(lifetime, unit="D"))
    churn_date = churn.where(churn <= ACTIVE_AS_OF)  # NaT for survivors -> active

    df = pd.DataFrame(
        {
            "customer_id": [f"C{i:06d}" for i in range(n)],
            "signup_date": signup,
            "churn_date": churn_date,
            "plan": rng.choice(["basic", "standard", "premium"], size=n),
            "channel": rng.choice(["organic", "paid", "partner"], size=n),
        }
    )

    if with_left_truncation:
        # Window-Cut: events are only observed from ANALYSIS_START, so customers who churned
        # earlier are absent -- older survivors remain (a left-truncated sample).
        keep = ~(df["churn_date"].notna() & (df["churn_date"] < ANALYSIS_START))
    else:
        # Clean acquisition cohort: only customers acquired on/after ANALYSIS_START.
        keep = df["signup_date"] >= ANALYSIS_START

    return df[keep].reset_index(drop=True)
