"""The headline demo: the LTV dollar gap from mishandled left-truncation.

This is the adoption hook. On the synthetic SVOD dataset (which excludes customers who churned
before the observation window opened -- a Window-Cut study), two pipelines are run that differ
in EXACTLY one thing: whether delayed entry is modeled.

- naive: signup treated as the origin with no delayed-entry handling, so the old survivors are
  (wrongly) at risk from tenure 0. Their early hazard is diluted -> survival and LTV inflated.
  The study-design audit flags this (TNR001).
- corrected: delayed entry modeled via ``event_observed_from`` (the date observation began), so
  each customer enters the risk set at their true tenure. This recovers the unbiased curve.

The gap (naive_ltv - corrected_ltv) is the dollars a naive analysis would over-state per
customer. It is deterministic for a given seed, so it doubles as a regression gate (NFR-CORR-3).
"""

from __future__ import annotations

from tenure.audit import audit
from tenure.datasets import ANALYSIS_START, load_svod_demo, svod_demo_truth
from tenure.estimators import KaplanMeier
from tenure.outputs import survival_weighted_ltv
from tenure.study_design import StudyDesign

_DEMO_COLS = {
    "id_col": "customer_id",
    "origin_col": "signup_date",
    "churn_date_col": "churn_date",
}


def _overall_ltv(design, *, period_margin, horizon):
    km = KaplanMeier().fit(design)
    return float(
        survival_weighted_ltv(km, period_margin=period_margin, horizon=horizon).iloc[0]["ltv"]
    )


def naive_vs_corrected_demo(
    df=None,
    *,
    seed: int = 0,
    period_margin: float = 12.0,
    horizon: float = 365.0,
):
    """Run the naive and corrected pipelines and return the LTV gap.

    Returns a dict: ``naive_ltv``, ``corrected_ltv``, ``ltv_dollar_diff`` (naive - corrected),
    ``true_ltv`` (the demo's closed-form ground truth), and ``audit`` (the naive design's audit,
    which flags the left-truncation that causes the gap).
    """
    if df is None:
        df = load_svod_demo(with_left_truncation=True, seed=seed)

    snapshot = "2026-05-31"

    # Naive: ignore left-truncation (no delayed entry); the audit warns about TNR001.
    naive_design = StudyDesign.from_event_dates(
        df,
        **_DEMO_COLS,
        active_as_of=snapshot,
        analysis_start=ANALYSIS_START,
        includes_pre_entry_churners=False,
    )
    naive_audit = audit(naive_design, strictness="warn")
    naive_ltv = _overall_ltv(naive_design, period_margin=period_margin, horizon=horizon)

    # Corrected: model delayed entry from when observation actually began.
    corrected_design = StudyDesign.from_event_dates(
        df,
        **_DEMO_COLS,
        active_as_of=snapshot,
        analysis_start=ANALYSIS_START,
        event_observed_from=ANALYSIS_START,
    )
    corrected_ltv = _overall_ltv(corrected_design, period_margin=period_margin, horizon=horizon)

    true_ltv = svod_demo_truth().ltv(period_margin, horizon_days=horizon)

    return {
        "naive_ltv": naive_ltv,
        "corrected_ltv": corrected_ltv,
        "ltv_dollar_diff": naive_ltv - corrected_ltv,
        "true_ltv": true_ltv,
        "audit": naive_audit,
    }
