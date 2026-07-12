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

import numpy as np
import pandas as pd

from tenure.audit import audit
from tenure.datasets import ANALYSIS_START, load_svod_demo, svod_demo_truth
from tenure.estimators import CoxPH, KaplanMeier, TimeVaryingCox
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


# --- The immortal-time prevention demo (v0.3) -------------------------------------------------
#
# The time-varying analogue of the left-truncation demo above. A future-looking attribute
# ("ever upgraded") can only be acquired by customers who survive long enough to acquire it, so a
# naive STATIC analysis that classifies everyone by their final ever-upgraded status credits the
# upgraded group with the immortal person-time before the upgrade -- producing an illusory
# protective effect even when the upgrade truly has NO effect on churn.
#
# The fix is the v0.3 data model: encode the upgrade as a time-varying covariate that is 0 before
# the upgrade and 1 after. The pre-upgrade person-time then sits in the X=0 risk set, so the bias
# cannot arise. The corrected (time-varying) hazard ratio recovers the truth (HR ~ 1); the naive
# (static "ever-upgraded") hazard ratio shows the spurious protection (HR << 1). Deterministic for
# a given seed, so it doubles as a regression gate.

_IMMORTAL_ORIGIN = pd.Timestamp("2022-01-01")


def _simulate_immortal_cohort(n, seed, *, lam, landmark, upgrade_prob, admin):
    """Synthesize a cohort where 'upgrade' is acquirable only after surviving to ``landmark``.

    The upgrade has NO causal effect on the churn hazard (true HR = 1). Returns two views of the
    SAME customers: a per-subject static frame (with the final ``ever_upgraded`` flag) and a
    counting-process interval frame (with the time-varying ``upgraded`` covariate).
    """
    rng = np.random.default_rng(seed)
    churn_time = rng.exponential(1.0 / lam, size=n)
    static_rows = []
    interval_rows = []
    for i in range(n):
        cid = f"c{i}"
        t = float(churn_time[i])
        churned = t <= admin
        obs_end = int(np.ceil(t)) if churned else int(admin)
        event = 1 if churned else 0
        upgraded = (t > landmark) and bool(rng.random() < upgrade_prob)

        static_rows.append(
            {
                "customer_id": cid,
                "signup_date": _IMMORTAL_ORIGIN,
                "churn_date": _IMMORTAL_ORIGIN + pd.Timedelta(days=obs_end) if event else pd.NaT,
                "ever_upgraded": int(upgraded),
            }
        )

        if upgraded:
            split = _IMMORTAL_ORIGIN + pd.Timedelta(days=int(landmark))
            interval_rows.append(
                {
                    "cid": cid,
                    "origin": _IMMORTAL_ORIGIN,
                    "start": _IMMORTAL_ORIGIN,
                    "end": split,
                    "event": 0,
                    "upgraded": 0,
                }
            )
            interval_rows.append(
                {
                    "cid": cid,
                    "origin": _IMMORTAL_ORIGIN,
                    "start": split,
                    "end": _IMMORTAL_ORIGIN + pd.Timedelta(days=obs_end),
                    "event": event,
                    "upgraded": 1,
                }
            )
        else:
            interval_rows.append(
                {
                    "cid": cid,
                    "origin": _IMMORTAL_ORIGIN,
                    "start": _IMMORTAL_ORIGIN,
                    "end": _IMMORTAL_ORIGIN + pd.Timedelta(days=obs_end),
                    "event": event,
                    "upgraded": 0,
                }
            )
    return pd.DataFrame(static_rows), pd.DataFrame(interval_rows)


def naive_vs_corrected_immortal_demo(
    *,
    n: int = 5000,
    seed: int = 0,
    landmark_time: float = 60.0,
    upgrade_prob: float = 0.5,
    admin: float = 365.0,
    median_tenure: float = 180.0,
):
    """Run the naive (static) and corrected (time-varying) pipelines on the same upgrade cohort.

    Returns a dict: ``true_hazard_ratio`` (1.0 by construction), ``naive_hazard_ratio`` (static
    'ever-upgraded' -- shows the illusory protective effect), ``corrected_hazard_ratio``
    (time-varying -- recovers truth), the raw ``naive_coef``/``corrected_coef``, and the two audits
    (``naive_audit`` warns TNR004; ``corrected_audit`` passes it, the bias being prevented).
    """
    lam = np.log(2.0) / median_tenure
    static_df, interval_df = _simulate_immortal_cohort(
        n, seed, lam=lam, landmark=landmark_time, upgrade_prob=upgrade_prob, admin=admin
    )

    # Naive: one static row per subject, classified by the FINAL ever-upgraded flag.
    naive_design = StudyDesign.from_event_dates(
        static_df,
        id_col="customer_id",
        origin_col="signup_date",
        churn_date_col="churn_date",
        active_as_of=_IMMORTAL_ORIGIN + pd.Timedelta(days=int(admin)),
        covariate_cols=["ever_upgraded"],
    )
    naive_cox = CoxPH().fit(naive_design)
    naive_coef = float(naive_cox.fitter.params_["ever_upgraded"])

    # Corrected: the upgrade is a time-varying covariate (0 before, 1 after) on interval rows.
    corrected_design = StudyDesign.from_intervals(
        interval_df,
        id_col="cid",
        origin_col="origin",
        interval_start_col="start",
        interval_end_col="end",
        event_col="event",
        covariate_cols=["upgraded"],
    )
    corrected_cox = TimeVaryingCox().fit(corrected_design)
    corrected_coef = float(corrected_cox.fitter.params_["upgraded"])

    return {
        "true_hazard_ratio": 1.0,
        "naive_hazard_ratio": float(np.exp(naive_coef)),
        "corrected_hazard_ratio": float(np.exp(corrected_coef)),
        "naive_coef": naive_coef,
        "corrected_coef": corrected_coef,
        "naive_audit": audit(naive_design, strictness="warn"),
        "corrected_audit": audit(corrected_design, strictness="warn"),
    }
