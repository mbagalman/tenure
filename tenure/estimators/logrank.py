"""Log-rank test for group comparison (left-truncation aware).

Tests the null hypothesis that two or more groups share the same survival distribution. This is
the standard inferential companion to grouped Kaplan-Meier: the curves *look* different, but is
the gap more than sampling noise?

Why hand-rolled rather than wrapping lifelines: lifelines' ``multivariate_logrank_test`` does not
accept entry times, so it cannot honor delayed entry (left truncation) -- the exact bias Tenure
exists to handle. The statistic here builds each event-time risk set from ``entry < t <= exit``,
so a delayed-entry cohort is compared correctly. With no delayed entry the risk sets reduce to
``exit >= t`` and the statistic matches lifelines exactly (a reference-match test pins this).

Standard (unweighted) log-rank only; weighted variants (Wilcoxon, Tarone-Ware) are not offered.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import chi2

from tenure._frame import as_estimator_frame, ensure_estimable
from tenure.estimators.kaplan_meier import _group_labels
from tenure.exceptions import TenureValidationError


@dataclass
class LogRankReport:
    """Result of a multivariate log-rank test: a tidy per-group table plus the test outcome.

    ``table`` has one row per group with columns ``group``, ``n`` (subjects), ``observed``
    (events), and ``expected`` (events under the equal-survival null). A group whose observed
    events fall well below expected retained better than the pooled average.
    """

    table: pd.DataFrame
    test_statistic: float
    p_value: float
    degrees_of_freedom: int

    def significant(self, alpha: float = 0.05) -> bool:
        """True if the group survival curves differ at level ``alpha`` (reject equal-survival)."""
        return self.p_value < alpha

    @property
    def summary(self) -> str:
        verdict = "differ" if self.significant() else "not distinguishable"
        return (
            f"log-rank: chi2={self.test_statistic:.3f}, df={self.degrees_of_freedom}, "
            f"p={self.p_value:.4g} -- groups {verdict} at alpha=0.05"
        )

    def __repr__(self) -> str:
        return (
            f"LogRankReport(test_statistic={self.test_statistic:.4f}, "
            f"p_value={self.p_value:.4g}, degrees_of_freedom={self.degrees_of_freedom})"
        )


def _logrank_statistic(
    entry: np.ndarray,
    duration: np.ndarray,
    event: np.ndarray,
    group_index: np.ndarray,
    n_groups: int,
) -> tuple[np.ndarray, np.ndarray, float, int, float]:
    """Multivariate log-rank over ``n_groups`` groups; returns (observed, expected, stat, df, p).

    Risk sets are ``entry < t <= exit`` so the test honors delayed entry. Variance uses the
    hypergeometric (multiple-groups) form with the standard ``(Y - d)/(Y - 1)`` tie correction.

    Vectorized to O((rows + events) log rows) (v1.0 performance pass -- the previous per-event
    boolean-mask loop was O(events x rows) and took ~a minute at 1e5 rows with continuous
    tenures): per-group at-risk counts at every event time come from two ``searchsorted`` calls
    on pre-sorted entry/duration arrays -- ``Y_g(t) = #{entry_g < t} - #{duration_g < t}`` is
    exactly the ``entry < t <= duration`` risk set -- death counts accumulate with ``np.add.at``,
    and the covariance sum collapses to einsum. The math is identical; the lifelines
    reference-match tests (1e-9) pin it.
    """
    event_times = np.unique(duration[event == 1])
    n_times = len(event_times)

    # Y[t, g] = per-group number at risk at each event time (entry < t and duration >= t).
    y = np.empty((n_times, n_groups))
    for g in range(n_groups):
        in_g = group_index == g
        entries_g = np.sort(entry[in_g])
        durations_g = np.sort(duration[in_g])
        y[:, g] = np.searchsorted(entries_g, event_times, side="left") - np.searchsorted(
            durations_g, event_times, side="left"
        )

    # D[t, g] = per-group deaths at each event time.
    d = np.zeros((n_times, n_groups))
    ev = event == 1
    t_idx = np.searchsorted(event_times, duration[ev])
    np.add.at(d, (t_idx, group_index[ev]), 1.0)

    y_total = y.sum(axis=1)
    d_total = d.sum(axis=1)
    valid = (y_total > 0) & (d_total > 0)
    yv, dv = y[valid], d[valid]
    ytv, dtv = y_total[valid], d_total[valid]

    observed = dv.sum(axis=0)
    expected = (dtv[:, None] * yv / ytv[:, None]).sum(axis=0)

    # covariance = sum_t factor_t * (Ytot_t * diag(y_t) - outer(y_t, y_t)) / Ytot_t^2
    var_rows = ytv > 1
    w = np.zeros(len(ytv))
    w[var_rows] = (
        dtv[var_rows] * (ytv[var_rows] - dtv[var_rows]) / (ytv[var_rows] - 1.0) / ytv[var_rows] ** 2
    )
    covariance = np.diag((w * ytv * yv.T).sum(axis=1)) - np.einsum("t,ti,tj->ij", w, yv, yv)

    # Drop one group for identifiability, then form the chi-square quadratic.
    z = (observed - expected)[:-1]
    v = covariance[:-1, :-1]
    statistic = float(z @ np.linalg.pinv(v) @ z)
    # Degrees of freedom = the covariance matrix's actual rank, NOT n_groups - 1 (review fix):
    # a group never at risk at any event time (e.g. all censored before the first event)
    # contributes zero variance, the rank drops, and testing the lower-rank statistic against
    # chi2_{K-1} inflates the p-value (overly conservative). pinv already handles the quadratic;
    # the reference distribution must match its rank. The tolerance is scaled to the matrix
    # magnitude: rank deficiency here comes from analytic cancellations (row sums are zero over
    # the at-risk groups) that floating-point accumulation leaves as ~1e-10 residuals, far above
    # matrix_rank's machine-epsilon default but far below any genuine variance dimension.
    df = int(np.linalg.matrix_rank(v, tol=1e-8 * max(1.0, float(np.abs(v).max()))))
    if df == 0:
        raise TenureValidationError(
            "log-rank is undefined: no two groups are ever at risk together at an event time "
            "(zero-rank covariance). Check group sizes and censoring."
        )
    p_value = float(chi2.sf(statistic, df))
    return observed, expected, statistic, df, p_value


def logrank_test(data: Any, *, by: str | list[str]) -> LogRankReport:
    """Log-rank test comparing survival across the ``by`` groups of a study design.

    The inferential companion to ``KaplanMeier.fit(..., by=...)``: it groups the same way and
    honors the same delayed entry, then tests whether the per-group survival curves differ.

    Args:
        data: A :class:`~tenure.study_design.StudyDesign` (or its derived canonical table). An
            unaudited design that dropped unmapped-status rows is refused (as for any estimator).
        by: Grouping column(s) -- a name or list of names, declared via ``group_cols`` on the
            design. Must resolve to at least two groups.

    Returns:
        A :class:`LogRankReport` with the per-group observed/expected table, the chi-square
        statistic, degrees of freedom, and the p-value.

    Raises:
        TenureValidationError: If ``by`` resolves to fewer than two groups, or no two groups are
            ever at risk together at an event time (zero-rank covariance).
    """
    ensure_estimable(data)
    table = data.derive() if hasattr(data, "derive") else data
    labels, order = _group_labels(table, by)
    if len(order) < 2:
        raise TenureValidationError(
            f"log-rank needs at least two groups to compare; by={by!r} produced {order}. "
            "Pass a grouping column with two or more levels."
        )

    ef = as_estimator_frame(table)
    group_pos = {label: k for k, label in enumerate(order)}
    group_index = np.array([group_pos[label] for label in labels.to_numpy()], dtype=int)

    observed, expected, statistic, df, p_value = _logrank_statistic(
        ef.entry, ef.duration, ef.event, group_index, len(order)
    )

    n = np.array([int((group_index == k).sum()) for k in range(len(order))])
    report_table = pd.DataFrame(
        {
            "group": order,
            "n": n,
            "observed": observed,
            "expected": expected,
        }
    )
    return LogRankReport(
        table=report_table,
        test_statistic=statistic,
        p_value=p_value,
        degrees_of_freedom=df,
    )
