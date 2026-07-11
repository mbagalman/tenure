"""Matplotlib survival-curve plotting (FR-RC-6/7).

Single or grouped Kaplan-Meier curves with confidence bands, an optional ggsurvplot-style
number-at-risk table aligned under the x-axis (D-S7), and a caveat stamp when the backing study
had audit warnings that were bypassed (FR-RC-7). Matplotlib only in v0.1; the full theme system
is deferred to v1.0.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from tenure.estimators.hybrid import HybridGroupCurve
from tenure.estimators.nelson_aalen import CumulativeHazardFunction
from tenure.estimators.survival import SurvivalFunction


def _resolve(estimator):
    """Return (SurvivalFunction, audit_report_or_None) from a result / estimator / curve."""
    if hasattr(estimator, "curves") and hasattr(estimator, "audit"):  # RetentionResult
        return estimator.curves, estimator.audit
    if hasattr(estimator, "survival_"):  # a fitted estimator (KaplanMeier)
        return estimator.survival_, None
    if isinstance(estimator, SurvivalFunction):
        return estimator, None
    raise TypeError(
        "plot_survival expects a RetentionResult, a fitted estimator (.survival_), or a "
        f"SurvivalFunction; got {type(estimator).__name__}."
    )


def plot_survival(
    estimator,
    *,
    ci: bool = True,
    at_risk: bool = True,
    ax=None,
    audit_report=None,
    figsize=(8, 5),
):
    """Plot survival curves and return the curve Axes.

    ``at_risk=True`` builds its own two-panel figure (curves + number-at-risk table) and so
    requires ``ax=None``. The caveat stamp is drawn only when ``audit_report`` (or the result's
    audit) has active warnings -- never on a clean design (preserves the no-crying-wolf invariant).
    """
    survival, auto_audit = _resolve(estimator)
    audit_report = audit_report if audit_report is not None else auto_audit
    groups = survival.groups

    if at_risk:
        if ax is not None:
            raise ValueError(
                "at_risk=True manages its own figure layout; pass ax=None (or at_risk=False)."
            )
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(2, 1, height_ratios=[4, 1], hspace=0.08)
        main_ax = fig.add_subplot(gs[0])
        table_ax = fig.add_subplot(gs[1], sharex=main_ax)
    else:
        main_ax = ax if ax is not None else plt.subplots(figsize=figsize)[1]
        table_ax = None
        fig = main_ax.figure

    any_hybrid = False
    for group in groups:
        curve = survival.curve(group)
        (line,) = main_ax.step(curve.times, curve.survival, where="post", label=group)
        if ci:
            main_ax.fill_between(
                curve.times,
                curve.ci_lower,
                curve.ci_upper,
                step="post",
                alpha=0.15,
                color=line.get_color(),
            )
        if isinstance(curve, HybridGroupCurve):  # mark where data ends and the model tail begins
            any_hybrid = True
            main_ax.axvline(
                curve.boundary, color=line.get_color(), linestyle=":", linewidth=1, alpha=0.6
            )
    if any_hybrid:
        fig.text(
            0.99,
            0.01,
            "Dotted line: data ends, model tail begins.",
            fontsize=7,
            color="dimgray",
            ha="right",
            va="bottom",
        )

    main_ax.set_ylim(0.0, 1.02)
    main_ax.set_ylabel("Survival probability")
    main_ax.set_xlabel(f"Tenure ({survival.time_unit})")
    if len(groups) > 1 or groups != ["overall"]:
        main_ax.legend(title="group", fontsize=9)

    if at_risk:
        _draw_at_risk_table(table_ax, main_ax, survival, groups)

    if audit_report is not None and audit_report.warnings:
        names = ", ".join(sorted({r.check_id for r in audit_report.warnings}))
        fig.text(
            0.01,
            0.01,
            f"Caveat: bypassed audit warning(s): {names}. See the study-design audit.",
            fontsize=7,
            color="firebrick",
            ha="left",
            va="bottom",
        )

    return main_ax


def _draw_at_risk_table(table_ax, main_ax, survival, groups) -> None:
    """Render number-at-risk per group at the curve's x-ticks, aligned via shared x-axis."""
    xlo, xhi = main_ax.get_xlim()
    ticks = [t for t in main_ax.get_xticks() if xlo <= t <= xhi]

    rows = list(reversed(groups))  # first group ends up on top
    table_ax.set_ylim(-0.5, len(rows) - 0.5)
    table_ax.set_yticks(range(len(rows)))
    table_ax.set_yticklabels(rows, fontsize=9)
    table_ax.tick_params(left=False)
    for spine in table_ax.spines.values():
        spine.set_visible(False)

    table_ax.set_xlabel(f"Tenure ({survival.time_unit})")
    plt.setp(main_ax.get_xticklabels(), visible=False)
    main_ax.set_xlabel("")
    table_ax.set_title("Number at risk", loc="left", fontsize=9)

    for row, group in enumerate(rows):
        curve = survival.curve(group)
        for tick in ticks:
            count = int(curve.n_at_risk_at(tick)[0])
            table_ax.text(tick, row, str(count), ha="center", va="center", fontsize=8)


def plot_log_log_survival(estimator, *, ax=None, figsize=(8, 5)):
    """Log-log survival plot: log(-log S(t)) vs log(tenure) per group.

    Parallel curves support the proportional-hazards assumption; crossing or diverging curves
    suggest a violation. A visual companion to ``CoxPH.proportional_hazards_test``.
    """
    survival, _ = _resolve(estimator)
    main_ax = ax if ax is not None else plt.subplots(figsize=figsize)[1]
    for group in survival.groups:
        curve = survival.curve(group)
        times, surv = curve.times, curve.survival
        mask = (times > 0) & (surv > 0) & (surv < 1)  # log(-log) undefined at S in {0, 1}
        if mask.any():
            main_ax.step(
                np.log(times[mask]), np.log(-np.log(surv[mask])), where="post", label=group
            )
    main_ax.set_xlabel(f"log(tenure [{survival.time_unit}])")
    main_ax.set_ylabel("log(-log S(t))")
    main_ax.set_title("Log-log survival (parallel => proportional hazards)")
    if survival.groups != ["overall"]:
        main_ax.legend(title="group", fontsize=9)
    return main_ax


def _resolve_hazard(estimator) -> CumulativeHazardFunction:
    if isinstance(estimator, CumulativeHazardFunction):
        return estimator
    hazard = getattr(estimator, "cumulative_hazard_", None)
    if isinstance(hazard, CumulativeHazardFunction):
        return hazard
    raise TypeError(
        "plot_cumulative_hazard expects a NelsonAalen estimator or a CumulativeHazardFunction; "
        f"got {type(estimator).__name__}."
    )


def plot_cumulative_hazard(estimator, *, ci: bool = True, ax=None, figsize=(8, 5)):
    """Plot Nelson-Aalen cumulative-hazard step curves (single or grouped) with CI bands."""
    hazard = _resolve_hazard(estimator)
    main_ax = ax if ax is not None else plt.subplots(figsize=figsize)[1]
    for group in hazard.groups:
        curve = hazard.curve(group)
        (line,) = main_ax.step(curve.times, curve.cumulative_hazard, where="post", label=group)
        if ci:
            main_ax.fill_between(
                curve.times,
                curve.ci_lower,
                curve.ci_upper,
                step="post",
                alpha=0.15,
                color=line.get_color(),
            )
    main_ax.set_xlabel(f"Tenure ({hazard.time_unit})")
    main_ax.set_ylabel("Cumulative hazard")
    main_ax.set_title("Nelson-Aalen cumulative hazard")
    if hazard.groups != ["overall"]:
        main_ax.legend(title="group", fontsize=9)
    return main_ax


def plot_calibration(result, *, ax=None, figsize=(6, 6)):
    """Plot a calibration (reliability) diagram from a ``calibration`` ValidationResult.

    Mean predicted survival (x) vs Kaplan-Meier observed survival (y), one point per bin sized by
    the bin's subject count, against the diagonal. Points on the diagonal == well calibrated.
    """
    table = result.table
    main_ax = ax if ax is not None else plt.subplots(figsize=figsize)[1]
    main_ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1, label="perfect")
    counts = table["n"].to_numpy(dtype=float)
    sizes = 20.0 + 180.0 * counts / counts.max() if counts.max() > 0 else 40.0
    main_ax.scatter(
        table["mean_predicted"], table["observed"], s=sizes, color="C0", zorder=3, label="bins"
    )
    main_ax.set_xlim(0.0, 1.0)
    main_ax.set_ylim(0.0, 1.0)
    main_ax.set_xlabel("Predicted survival")
    main_ax.set_ylabel("Observed survival (Kaplan-Meier)")
    horizon = result.metadata.get("horizon")
    error = result.metadata.get("calibration_error")
    title = "Calibration"
    if horizon is not None and error is not None:
        title = f"Calibration at horizon {horizon:.0f} (error {error:.3f})"
    main_ax.set_title(title)
    main_ax.legend(loc="lower right", fontsize=9)
    return main_ax
