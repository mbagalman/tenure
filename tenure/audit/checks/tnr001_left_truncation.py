"""TNR001 -- Left-truncation / delayed entry.

Keyed on observation completeness back to origin, NOT a naive date comparison. The subtle
distinction (Full Historical Cohort vs Window-Cut study): an old cohort with event history
complete to origin is unbiased; one whose events are only observed from a later date must be
modeled with delayed entry, or retention and LTV are biased upward.
"""

from __future__ import annotations

from tenure.audit.base import AuditCheck
from tenure.audit.registry import register
from tenure.audit.report import CheckResult, Status


@register
class LeftTruncationCheck(AuditCheck):
    id = "TNR001"
    title = "Left-truncation / delayed entry"

    def evaluate(self, design) -> CheckResult | None:
        # Delayed entry modeled (entry_col or event_observed_from) -> risk sets are correct.
        if design.entry_modeled:
            return CheckResult(
                self.id,
                Status.PASS,
                self.title,
                "Delayed entry is modeled; subjects join the risk set at their true tenure.",
            )

        analysis_start = design.analysis_start
        if analysis_start is None:
            # No observation-window signal and no modeling: nothing to detect from dates.
            return None

        n_pre = int((design.origin < analysis_start).sum())
        if n_pre == 0:
            return CheckResult(
                self.id,
                Status.PASS,
                self.title,
                "All origins fall on or after analysis_start; no left-truncation.",
            )

        incl = design.includes_pre_entry_churners
        start = analysis_start.date()
        details = {"n_pre_window": n_pre, "analysis_start": str(start)}

        if incl is True:
            return CheckResult(
                self.id,
                Status.PASS,
                self.title,
                f"{n_pre} customers predate analysis_start ({start}), but the population is "
                "complete back to origin (Full Historical Cohort) -- no truncation.",
                details=details,
            )

        if incl is False:
            return CheckResult(
                self.id,
                Status.BLOCK,
                self.title,
                f"{n_pre} customers have origins before analysis_start ({start}) and "
                "pre-entry churners are excluded, but delayed entry is not modeled. Their "
                "risk sets are wrong and retention/LTV will be biased upward.",
                remediation=(
                    "Model these customers with delayed entry -- set event_observed_from "
                    "(the date event recording became reliable) or entry_col so they enter "
                    "the risk set at their true tenure -- or, if event history is genuinely "
                    "complete back to origin, set includes_pre_entry_churners=True."
                ),
                details=details,
            )

        # incl is None -> completeness unknown; warn and ask for attestation.
        return CheckResult(
            self.id,
            Status.WARN,
            self.title,
            f"{n_pre} customers have origins before analysis_start ({start}). If their "
            "pre-window churns are unobserved, retention/LTV is biased upward.",
            remediation=(
                "Confirm whether event history is complete back to origin "
                "(set includes_pre_entry_churners), or model delayed entry via "
                "event_observed_from / entry_col."
            ),
            details=details,
        )
