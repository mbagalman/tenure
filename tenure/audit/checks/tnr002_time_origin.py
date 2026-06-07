"""TNR002 -- Time-origin / observation-window confusion.

The mistake: using the observation-window start (``analysis_start``) as t=0 instead of each
customer's true birth. Its data footprint is a large mass of origins sitting exactly on
``analysis_start``. A genuine same-date inception/launch cohort looks identical, so the block is
clearable via ``attest_origin_correct=True``.
"""

from __future__ import annotations

from tenure.audit.base import AuditCheck
from tenure.audit.registry import register
from tenure.audit.report import CheckResult, Status

# Fraction of origins sitting exactly on analysis_start above which window-as-origin is suspected.
_WINDOW_ORIGIN_THRESHOLD = 0.5


@register
class TimeOriginCheck(AuditCheck):
    id = "TNR002"
    title = "Time-origin / observation-window confusion"

    def evaluate(self, design) -> CheckResult | None:
        start = design.analysis_start
        if start is None:
            return None  # no observation-window reference to compare against

        frac_at_start = float((design.origin == start).mean())
        if frac_at_start < _WINDOW_ORIGIN_THRESHOLD:
            return CheckResult(
                self.id,
                Status.PASS,
                self.title,
                "Origins are not clustered at the observation-window start.",
            )

        details = {"frac_at_analysis_start": frac_at_start}
        if design.attest_origin_correct is True:
            return CheckResult(
                self.id,
                Status.PASS,
                self.title,
                f"{frac_at_start:.0%} of origins equal analysis_start, attested as genuine "
                "(e.g. a same-date launch cohort).",
                details=details,
            )

        return CheckResult(
            self.id,
            Status.BLOCK,
            self.title,
            f"{frac_at_start:.0%} of customers have origin exactly at analysis_start "
            f"({start.date()}). This is the signature of using the observation-window start as "
            "t=0 instead of true signup -- biased, because those customers had to survive to the "
            "window to appear.",
            remediation=(
                "Set origin to each customer's true birth (signup / first paid). If this really "
                "is a same-date inception cohort, pass attest_origin_correct=True."
            ),
            details=details,
        )
