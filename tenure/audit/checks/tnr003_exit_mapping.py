"""TNR003 -- Event/censoring mislabeling (and informative-censoring warning).

Applies to status-schema designs only (the event-date schema's event/active is unambiguous).
- BLOCK when statuses present in the data were not mapped (their rows were dropped, so their
  risk contribution is silently lost unless this is surfaced).
- WARN when a status is mapped to ``censored`` but its rows exit before the snapshot -- that is
  not ordinary administrative right-censoring and assumes the censoring is non-informative.

v0.1 prevents silent mislabeling and educates on informative censoring; it does not solve
competing risks (that is post-v0.4).
"""

from __future__ import annotations

from tenure.audit.base import AuditCheck
from tenure.audit.registry import register
from tenure.audit.report import CheckResult, Status


@register
class ExitMappingCheck(AuditCheck):
    id = "TNR003"
    title = "Event/censoring mislabeling"

    def evaluate(self, design) -> CheckResult | None:
        if getattr(design, "status_map", None) is None:
            return None  # event-date schema: event/active is unambiguous, nothing to map

        if design.unmapped_statuses:
            return CheckResult(
                self.id,
                Status.BLOCK,
                self.title,
                f"{design.n_unmapped} row(s) have status values absent from status_map: "
                f"{design.unmapped_statuses}. They were dropped, so their risk contribution is "
                "lost and your curves are biased.",
                remediation="Map every status to one of event/censored/exclude in status_map.",
                details={
                    "unmapped_statuses": list(design.unmapped_statuses),
                    "n_unmapped": design.n_unmapped,
                },
            )

        if design.informative_censoring_statuses:
            return CheckResult(
                self.id,
                Status.WARN,
                self.title,
                f"Status(es) {design.informative_censoring_statuses} are mapped to 'censored' but "
                "exit before the snapshot, so this is not ordinary administrative censoring. "
                "Treating it as independent censoring biases retention if those customers (e.g. "
                "upgraders/migrators) have a different churn risk.",
                remediation=(
                    "Confirm the censoring is non-informative, map them to 'exclude', or wait for "
                    "competing-risks (CIF) in a later release."
                ),
                details={"statuses": list(design.informative_censoring_statuses)},
            )

        return CheckResult(
            self.id,
            Status.PASS,
            self.title,
            "All statuses are explicitly mapped; no pre-snapshot ('informative') censoring found.",
        )
