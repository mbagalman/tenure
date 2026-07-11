"""The study-design audit: run every registered check and surface findings.

Surfacing contract (FR-BA-9): under ``strictness='block'`` a blocking finding raises
``AuditBlockedError`` *before* any number is computed. Under ``strictness='warn'`` blocking
findings are downgraded to warnings and the report is returned. The library never prints.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

from tenure.audit import checks as _checks  # noqa: F401  -- registers built-in checks
from tenure.audit.registry import registered_checks
from tenure.audit.report import AuditReport, CheckResult, Status
from tenure.exceptions import AuditBlockedError

if TYPE_CHECKING:
    from tenure.study_design import StudyDesign

__all__ = ["audit", "AuditReport", "CheckResult", "Status"]


def audit(design: StudyDesign, *, strictness: str = "block") -> AuditReport:
    """Audit a :class:`~tenure.study_design.StudyDesign` and return an :class:`AuditReport`.

    The hero entry point: run every registered check (TNR001-TNR004) against a design *before* any
    number is computed, classifying it block / warn / pass. Call this on your design before fitting.

    Under the default ``strictness="block"`` a blocking finding raises
    :class:`~tenure.exceptions.AuditBlockedError` immediately -- the dangerous design is the one
    that does not run. A clean (non-blocking) return marks the design audited so estimators may
    materialize it; a blocked design stays unfittable even if you catch the error, so a block cannot
    be bypassed. The library never prints -- render findings with :meth:`AuditReport.to_markdown`.

    A correctly designed cohort returns all-pass with zero warnings ("no crying wolf") -- a tested
    guarantee, not an aspiration.

    Args:
        design: The :class:`~tenure.study_design.StudyDesign` to check.
        strictness: ``"block"`` (default) raises ``AuditBlockedError`` on any blocking finding;
            ``"warn"`` downgrades blocking findings to warnings and returns the report (charts then
            carry a caveat stamp, FR-RC-7). Any other value raises ``ValueError``.

    Returns:
        An :class:`AuditReport` carrying every non-``None`` :class:`CheckResult`.

    Raises:
        AuditBlockedError: Under ``strictness="block"`` when a check blocks.
        ValueError: If ``strictness`` is not ``"block"`` or ``"warn"``.
    """
    if strictness not in ("block", "warn"):
        raise ValueError(f"strictness must be 'block' or 'warn', got {strictness!r}.")

    results: list[CheckResult] = []
    for check in registered_checks():
        result = check.evaluate(design)
        if result is None:
            continue
        if strictness == "warn" and result.status is Status.BLOCK:
            result = replace(result, status=Status.WARN)
        results.append(result)

    report = AuditReport(results=results, strictness=strictness)
    if report.blocks:  # only possible under strictness='block'
        raise AuditBlockedError(report)
    # Mark the design audited ONLY on a clean (non-blocking) return, so estimators may materialize
    # it even when it dropped unmapped-status rows -- the user has now been shown that finding
    # (TNR003). Marking before the raise would let a caller catch AuditBlockedError and then fit a
    # blocked design, bypassing ensure_estimable().
    if hasattr(design, "audited"):
        design.audited = True
    return report
