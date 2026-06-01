"""The study-design audit: run every registered check and surface findings.

Surfacing contract (FR-BA-9): under ``strictness='block'`` a blocking finding raises
``AuditBlockedError`` *before* any number is computed. Under ``strictness='warn'`` blocking
findings are downgraded to warnings and the report is returned. The library never prints.
"""

from __future__ import annotations

from dataclasses import replace

from tenure.audit import checks as _checks  # noqa: F401  -- registers built-in checks
from tenure.audit.registry import registered_checks
from tenure.audit.report import AuditReport, CheckResult, Status
from tenure.exceptions import AuditBlockedError

__all__ = ["audit", "AuditReport", "CheckResult", "Status"]


def audit(design, *, strictness: str = "block") -> AuditReport:
    """Audit a :class:`~tenure.study_design.StudyDesign` and return an :class:`AuditReport`."""
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
    return report
