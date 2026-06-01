"""Exception hierarchy for Tenure."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tenure.audit.report import AuditReport


class TenureError(Exception):
    """Base class for all Tenure errors."""


class TenureValidationError(TenureError, ValueError):
    """Raised when user input violates a study-design or data contract."""


class AuditBlockedError(TenureError):
    """Raised when a blocking audit finding is present under ``strictness='block'``.

    The full :class:`~tenure.audit.report.AuditReport` is attached as ``.report`` so
    callers can inspect every finding and its remediation.
    """

    def __init__(self, report: AuditReport) -> None:
        self.report = report
        ids = ", ".join(r.check_id for r in report.blocks)
        super().__init__(
            f"Study-design audit blocked ({ids}). Inspect the report (report.to_markdown()) "
            "and fix the design, or rerun with strictness='warn' to proceed at your own risk."
        )
