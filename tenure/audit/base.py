"""Base class for audit checks.

Every check is a small, self-contained unit (AD-6 / ROADMAP A4). It declares a stable
``id`` and ``title`` and implements ``evaluate``, returning a :class:`CheckResult` with its
intrinsic status, or ``None`` when the check does not apply to the given design.
"""

from __future__ import annotations

from tenure.audit.report import CheckResult


class AuditCheck:
    id: str = ""
    title: str = ""

    def evaluate(self, design) -> CheckResult | None:  # noqa: D401
        raise NotImplementedError
