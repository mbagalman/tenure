"""Audit results: statuses, per-check findings, and the report object."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Status(str, Enum):
    PASS = "pass"
    WARN = "warn"
    BLOCK = "block"


@dataclass(frozen=True)
class CheckResult:
    """One audit check's finding. ``check_id`` is a stable public contract."""

    check_id: str
    status: Status
    title: str
    message: str
    remediation: str = ""
    details: dict = field(default_factory=dict)


@dataclass
class AuditReport:
    """The collected findings of a study-design audit."""

    results: list[CheckResult] = field(default_factory=list)
    strictness: str = "block"

    @property
    def blocks(self) -> list[CheckResult]:
        return [r for r in self.results if r.status is Status.BLOCK]

    @property
    def warnings(self) -> list[CheckResult]:
        return [r for r in self.results if r.status is Status.WARN]

    @property
    def passes(self) -> list[CheckResult]:
        return [r for r in self.results if r.status is Status.PASS]

    @property
    def ok(self) -> bool:
        """True when there are no blocking findings (safe to compute numbers)."""
        return not self.blocks

    @property
    def clean(self) -> bool:
        """True when nothing fired -- no blocks and no warnings (the clean-cohort case)."""
        return not self.blocks and not self.warnings

    def to_markdown(self) -> str:
        lines = [
            "# Study-design audit",
            "",
            f"- strictness: `{self.strictness}`",
            f"- blocks: {len(self.blocks)}  |  warnings: {len(self.warnings)}  "
            f"|  passes: {len(self.passes)}",
            "",
        ]
        if not self.results:
            lines.append("_No applicable checks fired._")
        for r in self.results:
            lines.append(f"## [{r.status.value.upper()}] {r.check_id} -- {r.title}")
            lines.append("")
            lines.append(r.message)
            if r.remediation:
                lines.append("")
                lines.append(f"**Remediation:** {r.remediation}")
            lines.append("")
        return "\n".join(lines).rstrip() + "\n"

    def __str__(self) -> str:
        return self.to_markdown()

    def __repr__(self) -> str:
        return (
            f"AuditReport(blocks={len(self.blocks)}, warnings={len(self.warnings)}, "
            f"passes={len(self.passes)}, strictness={self.strictness!r})"
        )
