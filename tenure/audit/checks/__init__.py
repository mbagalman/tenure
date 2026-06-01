"""Built-in audit checks. Importing this package registers each check."""

from __future__ import annotations

from tenure.audit.checks import tnr001_left_truncation  # noqa: F401

__all__: list[str] = []
