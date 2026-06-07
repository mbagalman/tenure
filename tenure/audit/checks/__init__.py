"""Built-in audit checks. Importing this package registers each check."""

from __future__ import annotations

from tenure.audit.checks import (  # noqa: F401
    tnr001_left_truncation,
    tnr002_time_origin,
    tnr003_exit_mapping,
    tnr004_immortal_time,
)

__all__: list[str] = []
