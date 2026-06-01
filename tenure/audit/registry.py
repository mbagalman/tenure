"""The pluggable audit-check registry (ROADMAP A4).

Adding a check is additive: define an ``AuditCheck`` subclass, decorate it with
``@register``, and it is picked up by ``audit()`` -- no edits to the orchestrator.
"""

from __future__ import annotations

from tenure.audit.base import AuditCheck

_REGISTRY: list[AuditCheck] = []


def register(check_cls: type[AuditCheck]) -> type[AuditCheck]:
    """Class decorator: instantiate ``check_cls`` and add it to the registry."""
    _REGISTRY.append(check_cls())
    return check_cls


def registered_checks() -> list[AuditCheck]:
    """All registered checks, in registration order."""
    return list(_REGISTRY)
