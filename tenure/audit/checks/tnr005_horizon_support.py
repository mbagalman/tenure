"""TNR005 -- Weak / over-extrapolated horizon support.

Unlike TNR001-004 (design-time checks), TNR005 is evaluated at OUTPUT time: it depends on the
fitted curve's support (last event time, at-risk counts), so it is not registered in the
design-audit registry. The output/summary layer calls ``evaluate_horizon_support`` and surfaces
the findings alongside the numbers.
"""

from __future__ import annotations

from tenure.audit.report import CheckResult, Status

CHECK_ID = "TNR005"
TITLE = "Weak / over-extrapolated horizon support"


def evaluate_horizon_support(survival, horizons, *, min_at_risk: int = 10) -> list[CheckResult]:
    """One WARN finding per (group, horizon) whose requested horizon exceeds its support."""
    findings: list[CheckResult] = []
    for group in survival.groups:
        curve = survival.curve(group)
        for horizon in horizons:
            h = float(horizon)
            h_eff = curve.effective_horizon(h, min_at_risk)
            if h_eff < h - 1e-9:
                findings.append(
                    CheckResult(
                        CHECK_ID,
                        Status.WARN,
                        TITLE,
                        f"Group {group!r}: requested horizon {h:g} exceeds the supported horizon "
                        f"{h_eff:.1f} (capped by the last event time / at-risk >= {min_at_risk}). "
                        "The estimate rests on a small/empty risk set or the flat KM tail; the "
                        "output is truncated and relabeled to the supported horizon.",
                        remediation=(
                            "Shorten the horizon, pool groups, or accept the caveat explicitly."
                        ),
                        details={"group": group, "requested": h, "effective": h_eff},
                    )
                )
    return findings
