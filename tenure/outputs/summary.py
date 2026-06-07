"""SummaryReport: a self-documenting retention + LTV summary (FR-BO-4/5).

Metadata (units, horizons, currency, period, effective horizons, audit verdict) lives on the
report object -- not jammed into DataFrame cells -- so a stakeholder reading the table never
sees dollars divorced from their caveats.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from tenure.audit.checks.tnr005_horizon_support import evaluate_horizon_support
from tenure.outputs._common import DEFAULT_HORIZONS, DEFAULT_MIN_AT_RISK, as_survival
from tenure.outputs.ltv import survival_weighted_ltv
from tenure.outputs.retention import retention_at, rmst


def _verdict(audit_report) -> str:
    if audit_report is None:
        return "not attached"
    if audit_report.clean:
        return "clean (no findings)"
    return f"{len(audit_report.blocks)} block(s), {len(audit_report.warnings)} warning(s)"


def _df_to_markdown(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    header = "| " + " | ".join(str(c) for c in cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    lines = [header, sep]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(_fmt(row[c]) for c in cols) + " |")
    return "\n".join(lines)


def _fmt(value) -> str:
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


@dataclass
class SummaryReport:
    """A retention + LTV summary table plus the metadata needed to read it correctly."""

    table: pd.DataFrame
    metadata: dict = field(default_factory=dict)

    def to_markdown(self) -> str:
        lines = ["# Retention & LTV summary", ""]
        for key, value in self.metadata.items():
            lines.append(f"- {key}: {value}")
        lines.append("")
        lines.append(_df_to_markdown(self.table))
        lines.append("")
        return "\n".join(lines)

    def to_csv(self, path=None) -> str:
        header = "".join(f"# {key}: {value}\n" for key, value in self.metadata.items())
        csv = header + self.table.to_csv(index=False)
        if path is not None:
            with open(path, "w", encoding="utf-8", newline="") as handle:
                handle.write(csv)
        return csv

    def __repr__(self) -> str:
        return f"SummaryReport(groups={len(self.table)}, metadata_keys={list(self.metadata)})"


def summarize(
    estimator,
    *,
    period_margin: float,
    ltv_horizon: float,
    horizons=DEFAULT_HORIZONS,
    period: str = "month",
    discount_rate: float = 0.0,
    currency: str = "USD",
    min_at_risk: int = DEFAULT_MIN_AT_RISK,
    audit_report=None,
) -> SummaryReport:
    """Assemble retention-at-horizons, RMST, and LTV into one self-documenting report."""
    survival = as_survival(estimator)
    horizons = [float(h) for h in horizons]

    retention = retention_at(survival, horizons, min_at_risk=min_at_risk)
    rmst_frame = rmst(survival, horizon=ltv_horizon, min_at_risk=min_at_risk)
    ltv_frame = survival_weighted_ltv(
        survival,
        period_margin=period_margin,
        horizon=ltv_horizon,
        discount_rate=discount_rate,
        period=period,
        min_at_risk=min_at_risk,
    )
    support = evaluate_horizon_support(
        survival, sorted(set(horizons) | {float(ltv_horizon)}), min_at_risk=min_at_risk
    )

    retention_wide = retention.pivot(index="group", columns="horizon", values="retention")
    retention_wide.columns = [f"retention@{h:g}" for h in retention_wide.columns]
    table = (
        retention_wide.reset_index()
        .merge(
            rmst_frame[["group", "rmst", "effective_horizon"]].rename(
                columns={"effective_horizon": "rmst_horizon"}
            ),
            on="group",
        )
        .merge(
            ltv_frame[["group", "ltv", "effective_horizon"]].rename(
                columns={"effective_horizon": "ltv_horizon"}
            ),
            on="group",
        )
        .reset_index(drop=True)
    )

    truncated = sorted(
        set(rmst_frame.loc[rmst_frame["truncated"], "group"])
        | set(ltv_frame.loc[ltv_frame["truncated"], "group"])
    )
    metadata = {
        "time_unit": survival.time_unit,
        "horizons": horizons,
        "currency": currency,
        "period": period,
        "ltv_horizon_requested": float(ltv_horizon),
        "discount_rate": float(discount_rate),
        "min_at_risk": int(min_at_risk),
        "truncated_groups": truncated,
        "horizon_support_warnings": [f.message for f in support],
        "audit_verdict": _verdict(audit_report),
    }
    return SummaryReport(table=table, metadata=metadata)
