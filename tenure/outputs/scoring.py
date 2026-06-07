"""Individual churn-risk scoring & ranking (v0.2; DV2-3).

`churn_risk_scores` turns a fitted Cox model into a per-customer risk table -- the first
prediction output. It carries the audited-design provenance, like SummaryReport.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from tenure._frame import ID
from tenure.outputs._common import audit_verdict

_SCORE_COLUMNS = ["id", "risk_score", "survival_at_horizon", "risk_percentile"]


@dataclass
class RiskScores:
    """Per-customer churn-risk scores (`.table`) plus provenance/context (`.metadata`)."""

    table: pd.DataFrame
    metadata: dict = field(default_factory=dict)

    def to_csv(self, path=None) -> str:
        header = "".join(f"# {key}: {value}\n" for key, value in self.metadata.items())
        csv = header + self.table.to_csv(index=False)
        if path is not None:
            with open(path, "w", encoding="utf-8", newline="") as handle:
                handle.write(csv)
        return csv

    def __repr__(self) -> str:
        return (
            f"RiskScores(n={len(self.table)}, horizon={self.metadata.get('horizon')}, "
            f"audit_verdict={self.metadata.get('audit_verdict')!r})"
        )


def churn_risk_scores(cox, design=None, *, horizon: float = 365.0, audit_report=None) -> RiskScores:
    """Score each customer in ``design`` (default: the model's training design) with a fitted CoxPH.

    Returns a `RiskScores` whose `.table` is tidy `[id, risk_score, survival_at_horizon,
    risk_percentile]`:
    - ``risk_score`` = Cox partial hazard ratio exp(beta^T X), i.e. lifelines
      predict_partial_hazard (higher = riskier).
    - ``survival_at_horizon`` = predicted survival probability at ``horizon``.
    - ``risk_percentile`` = ``rank(pct=True)`` of risk_score within the cohort, in [0, 1].
    """
    design = design if design is not None else cox.design
    table = design.derive()
    encoded = cox.encode_for_prediction(design)

    risk_score = cox.fitter.predict_partial_hazard(encoded).to_numpy(dtype=float)
    survival = cox.fitter.predict_survival_function(encoded, times=[float(horizon)])
    survival_at_horizon = survival.iloc[0].to_numpy(dtype=float)

    scores = pd.DataFrame(
        {
            "id": table[ID].to_numpy(),
            "risk_score": risk_score,
            "survival_at_horizon": survival_at_horizon,
            "risk_percentile": pd.Series(risk_score).rank(pct=True).to_numpy(),
        }
    )[_SCORE_COLUMNS]

    metadata = {
        "horizon": float(horizon),
        "time_unit": cox.design.time_unit,
        "n_customers": int(len(scores)),
        "covariates": list(cox.design.covariate_cols),
        "audit_verdict": audit_verdict(audit_report),
    }
    return RiskScores(table=scores, metadata=metadata)
