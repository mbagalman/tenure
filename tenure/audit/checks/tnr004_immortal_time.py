"""TNR004 -- Immortal-time / future-looking covariate (data-driven, WARN only).

For each grouping covariate, a quantile-based shift test (D-S5): the baseline level is the one
with the lowest minimum tenure (earliest-available); any other viable level whose minimum tenure
clears the baseline's 10th percentile AND an absolute floor cannot contain early churners -- a
signature consistent with conditioning on future survival (immortal-time bias).

Honest about its limits: it WARNS ("consistent with"), only inspects ``group_cols``, requires a
minimum sample per level, and is cleared by ``attest_invariant_covariates``. Full prevention
arrives with time-varying covariates (v0.3).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tenure._frame import EXIT
from tenure.audit.base import AuditCheck
from tenure.audit.registry import register
from tenure.audit.report import CheckResult, Status

_DELTA = 30.0  # absolute minimum tenure gap (in the design's time unit) to flag
_N_MIN = 30  # minimum sample per level to assess
_P_LOW = 10.0  # baseline percentile to compare against


@register
class ImmortalTimeCheck(AuditCheck):
    id = "TNR004"
    title = "Immortal-time / future-looking covariate"

    def evaluate(self, design) -> CheckResult | None:
        scan_cols = list(
            dict.fromkeys([*design.group_cols, *getattr(design, "covariate_cols", [])])
        )
        if not scan_cols:
            return None

        attested = set(design.attest_invariant_covariates)
        cols = [c for c in scan_cols if c not in attested]
        if not cols:
            return CheckResult(
                self.id,
                Status.PASS,
                self.title,
                "All grouping covariates are attested time-invariant.",
            )

        table = design.canonical
        tenure = table[EXIT].to_numpy()
        flagged: list[str] = []
        for col in cols:
            if self._has_immortal_signature(table[col].to_numpy(), tenure):
                flagged.append(col)

        if flagged:
            return CheckResult(
                self.id,
                Status.WARN,
                self.title,
                f"Covariate(s) {flagged} have a level that appears only for higher-tenure "
                "customers -- a signature consistent with immortal-time bias. Grouped curves may "
                "show an illusory protective effect.",
                remediation=(
                    "If these covariates are genuinely set at origin (time-invariant), pass "
                    "attest_invariant_covariates=[...]. Full prevention arrives with time-varying "
                    "covariates (v0.3)."
                ),
                details={"covariates": flagged},
            )

        return CheckResult(
            self.id,
            Status.PASS,
            self.title,
            "No grouping covariate shows an immortal-time signature.",
        )

    @staticmethod
    def _has_immortal_signature(values: np.ndarray, tenure: np.ndarray) -> bool:
        stats = {}
        for level in pd.unique(values):
            mask = values == level
            count = int(mask.sum())
            if count >= _N_MIN:
                stats[level] = float(tenure[mask].min())
        if len(stats) < 2:
            return False
        baseline = min(stats, key=stats.get)
        p_low = float(np.percentile(tenure[values == baseline], _P_LOW))
        return any(
            min_tenure > p_low and min_tenure > _DELTA
            for level, min_tenure in stats.items()
            if level != baseline
        )
