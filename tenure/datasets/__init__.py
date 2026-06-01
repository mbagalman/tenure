"""Bundled demo datasets (offline, deterministic)."""

from __future__ import annotations

from tenure.datasets.svod_demo import (
    ACTIVE_AS_OF,
    ANALYSIS_START,
    COHORT_START,
    SvodTruth,
    load_svod_demo,
    svod_demo_truth,
)

__all__ = [
    "load_svod_demo",
    "svod_demo_truth",
    "SvodTruth",
    "COHORT_START",
    "ANALYSIS_START",
    "ACTIVE_AS_OF",
]
