"""Panel-aware cross-validation: folds that never split a customer (DV4-7).

Naive k-fold on a survival panel scatters one customer's rows across train and test -- for
interval (time-varying) designs that literally trains on a customer's early intervals and "predicts"
their later ones, and even for single-spell designs it invites id-level leakage the moment features
are engineered per customer. ``panel_folds`` groups by customer ``id`` so each customer is wholly
inside one fold, and ``ensure_panel_safe`` is the ``VAL003_PANEL_LEAKAGE`` guard for splits the
user builds themselves.

WHAT PANEL CV IS (and is not): cross-sectional model assessment -- k out-of-sample estimates of a
model's covariate discrimination, giving a mean AND a spread instead of one number. It does NOT
test forward-in-time generalization: every fold sees the full calendar range, so market-wide drift
leaks across folds by construction. ``temporal_holdout`` remains the honest headline validation;
use this to put error bars on it, not to replace it.

Fold evaluation uses an ENTRY-AWARE Harrell concordance, hand-rolled here (the log-rank / IPCW
Brier precedent): lifelines' ``concordance_index`` cannot honor delayed entry, and a window-cut
cohort scored as if everyone were observed from tenure zero compares pairs that were never at risk
together -- exactly the class of silent bias this library exists to prevent. Each event at time
``T`` is compared only against subjects actually AT RISK at ``T`` (``entry < T``, the same risk-set
convention as the delayed-entry Kaplan-Meier and the log-rank test). With no delayed entry it
reduces to -- and is reference-match tested against -- lifelines' ``concordance_index``, including
its tie conventions (an event and a censoring at the same time are comparable; two events at the
same time are not; tied risks earn half credit). For interval designs each interval row is a risk
unit with its own per-interval risk -- the standard counting-process concordance.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from tenure._frame import ENTRY, EVENT, EXIT, ID, ensure_estimable
from tenure.exceptions import TenureValidationError
from tenure.study_design import StudyDesign
from tenure.validation.result import VAL003_PANEL_LEAKAGE, ValidationResult


def ensure_panel_safe(*id_groups: Any) -> None:
    """Raise (citing VAL003) when any customer id appears in more than one of ``id_groups``.

    The DV4-7 leakage guard for user-supplied splits: a customer split across train/test (or
    across folds) leaks their own outcome into training and silently inflates every metric.
    ``panel_folds`` calls this on every fold it builds; call it yourself on any split you
    construct by hand.

    Args:
        *id_groups: Two or more iterables of customer ids (one per side/fold).

    Raises:
        TenureValidationError: Naming the offending ids, when any two groups overlap.
    """
    sets = [set(g) for g in id_groups]
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            overlap = sets[i] & sets[j]
            if overlap:
                sample = sorted(str(x) for x in overlap)[:5]
                raise TenureValidationError(
                    f"{VAL003_PANEL_LEAKAGE}: {len(overlap)} customer id(s) appear in more than "
                    f"one fold/side (e.g. {sample}). A customer must be wholly inside one fold -- "
                    "their own future otherwise leaks into training. Split by customer id "
                    "(panel_folds does this for you)."
                )


def _rebuild(design, rows: pd.DataFrame) -> StudyDesign:
    """Build a fold StudyDesign directly from a subset of canonical rows.

    Unlike the temporal holdout (which MODIFIES rows and so re-ingests through
    ``from_intervals``), a fold is a pure row subset -- so the canonical float tenures are carried
    over verbatim (no tenure -> date -> tenure round trip, no float precision loss; review fix)
    and the parent's ``covariate_mappings`` are inherited, keeping every fold's encoded column
    space identical to the parent's.
    """
    clone = StudyDesign(
        canonical=rows.reset_index(drop=True),
        analysis_start=design.analysis_start,
        event_observed_from=design.event_observed_from,
        entry_modeled=design.entry_modeled,
        includes_pre_entry_churners=design.includes_pre_entry_churners,
        group_cols=list(design.group_cols),
        time_unit=design.time_unit,
        covariate_cols=list(design.covariate_cols),
        covariate_mappings=dict(design.covariate_mappings),
        interval=design.interval,
        attest_origin_correct=design.attest_origin_correct,
        attest_invariant_covariates=list(design.attest_invariant_covariates),
        status_map=design.status_map,
        n_excluded=design.n_excluded,
        n_unmapped=design.n_unmapped,
        unmapped_statuses=list(design.unmapped_statuses),
        informative_censoring_statuses=list(design.informative_censoring_statuses),
    )
    # Carry the audit gate: panel_folds already required the parent to be estimable, so its folds
    # are too (a fresh __init__ resets audited=False, which would re-block an audited parent).
    clone.audited = design.audited
    return clone


def panel_folds(design: Any, k: int = 5, *, seed: int = 0) -> list:
    """Split a design into ``k`` cross-validation folds that never split a customer (DV4-7).

    Customer ids are shuffled deterministically (``seed``) and partitioned into ``k`` groups;
    fold ``i`` holds group ``i`` out as test and trains on the rest. Every row of a customer
    (all intervals, for a time-varying design) travels together. Each fold's disjointness is
    asserted via ``ensure_panel_safe`` before it is returned -- the leakage guarantee is checked,
    not assumed.

    This is a CROSS-SECTIONAL split: it estimates covariate discrimination with a spread, and
    complements (never replaces) the forward-in-time ``temporal_holdout``.

    Args:
        design: The StudyDesign to fold (single-spell or interval).
        k: Number of folds; must satisfy ``2 <= k <= number of customers``.
        seed: Shuffle seed, for reproducible folds.

    Returns:
        A list of ``k`` tuples ``(train_design, test_design)``.
    """
    ensure_estimable(design)
    if not isinstance(k, int) or k < 2:
        raise TenureValidationError(f"k must be an integer >= 2; got {k!r}.")
    table = design.canonical
    # Sorted before shuffling so folds depend only on (id set, seed), not on the dataframe's row
    # order -- the same records in a different order produce identical folds (review fix).
    ids = np.sort(pd.unique(table[ID]))
    if k > len(ids):
        raise TenureValidationError(
            f"k={k} exceeds the number of customers ({len(ids)}); every fold needs at least one."
        )

    shuffled = np.random.default_rng(seed).permutation(ids)
    chunks = np.array_split(shuffled, k)

    folds = []
    for chunk in chunks:
        test_ids = set(chunk)
        is_test = table[ID].isin(test_ids).to_numpy()
        train_design = _rebuild(design, table.loc[~is_test])
        test_design = _rebuild(design, table.loc[is_test])
        ensure_panel_safe(train_design.canonical[ID].unique(), test_design.canonical[ID].unique())
        folds.append((train_design, test_design))
    return folds


def _entry_aware_concordance(
    entry: np.ndarray,
    duration: np.ndarray,
    event: np.ndarray,
    risk: np.ndarray,
    groups: np.ndarray | None = None,
) -> tuple[float, int]:
    """Harrell's C restricted to pairs genuinely at risk together (delayed-entry aware).

    For each event at time ``T``, the comparable set is every other row AT RISK at ``T``
    (``entry < T``) that outlived it (``duration > T``, or censored exactly at ``T`` -- known to
    survive at least ``T``). Two events at the same time are not comparable; tied risks earn half
    credit. With ``entry == 0`` everywhere this reproduces lifelines' ``concordance_index``
    exactly (reference-match tested). Returns ``(c_index, n_comparable_pairs)``.

    ``groups`` (optional) further restricts comparability to SAME-GROUP pairs -- the stratified
    C-index. A stratified Cox's partial hazard ``exp(beta^T x)`` carries no baseline, so ranking
    it across strata assumes the strata share a baseline hazard, which is precisely what the model
    rejects; within-stratum pairs pooled by pair count (the pair-weighted average) score what the
    model actually claims (review fix).

    Complexity: O(events x rows) boolean work per call -- exact and simple, fine at CV-fold sizes,
    but a known bottleneck on very large cohorts (e.g. 5e4 events x 5e5 rows).
    """
    credit = 0.0
    n_pairs = 0
    for i in np.flatnonzero(event == 1):
        t_i = duration[i]
        comparable = (entry < t_i) & ((duration > t_i) | ((duration == t_i) & (event == 0)))
        if groups is not None:
            comparable &= groups == groups[i]
        n = int(comparable.sum())
        if n == 0:
            continue
        n_pairs += n
        r = risk[comparable]
        credit += float((risk[i] > r).sum()) + 0.5 * float((risk[i] == r).sum())
    if n_pairs == 0:
        raise TenureValidationError(
            "entry-aware C-index is undefined: no comparable pairs (no two subjects were at risk "
            "together around an event, within a stratum if stratified). The fold may be too small "
            "or all-censored; reduce k."
        )
    return credit / n_pairs, n_pairs


def _strata_labels(table: pd.DataFrame, strata: list) -> np.ndarray:
    """One combined per-row stratum label (multi-column strata joined, KM-label style)."""
    labels = table[strata[0]].astype(str)
    for col in strata[1:]:
        labels = labels + "|" + table[col].astype(str)
    return labels.to_numpy()


def _fold_risk(model, test_design) -> np.ndarray:
    """Per-row partial hazard of a fitted Cox-family model on a held-out fold design."""
    fitter = getattr(model, "fitter", None)
    encode = getattr(model, "encode_for_prediction", None)
    if fitter is None or encode is None:
        raise TenureValidationError(
            "cross_validate needs a Cox-family model (CoxPH or TimeVaryingCox -- something with "
            f"per-subject covariate risk); got {type(model).__name__}. An overall KM predicts one "
            "cohort curve, so its C-index is ~0.5 by construction (DV4-5) and CV adds nothing."
        )
    encoded = encode(test_design)
    return np.asarray(fitter.predict_partial_hazard(encoded), dtype=float)


def cross_validate(
    model_factory: Callable[[], Any], design: Any, *, k: int = 5, seed: int = 0
) -> ValidationResult:
    """K-fold panel-aware cross-validation of a Cox-family model's discrimination.

    Fits a fresh model per fold (``model_factory()`` then ``.fit(train)``) and scores the
    entry-aware Harrell C-index on the held-out customers, yielding a mean AND a spread instead of
    a single number. Folds come from ``panel_folds`` (customers never split; leakage asserted).
    Cross-sectional by construction -- see ``panel_folds``; ``temporal_holdout`` remains the
    forward-in-time headline validation.

    Stratified models are scored with the STRATIFIED C-index (within-stratum pairs, pooled by
    pair count): a stratified Cox's partial hazard carries no baseline, so cross-strata ranking
    is meaningless by the model's own assumptions. Note what that scores: discrimination from the
    remaining covariates. The stratified variable's effect lives in the baselines and is
    deliberately not part of the C-index -- if you want it scored, keep it as a covariate instead.

    Args:
        model_factory: Zero-argument callable returning an unfitted estimator, e.g.
            ``lambda: tenure.CoxPH()`` (stratified Cox and ``TimeVaryingCox`` work too). A fresh
            instance per fold prevents state bleeding between folds.
        design: The StudyDesign to validate on (must carry ``covariate_cols``).
        k: Number of folds (``2 <= k <= number of customers``).
        seed: Fold-shuffle seed, for reproducibility.

    Returns:
        A ``ValidationResult``: ``.table`` has one row per fold
        ``[fold, c_index, n_pairs, n_train_subjects, n_test_subjects, n_test_events]``;
        ``.metadata['estimate']`` is the mean C-index and ``'std'`` its sample standard deviation.
    """
    folds = panel_folds(design, k, seed=seed)
    rows = []
    for fold_i, (train_design, test_design) in enumerate(folds):
        model = model_factory()
        model.fit(train_design)
        test_table = test_design.canonical
        risk = _fold_risk(model, test_design)
        if len(risk) != len(test_table):
            raise TenureValidationError(
                f"fold {fold_i}: risk length ({len(risk)}) != test rows ({len(test_table)})."
            )
        if not np.isfinite(risk).all():
            raise TenureValidationError(
                f"fold {fold_i}: non-finite risk scores; the model may not have converged."
            )
        # A stratified model's partial hazard carries no baseline, so pairs are restricted to
        # within-stratum (the stratified C-index) -- cross-strata ranking would silently assume a
        # shared baseline, throwing away exactly what the strata encode (review fix).
        strata = list(getattr(model, "strata", None) or [])
        groups = _strata_labels(test_table, strata) if strata else None
        try:
            c, n_pairs = _entry_aware_concordance(
                test_table[ENTRY].to_numpy(dtype=float),
                test_table[EXIT].to_numpy(dtype=float),
                test_table[EVENT].to_numpy(dtype=int),
                risk,
                groups=groups,
            )
        except TenureValidationError as exc:
            raise TenureValidationError(f"fold {fold_i}: {exc}") from exc
        rows.append(
            {
                "fold": fold_i,
                "c_index": c,
                "n_pairs": n_pairs,
                "n_train_subjects": int(train_design.canonical[ID].nunique()),
                "n_test_subjects": int(test_table[ID].nunique()),
                "n_test_events": int(test_table[EVENT].sum()),
            }
        )

    table = pd.DataFrame(rows)
    estimates = table["c_index"].to_numpy(dtype=float)
    entry_aware = bool((design.canonical[ENTRY].to_numpy(dtype=float) > 0).any())
    metadata = {
        "metric": "c_index_cv",
        "estimate": float(estimates.mean()),
        "std": float(estimates.std(ddof=1)),
        "k": int(k),
        "seed": int(seed),
        "n_subjects": int(design.canonical[ID].nunique()),
        "entry_aware": entry_aware,  # delayed entry present -> risk-set-restricted pairs
        # Within-stratum pairs for stratified models: the stratified variable's own effect lives
        # in the baselines and is deliberately NOT scored (that is what stratifying means).
        "pair_restriction": "within_stratum" if groups is not None else "all",
        "censoring_method": "right_censored_harrell_entry_aware",
        "model_type": type(model).__name__,  # the last fitted fold model
        "note": (
            "cross-sectional (panel-aware) CV: estimates covariate discrimination with a spread; "
            "complements temporal_holdout, which tests forward-in-time generalization."
        ),
        "warnings": [],
    }
    return ValidationResult(table=table, metadata=metadata)
