from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from lifelines.utils import concordance_index

import tenure
from tenure import CoxPH, StudyDesign, TenureValidationError, TimeVaryingCox
from tenure._frame import ENTRY, EVENT, EXIT, ID
from tenure.validation.cv import _entry_aware_concordance
from tenure.validation.result import VAL003_PANEL_LEAKAGE


def _df(n=900, seed=0, scale_basic=150.0, scale_premium=600.0):
    """Tier strongly drives lifetime (basic churns ~4x faster) => a Cox model discriminates."""
    rng = np.random.default_rng(seed)
    signup = pd.Timestamp("2024-01-01")
    tier = rng.choice(["basic", "premium"], size=n)
    age = rng.integers(20, 70, size=n).astype(float)  # noise
    scale = np.where(tier == "premium", scale_premium, scale_basic)
    lifetime = rng.exponential(scale)
    churn = pd.Series(signup + pd.to_timedelta(lifetime, unit="D"))
    return pd.DataFrame(
        {
            "cid": [f"c{i}" for i in range(n)],
            "start": signup,
            "churn": churn.where(churn <= pd.Timestamp("2026-05-31")),
            "tier": tier,
            "age": age,
        }
    )


def _design(df=None, **kwargs):
    return StudyDesign.from_event_dates(
        _df() if df is None else df,
        id_col="cid",
        origin_col="start",
        churn_date_col="churn",
        active_as_of="2026-05-31",
        covariate_cols=["tier", "age"],
        **kwargs,
    )


# --- the entry-aware concordance itself -------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_concordance_matches_lifelines_at_zero_entry(seed):  # reference-match, ties included
    rng = np.random.default_rng(seed)
    n = 250
    durations = np.round(rng.exponential(100.0, n)) + 1.0  # rounding forces time ties
    events = (rng.random(n) < 0.7).astype(int)
    risk = np.round(rng.random(n), 2)  # rounding forces risk ties
    ours, _ = _entry_aware_concordance(np.zeros(n), durations, events, risk)
    ref = concordance_index(durations, -risk, events)
    assert np.isclose(ours, ref, atol=1e-12)


def test_tie_conventions_match_lifelines():
    # Pinned to lifelines behavior (probed empirically): event vs censored at the same time IS
    # comparable; two events at the same time are NOT; tied risks earn half credit.
    zero = np.zeros(2)
    c, n = _entry_aware_concordance(
        zero, np.array([5.0, 5.0]), np.array([1, 0]), np.array([2.0, 1.0])
    )
    assert (c, n) == (1.0, 1)
    with pytest.raises(TenureValidationError, match="no comparable pairs"):
        _entry_aware_concordance(zero, np.array([5.0, 5.0]), np.array([1, 1]), np.array([2.0, 1.0]))
    c, n = _entry_aware_concordance(
        zero, np.array([3.0, 7.0]), np.array([1, 0]), np.array([1.0, 1.0])
    )
    assert (c, n) == (0.5, 1)


def test_within_stratum_pairs_hand_computed():
    # Stratum A churns fast with LOW partial hazards; stratum B churns slow with HIGH partial
    # hazards (its protection lives in the baseline, which exp(Xb) does not carry). Within each
    # stratum the ranking is perfect: a1(ev@10, r=5) vs a2(cens@20, r=1) concordant; b1(ev@100,
    # r=9) vs b2(cens@200, r=8) concordant -> stratified C = 2/2 = 1.0. UNRESTRICTED pairs add
    # the misleading cross-strata comparisons (a1 vs b1, a1 vs b2: discordant) -> C = 2/4 = 0.5.
    entry = np.zeros(4)
    duration = np.array([10.0, 20.0, 100.0, 200.0])
    event = np.array([1, 0, 1, 0])
    risk = np.array([5.0, 1.0, 9.0, 8.0])
    groups = np.array(["A", "A", "B", "B"])
    grouped, n_grouped = _entry_aware_concordance(entry, duration, event, risk, groups=groups)
    assert (grouped, n_grouped) == (1.0, 2)
    ungrouped, n_all = _entry_aware_concordance(entry, duration, event, risk)
    assert (ungrouped, n_all) == (0.5, 4)  # the flaw the restriction corrects


def test_entry_awareness_hand_computed():
    # i: event at t=10 (risk 3). j: censored at 20 (risk 1) -> comparable, concordant.
    # k: ENTERS at 15, after i's event (risk 5, event at 30) -> NOT at risk at t=10, excluded;
    # k's own event at 30 has no one at risk beyond it. Entry-aware C = 1/1 = 1.0.
    # Ignoring entry (lifelines) would also count (i, k) as discordant -> C = 0.5. The risk-set
    # restriction is exactly what separates the two.
    entry = np.array([0.0, 0.0, 15.0])
    duration = np.array([10.0, 20.0, 30.0])
    event = np.array([1, 0, 1])
    risk = np.array([3.0, 1.0, 5.0])
    ours, n_pairs = _entry_aware_concordance(entry, duration, event, risk)
    assert (ours, n_pairs) == (1.0, 1)
    naive = concordance_index(duration, -risk, event)
    assert np.isclose(naive, 0.5)  # the biased answer the entry-aware form corrects


# --- panel_folds: the leakage-safe splitter ---------------------------------------------------


def test_folds_partition_customers():
    design = _design()
    folds = tenure.panel_folds(design, k=4, seed=0)
    assert len(folds) == 4
    test_sets = [set(test.canonical[ID]) for _, test in folds]
    assert set().union(*test_sets) == set(design.canonical[ID])  # union = everyone
    for i in range(4):
        for j in range(i + 1, 4):
            assert not (test_sets[i] & test_sets[j])  # pairwise disjoint
    for train, test in folds:
        assert not (set(train.canonical[ID]) & set(test.canonical[ID]))  # no leak within a fold


def test_folds_deterministic_and_seed_sensitive():
    design = _design()
    a = tenure.panel_folds(design, k=3, seed=7)
    b = tenure.panel_folds(design, k=3, seed=7)
    c = tenure.panel_folds(design, k=3, seed=8)
    for (_, ta), (_, tb) in zip(a, b, strict=True):
        assert set(ta.canonical[ID]) == set(tb.canonical[ID])
    assert any(
        set(ta.canonical[ID]) != set(tc.canonical[ID])
        for (_, ta), (_, tc) in zip(a, c, strict=True)
    )


def test_folds_invariant_to_row_order():
    # Same records, different dataframe order -> identical folds for the same seed (ids are
    # sorted before shuffling; review fix).
    df = _df()
    a = tenure.panel_folds(_design(df), k=3, seed=0)
    b = tenure.panel_folds(_design(df.sample(frac=1.0, random_state=42)), k=3, seed=0)
    for (_, ta), (_, tb) in zip(a, b, strict=True):
        assert set(ta.canonical[ID]) == set(tb.canonical[ID])


def test_fold_designs_round_trip_exactly():
    # Fold designs carry the parent's canonical rows verbatim -- float tenures bit-for-bit (no
    # tenure -> date -> tenure round trip; review fix), covariates, and the parent's
    # covariate_mappings (so every fold encodes into the identical column space).
    df = tenure.load_svod_demo(with_left_truncation=True)
    design = StudyDesign.from_event_dates(
        df,
        id_col="customer_id",
        origin_col="signup_date",
        churn_date_col="churn_date",
        active_as_of="2026-05-31",
        analysis_start="2024-01-01",
        event_observed_from="2024-01-01",
        covariate_cols=["plan", "channel"],
    )
    tenure.audit(design)
    parent = design.canonical
    assert parent[ENTRY].to_numpy(float).max() > 0.0  # real delayed entry

    train, test = tenure.panel_folds(design, k=3, seed=0)[0]
    for side in (train, test):
        sub = side.canonical.sort_values(ID).reset_index(drop=True)
        ref = parent[parent[ID].isin(set(sub[ID]))].sort_values(ID).reset_index(drop=True)
        assert np.array_equal(sub[ENTRY].to_numpy(float), ref[ENTRY].to_numpy(float))  # exact
        assert np.array_equal(sub[EXIT].to_numpy(float), ref[EXIT].to_numpy(float))  # exact
        assert (sub[EVENT].to_numpy(int) == ref[EVENT].to_numpy(int)).all()
        assert (sub["plan"].to_numpy() == ref["plan"].to_numpy()).all()
        assert side.covariate_mappings == design.covariate_mappings  # inherited, not re-derived
        assert side.time_unit == design.time_unit
        assert side.interval == design.interval
    assert test.canonical[ENTRY].to_numpy(float).max() > 0.0  # delayed entry preserved


def test_interval_customers_travel_whole():
    # Every interval row of a customer lands in the same fold (the panel guarantee for
    # time-varying designs).
    panel = _tv_panel()
    design = StudyDesign.from_intervals(
        panel,
        id_col="cid",
        origin_col="orig",
        interval_start_col="s",
        interval_end_col="e",
        event_col="ev",
        covariate_cols=["active"],
    )
    folds = tenure.panel_folds(design, k=3, seed=0)
    parent_counts = design.canonical.groupby(ID).size()
    seen: dict = {}
    for _, test in folds:
        counts = test.canonical.groupby(ID).size()
        for cid, n_rows in counts.items():
            assert cid not in seen  # a customer appears in exactly one test fold
            seen[cid] = n_rows
            assert n_rows == parent_counts[cid]  # with ALL of their intervals
    assert len(seen) == len(parent_counts)


def test_k_validation():
    design = _design()
    with pytest.raises(TenureValidationError, match="k must be"):
        tenure.panel_folds(design, k=1)
    with pytest.raises(TenureValidationError, match="exceeds the number of customers"):
        tenure.panel_folds(design, k=10_000)


def test_unaudited_unmapped_design_refused():
    df = _df(n=60, seed=5)
    df["status"] = np.where(df["churn"].notna(), "churned", "active")
    df.loc[df.index[:5], "status"] = "mystery"  # unmapped -> dropped + counted
    df["exit_date"] = df["churn"].fillna(pd.Timestamp("2026-05-31"))
    design = StudyDesign.from_status(
        df,
        id_col="cid",
        origin_col="start",
        exit_col="exit_date",
        status_col="status",
        status_map={"churned": "event", "active": "censored"},
        active_as_of="2026-05-31",
    )
    with pytest.raises(TenureValidationError, match="audit"):
        tenure.panel_folds(design, k=3)


# --- ensure_panel_safe: the VAL003 guard ------------------------------------------------------


def test_ensure_panel_safe():
    tenure.ensure_panel_safe(["a", "b"], ["c", "d"], ["e"])  # disjoint -> silent
    with pytest.raises(TenureValidationError, match=VAL003_PANEL_LEAKAGE):
        tenure.ensure_panel_safe(["a", "b"], ["b", "c"])
    with pytest.raises(TenureValidationError, match=VAL003_PANEL_LEAKAGE):
        tenure.ensure_panel_safe(["a"], ["b"], ["c", "a"])  # any pair of groups


# --- cross_validate ---------------------------------------------------------------------------


def test_cross_validate_informative_covariates():
    design = _design()  # tier drives a 4x lifetime gap
    res = tenure.cross_validate(lambda: CoxPH(), design, k=4, seed=0)
    assert len(res.table) == 4
    assert res.metadata["metric"] == "c_index_cv"
    assert res.metadata["estimate"] > 0.6  # real discrimination, recovered out of fold
    assert res.metadata["std"] >= 0.0
    assert (res.table["c_index"] > 0.55).all()
    assert res.metadata["entry_aware"] is False
    assert res.metadata["model_type"] == "CoxPH"


def test_cross_validate_noise_covariates_near_chance():
    df = tenure.load_svod_demo(with_left_truncation=False)
    design = StudyDesign.from_event_dates(
        df,
        id_col="customer_id",
        origin_col="signup_date",
        churn_date_col="churn_date",
        active_as_of="2026-05-31",
        covariate_cols=["plan", "channel"],  # noise by construction in the demo
    )
    tenure.audit(design)
    res = tenure.cross_validate(lambda: CoxPH(), design, k=4, seed=0)
    assert 0.45 < res.metadata["estimate"] < 0.55


def test_cross_validate_deterministic():
    design = _design()
    a = tenure.cross_validate(lambda: CoxPH(), design, k=3, seed=1)
    b = tenure.cross_validate(lambda: CoxPH(), design, k=3, seed=1)
    assert np.allclose(a.table["c_index"], b.table["c_index"])


def test_cross_validate_delayed_entry_flagged():
    df = tenure.load_svod_demo(with_left_truncation=True)
    design = StudyDesign.from_event_dates(
        df,
        id_col="customer_id",
        origin_col="signup_date",
        churn_date_col="churn_date",
        active_as_of="2026-05-31",
        analysis_start="2024-01-01",
        event_observed_from="2024-01-01",
        covariate_cols=["plan", "channel"],
    )
    tenure.audit(design)
    res = tenure.cross_validate(lambda: CoxPH(), design, k=3, seed=0)
    assert res.metadata["entry_aware"] is True  # risk-set-restricted pairs were used


def test_cross_validate_stratified_noise_covariate_near_chance():
    # In _design, tier drives the 4x lifetime gap and age is noise. Stratifying on tier moves
    # its effect into the baselines, so the stratified C-index (within-stratum pairs) scores only
    # what remains -- noise -- and must sit near chance, NOT inherit tier's discrimination.
    design = _design()
    res = tenure.cross_validate(lambda: CoxPH(strata=["tier"]), design, k=3, seed=0)
    assert len(res.table) == 3
    assert res.metadata["pair_restriction"] == "within_stratum"
    assert 0.40 < res.metadata["estimate"] < 0.60


def _informative_within_stratum_df(n=900, seed=0):
    """Tier shifts the baseline 4x; age drives hazard WITHIN each tier (exp(0.02 * age))."""
    rng = np.random.default_rng(seed)
    signup = pd.Timestamp("2024-01-01")
    tier = rng.choice(["basic", "premium"], size=n)
    age = rng.integers(20, 70, size=n).astype(float)
    scale = np.where(tier == "premium", 600.0, 150.0) * np.exp(-0.02 * (age - 45.0))
    lifetime = rng.exponential(scale)
    churn = pd.Series(signup + pd.to_timedelta(lifetime, unit="D"))
    return pd.DataFrame(
        {
            "cid": [f"c{i}" for i in range(n)],
            "start": signup,
            "churn": churn.where(churn <= pd.Timestamp("2026-05-31")),
            "tier": tier,
            "age": age,
        }
    )


def test_cross_validate_stratified_recovers_within_stratum_signal():
    design = _design(_informative_within_stratum_df())
    res = tenure.cross_validate(lambda: CoxPH(strata=["tier"]), design, k=3, seed=0)
    assert res.metadata["pair_restriction"] == "within_stratum"
    assert res.metadata["estimate"] > 0.58  # age's real within-stratum effect, out of fold


def test_concordance_holdout_stratified_uses_within_stratum_pairs():
    # The same review fix applies to the temporal-holdout concordance: a stratified model's
    # partial hazard must not be ranked across strata there either.
    design = _design(_informative_within_stratum_df())
    train, test = tenure.temporal_holdout(design, cutoff="2025-01-01")
    model = CoxPH(strata=["tier"]).fit(train)
    res = tenure.concordance(model, test)
    assert res.metadata["pair_restriction"] == "within_stratum"
    assert np.isfinite(res.estimate)
    plain = tenure.concordance(CoxPH().fit(train), test)
    assert plain.metadata["pair_restriction"] == "all"  # unstratified path unchanged


def _tv_panel(n=400, seed=0):
    """Two-interval panel: `active` flips 0 -> 1 at a PER-SUBJECT time and triples the hazard.

    The flip time must differ across subjects: if everyone flipped at the same tenure, the
    covariate would be constant within every risk set (complete separation, nothing to estimate,
    and every comparable pair a risk tie -> C = 0.5 exactly).
    """
    rng = np.random.default_rng(seed)
    orig = pd.Timestamp("2024-01-01")
    rows = []
    for i in range(n):
        u = rng.uniform(50.0, 300.0)  # this subject's flip time
        t1 = rng.exponential(400.0)  # hazard while active=0
        if t1 < u:
            rows.append({"cid": f"c{i}", "orig": orig, "s": 0.0, "e": t1, "ev": 1, "active": 0})
            continue
        t2 = u + rng.exponential(133.0)  # ~3x hazard while active=1
        end, ev = (t2, 1) if t2 < 500.0 else (500.0, 0)
        rows.append({"cid": f"c{i}", "orig": orig, "s": 0.0, "e": u, "ev": 0, "active": 0})
        rows.append({"cid": f"c{i}", "orig": orig, "s": u, "e": end, "ev": ev, "active": 1})
    df = pd.DataFrame(rows)
    for col in ("s", "e"):
        df[col] = orig + pd.to_timedelta(df[col], unit="D")
    return df


def test_cross_validate_time_varying_cox():
    design = StudyDesign.from_intervals(
        _tv_panel(),
        id_col="cid",
        origin_col="orig",
        interval_start_col="s",
        interval_end_col="e",
        event_col="ev",
        covariate_cols=["active"],
    )
    res = tenure.cross_validate(lambda: TimeVaryingCox(), design, k=3, seed=0)
    # Counting-process concordance: interval rows are the risk units; the time-varying
    # covariate genuinely drives hazard, so out-of-fold discrimination beats chance.
    assert res.metadata["estimate"] > 0.55
    assert res.metadata["model_type"] == "TimeVaryingCox"


def test_cross_validate_rejects_non_cox_model():
    design = _design()
    with pytest.raises(TenureValidationError, match="Cox-family"):
        tenure.cross_validate(lambda: tenure.KaplanMeier(), design, k=3)
