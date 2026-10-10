#!/usr/bin/env python3
"""Rank-aware threshold experiment (v5).

Hypothesis
----------
The current global threshold (0.998722) is chosen to maximise calibration
macro F0.5 with a single cutoff applied uniformly to all candidates.
address_rescue_rank is the 3rd most important LR feature (coef -2.07), meaning
the model already encodes rank information in its probability scores.  However,
a single threshold cannot simultaneously be permissive enough for high-rank
candidates (rank 1-20, almost always true matches when probability is high) and
strict enough for low-rank candidates (rank 150-200, mostly hard negatives).

Change vs v2
------------
After training the IDENTICAL LR model on the IDENTICAL training split, the
calibration step searches over a 2D grid of (rank_cutoff R, threshold_low t_lo)
while keeping a fixed threshold_high = the v2 global threshold.  For each S1:
  - candidates with address_rescue_rank <= R use t_lo
  - candidates with address_rescue_rank >  R use threshold_high
Both R and t_lo are chosen to maximise calibration macro F0.5.  Validation
labels are never consulted during this search.

Everything else is byte-for-byte identical to v2:
  same 400 S1s, same seed, same 60/20/20 split, same CAP=200, same Address
  Rescue ranking, same 40 features, same LR config, same correctness checks.
"""
from __future__ import annotations

import gc
import sys
import time
from pathlib import Path

import numpy as np

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from experiment_matching_model import (  # noqa: E402
    FEATURES, CAP, make_feature_vector, predict_sets, stratified_sample_and_split,
)
from optimize_matching_experiment import NumpyQueryIndex, SPLITS, WEIGHTS  # noqa: E402
from experiment_matching_model import address_rescue_score  # noqa: E402
from src.evaluation.metrics import macro_f05  # noqa: E402
from src.modeling.train import predict_proba, train_logistic_regression  # noqa: E402
from src.pipeline.run_pipeline import (  # noqa: E402
    CompactBlockingIndex, load_ground_truth, stream_records,
)
from src.preprocessing.normalization import normalize_record  # noqa: E402

COUNTRIES = ("india", "us")
REQUESTED_SAMPLE_PER_COUNTRY = 200
FALLBACK_SAMPLE_PER_COUNTRY = 100
SEED = 42
PROGRESS_EVERY = 50

# Rank-aware grid: cutoff values to search over calibration data only.
# R_CANDIDATES: address_rescue_rank values that split the candidate list.
# T_LO_STEPS: number of quantile steps for the low threshold search.
R_CANDIDATES = [5, 10, 15, 20, 30, 50]
T_LO_STEPS = 200


def peak_memory_mb():
    try:
        import psutil
        info = psutil.Process().memory_info()
        peak = getattr(info, "peak_wset", None) or getattr(info, "peak_rss", None)
        return peak / (1024 * 1024) if peak is not None else info.rss / (1024 * 1024)
    except Exception:
        return None


def choose_sample(gt, country):
    try:
        return stratified_sample_and_split(
            gt, country, REQUESTED_SAMPLE_PER_COUNTRY,
            SEED + (0 if country == "india" else 1),
        ), REQUESTED_SAMPLE_PER_COUNTRY
    except ValueError as exc:
        print(f"  {country}: 200 S1s unavailable ({exc}); reducing to 100.", flush=True)
        return stratified_sample_and_split(
            gt, country, FALLBACK_SAMPLE_PER_COUNTRY,
            SEED + (0 if country == "india" else 1),
        ), FALLBACK_SAMPLE_PER_COUNTRY


def build_country_index(country):
    idx = CompactBlockingIndex()
    ids = []
    seen = set()
    started = time.perf_counter()
    for source in ("S2", "S3"):
        path = SPLITS / f"train_{source}_{country}.tsv"
        for eid, name, address, ctry in stream_records(path):
            if eid in seen:
                continue
            seen.add(eid)
            pos = len(ids)
            ids.append(eid)
            idx.add(pos, normalize_record(eid, name, address, ctry))
            if len(ids) % 500_000 == 0:
                print(
                    f"  {country}: indexed {len(ids):,} S2+S3 candidates "
                    f"({time.perf_counter()-started:.1f}s)",
                    flush=True,
                )
    elapsed = time.perf_counter() - started
    ids_unique = len(ids) == len(seen)
    del seen
    return idx, ids, elapsed, ids_unique


def load_sampled_s1(country, wanted):
    records = {}
    for eid, name, address, ctry in stream_records(
        SPLITS / f"train_S1_{country}.tsv"
    ):
        if eid in wanted:
            records[eid] = normalize_record(eid, name, address, ctry)
    missing = wanted - records.keys()
    if missing:
        raise RuntimeError(
            f"{country}: could not load {len(missing)} sampled S1 rows"
        )
    return records


def generate_country_candidates(country, s1_records):
    gen_started = time.perf_counter()
    idx, candidate_ids, index_seconds, ids_unique = build_country_index(country)
    query_index = NumpyQueryIndex(idx, candidate_ids, list(s1_records.values()))
    selected = {}
    pool_positions_by_sid = {}
    query_seconds = 0.0
    for number, sid in enumerate(sorted(s1_records), 1):
        s1 = s1_records[sid]
        t0 = time.perf_counter()
        pool, touched = query_index.generate(s1)
        if len(pool) != len(set(pool.tolist())):
            raise RuntimeError(
                f"{sid}: optimized uncapped pool contains duplicate positions"
            )
        top, _ = query_index.rank(pool)
        query_seconds += time.perf_counter() - t0
        pool_positions = set(int(pos) for pos in pool.tolist())
        top_positions = [int(pos) for pos in top]
        top_ids = [candidate_ids[pos] for pos in top_positions]
        if len(top_ids) != len(set(top_ids)):
            raise RuntimeError(
                f"{sid}: selected top-200 contains duplicate candidate IDs"
            )
        if len(top_ids) > CAP or not set(top_positions).issubset(pool_positions):
            raise RuntimeError(
                f"{sid}: invalid top-200 selection or candidate injection detected"
            )
        repeated_top, _ = query_index.rank(pool)
        if [int(pos) for pos in repeated_top] != top_positions:
            raise RuntimeError(f"{sid}: top-200 ranking is not deterministic")
        rows = []
        for rank, pos in enumerate(top_positions, 1):
            blocker = tuple(int(v) for v in query_index.features[pos].tolist())
            score = int(np.asarray(blocker, dtype=np.int32) @ WEIGHTS)
            if score != address_rescue_score(blocker):
                raise RuntimeError(
                    f"{sid}/{candidate_ids[pos]}: Address Rescue score mismatch"
                )
            rows.append((candidate_ids[pos], blocker, score, rank))
        selected[sid] = rows
        pool_positions_by_sid[sid] = np.asarray(pool, dtype=np.int32).copy()
        query_index.clear(touched)
        if number % PROGRESS_EVERY == 0 or number == len(s1_records):
            print(
                f"  {country}: generated candidates for "
                f"{number}/{len(s1_records)} S1s",
                flush=True,
            )
    total_seconds = time.perf_counter() - gen_started
    return (
        selected, pool_positions_by_sid, candidate_ids,
        index_seconds, query_seconds, total_seconds,
        query_index.cache_seconds, ids_unique,
    )


def load_selected_candidate_records(country, selected):
    needed = {cid for rows in selected.values() for cid, _, _, _ in rows}
    records = {}
    for source in ("S2", "S3"):
        for eid, name, address, ctry in stream_records(
            SPLITS / f"train_{source}_{country}.tsv"
        ):
            if eid in needed and eid not in records:
                records[eid] = normalize_record(eid, name, address, ctry)
    missing = needed - records.keys()
    if missing:
        raise RuntimeError(
            f"{country}: {len(missing)} selected candidate records could not be reloaded"
        )
    return records


def predict_sets_rank_aware(
    ids, groups, candidate_ids, ranks, probabilities,
    threshold_high, rank_cutoff, threshold_low,
):
    """Apply a two-level threshold split on address_rescue_rank.

    Candidates with rank <= rank_cutoff use threshold_low.
    Candidates with rank >  rank_cutoff use threshold_high.
    ranks[i] is the address_rescue_rank (1-based) for row i.
    """
    out = {}
    for sid in ids:
        start, end = groups[sid]
        matched = set()
        for i in range(start, end):
            t = threshold_low if ranks[i] <= rank_cutoff else threshold_high
            if probabilities[i] >= t:
                matched.add(candidate_ids[i])
        out[sid] = matched
    return out


def macro_threshold_global(cal_ids, groups, candidate_ids, probabilities, gt):
    """Standard single-threshold search on calibration data (identical to v2)."""
    candidates = np.unique(
        np.quantile(probabilities, np.linspace(0.0, 1.0, 401))
    )
    candidates = np.unique(np.concatenate(([0.0], candidates, [1.000001])))
    best = None
    for threshold in candidates:
        predictions = predict_sets(
            cal_ids, groups, candidate_ids, probabilities, float(threshold)
        )
        metrics = macro_f05(predictions, {sid: gt[sid] for sid in cal_ids})
        key = (metrics["f05_macro"], metrics["precision_macro"], float(threshold))
        if best is None or key > best[0]:
            best = (key, float(threshold), metrics)
    return best[1], best[2]


def macro_threshold_rank_aware(
    cal_ids, groups, candidate_ids, ranks, probabilities, gt, threshold_high,
):
    """Grid search over (rank_cutoff, threshold_low) on calibration data only.

    threshold_high is fixed to the global threshold already found.
    Returns (rank_cutoff, threshold_low, cal_metrics).
    """
    # Build a dense grid of low-threshold candidates from calibration probs.
    t_lo_candidates = np.unique(
        np.quantile(probabilities, np.linspace(0.0, 1.0, T_LO_STEPS + 1))
    )
    # Only consider thresholds <= threshold_high (no point going higher).
    t_lo_candidates = t_lo_candidates[t_lo_candidates <= threshold_high]
    t_lo_candidates = np.unique(
        np.concatenate(([0.0], t_lo_candidates, [threshold_high]))
    )

    best_key = None
    best = None
    gt_cal = {sid: gt[sid] for sid in cal_ids}

    for R in R_CANDIDATES:
        for t_lo in t_lo_candidates:
            predictions = predict_sets_rank_aware(
                cal_ids, groups, candidate_ids, ranks,
                probabilities, threshold_high, R, float(t_lo),
            )
            metrics = macro_f05(predictions, gt_cal)
            # Tiebreak: prefer higher precision, then lower t_lo (more recall).
            key = (
                metrics["f05_macro"],
                metrics["precision_macro"],
                -float(t_lo),
            )
            if best_key is None or key > best_key:
                best_key = key
                best = (R, float(t_lo), metrics)

    return best


def main():
    wall_start = time.perf_counter()
    peak_mb = peak_memory_mb()
    all_gt = {}
    split_ids = {name: set() for name in ("train", "calibration", "validation")}
    categories = {}
    split_arrays = {
        name: {"X": [], "y": [], "sids": [], "cids": [], "groups": {}, "ranks": []}
        for name in split_ids
    }
    sample_counts = {}
    blocking = {
        "true": 0, "uncapped": 0, "top200": 0,
        "never_generated": 0, "cap_dropped": 0,
    }
    uncapped_by_sid = {}
    candidate_generation_seconds = 0.0
    index_seconds_total = 0.0
    query_seconds_total = 0.0
    cache_seconds_total = 0.0
    feature_seconds = 0.0
    max_memory = peak_mb

    for country in COUNTRIES:
        print(f"\n=== {country.upper()} ===", flush=True)
        gt = load_ground_truth(SPLITS / f"train_GT_{country}.tsv")
        (splits, cat_map, quotas), sample_n = choose_sample(gt, country)
        sample_counts[country] = sample_n
        categories.update(cat_map)
        for split_name, ids in splits.items():
            overlap = split_ids[split_name] & ids
            if overlap:
                raise RuntimeError(
                    f"S1 leakage: {len(overlap)} IDs repeated in {split_name}"
                )
            split_ids[split_name].update(ids)
        if (
            splits["train"] & splits["calibration"]
            or splits["train"] & splits["validation"]
            or splits["calibration"] & splits["validation"]
        ):
            raise RuntimeError(f"{country}: S1 appears in multiple splits")
        wanted = set().union(*splits.values())
        print(
            f"  category sample={quotas}; "
            f"split counts={{{', '.join(f'{k}: {len(v)}' for k,v in splits.items())}}}",
            flush=True,
        )
        s1_records = load_sampled_s1(country, wanted)

        (
            selected, pools, candidate_ids,
            index_s, query_s, gen_s, cache_s, ids_unique,
        ) = generate_country_candidates(country, s1_records)
        if not ids_unique:
            raise RuntimeError(
                f"{country}: candidate IDs are not unique in the blocker index"
            )
        candidate_generation_seconds += gen_s
        index_seconds_total += index_s
        query_seconds_total += query_s
        cache_seconds_total += cache_s
        max_memory = max(max_memory or 0, peak_memory_mb() or 0) or max_memory
        print(
            f"  {country}: index={index_s:.1f}s; "
            f"query+rank={query_s:.1f}s; posting cache={cache_s:.1f}s",
            flush=True,
        )

        truth_by_sid = {sid: set(gt[sid]) for sid in wanted}
        sample_truth_ids = (
            set().union(*truth_by_sid.values()) if truth_by_sid else set()
        )
        truth_positions = {
            eid: pos
            for pos, eid in enumerate(candidate_ids)
            if eid in sample_truth_ids
        }
        for sid in sorted(wanted):
            truth = truth_by_sid[sid]
            pool_positions = set(pools[sid].tolist())
            pool_ids = {candidate_ids[pos] for pos in pool_positions}
            top_ids = {row[0] for row in selected[sid]}
            blocking["true"] += len(truth)
            raw_found = sum(
                truth_positions.get(eid, -1) in pool_positions for eid in truth
            )
            capped_found = len(truth & top_ids)
            uncapped_by_sid[sid] = raw_found
            blocking["uncapped"] += raw_found
            blocking["top200"] += capped_found
            blocking["never_generated"] += len(truth - pool_ids)
            blocking["cap_dropped"] += raw_found - capped_found
            all_gt[sid] = truth
        del pools

        feature_started = time.perf_counter()
        candidate_records = load_selected_candidate_records(country, selected)
        sid_to_split = {
            sid: split_name
            for split_name, ids in splits.items()
            for sid in ids
        }
        for sid in sorted(wanted):
            split_name = sid_to_split[sid]
            data = split_arrays[split_name]
            start = len(data["y"])
            truth = truth_by_sid[sid]
            for cid, blocker, score, rank in selected[sid]:
                vector = make_feature_vector(
                    s1_records[sid], candidate_records[cid], blocker, rank
                )
                if len(vector) != len(FEATURES):
                    raise RuntimeError(
                        f"{sid}/{cid}: feature width mismatch "
                        f"{len(vector)} vs {len(FEATURES)}"
                    )
                data["X"].append(vector)
                data["y"].append(int(cid in truth))
                data["sids"].append(sid)
                data["cids"].append(cid)
                data["ranks"].append(rank)
            data["groups"][sid] = (start, len(data["y"]))
        feature_seconds += time.perf_counter() - feature_started
        max_memory = max(max_memory or 0, peak_memory_mb() or 0) or max_memory
        print(
            f"  {country}: extracted pair features in "
            f"{time.perf_counter()-feature_started:.1f}s",
            flush=True,
        )
        del (
            selected, candidate_records, s1_records, candidate_ids,
            truth_positions, sample_truth_ids, truth_by_sid, gt, splits,
        )
        gc.collect()

    train_ids, cal_ids, val_ids = (
        split_ids[k] for k in ("train", "calibration", "validation")
    )
    if train_ids & cal_ids or train_ids & val_ids or cal_ids & val_ids:
        raise RuntimeError(
            "S1 leakage detected across train/calibration/validation"
        )
    expected_total = sum(sample_counts.values())
    if len(train_ids | cal_ids | val_ids) != expected_total:
        raise RuntimeError("Sample/split count mismatch")
    if not cal_ids.isdisjoint(val_ids):
        raise RuntimeError("Validation S1s overlap calibration S1s")

    matrices = {}
    for name, data in split_arrays.items():
        X = np.asarray(data["X"], dtype=np.float32)
        y = np.asarray(data["y"], dtype=np.int8)
        ranks_arr = np.asarray(data["ranks"], dtype=np.int32)
        if (
            X.ndim != 2
            or X.shape[1] != len(FEATURES)
            or not np.isfinite(X).all()
        ):
            raise RuntimeError(f"{name}: invalid/non-finite feature matrix")
        if y.size and not set(np.unique(y)).issubset({0, 1}):
            raise RuntimeError(f"{name}: invalid labels")
        matrices[name] = (
            X, y, data["groups"], data["sids"], data["cids"], ranks_arr,
        )
        print(
            f"\n{name}: S1={len(split_ids[name])}; pairs={len(y):,}; "
            f"positive={int(y.sum()):,}; negative={len(y)-int(y.sum()):,}",
            flush=True,
        )
    Xtr, ytr, train_groups, train_row_sids, train_cids, train_ranks = matrices["train"]
    Xcal, ycal, cal_groups, cal_row_sids, cal_cids, cal_ranks = matrices["calibration"]
    Xval, yval, val_groups, val_row_sids, val_cids, val_ranks = matrices["validation"]
    if len(np.unique(ytr)) != 2:
        raise RuntimeError(
            "Training split does not contain both positive and negative candidate pairs"
        )

    # Split identity assertions (same as v3 LightGBM run).
    assert len(train_ids) == 236, f"train S1 count: {len(train_ids)}"
    assert len(cal_ids)   == 78,  f"cal S1 count: {len(cal_ids)}"
    assert len(val_ids)   == 86,  f"val S1 count: {len(val_ids)}"
    assert Xtr.shape  == (47200, 40), f"train shape: {Xtr.shape}"
    assert Xcal.shape == (15600, 40), f"cal shape: {Xcal.shape}"
    assert Xval.shape == (17200, 40), f"val shape: {Xval.shape}"
    assert int(ytr.sum())  == 721, f"train pos: {int(ytr.sum())}"
    assert int(ycal.sum()) == 228, f"cal pos: {int(ycal.sum())}"
    assert int(yval.sum()) == 265, f"val pos: {int(yval.sum())}"
    print("\nSplit identity check vs LR/LightGBM baselines: PASSED", flush=True)

    # ── Train identical LR model ─────────────────────────────────────────────
    print("\nFitting class-weighted Logistic Regression (identical to v2)", flush=True)
    training_started = time.perf_counter()
    model, scaler = train_logistic_regression(Xtr, ytr, feature_names=FEATURES)
    training_seconds = time.perf_counter() - training_started

    # ── Step 1: find global threshold on calibration (identical to v2) ───────
    inference_started = time.perf_counter()
    pcal = predict_proba(model, Xcal, scaler=scaler)
    cal_inference_seconds = time.perf_counter() - inference_started

    threshold_started = time.perf_counter()
    cal_id_list = sorted(cal_ids)
    threshold_global, cal_metrics_global = macro_threshold_global(
        cal_id_list, cal_groups, cal_cids, pcal, all_gt,
    )
    if not np.isfinite(threshold_global):
        raise RuntimeError("Global calibration produced an invalid threshold")
    print(
        f"\nGlobal threshold (v2 baseline): {threshold_global:.6f}; "
        f"calibration macro F0.5={cal_metrics_global['f05_macro']:.4f}",
        flush=True,
    )

    # ── Step 2: rank-aware grid search on calibration only ───────────────────
    best_R, best_t_lo, cal_metrics_rank = macro_threshold_rank_aware(
        cal_id_list, cal_groups, cal_cids, cal_ranks, pcal, all_gt,
        threshold_high=threshold_global,
    )
    threshold_seconds = time.perf_counter() - threshold_started
    print(
        f"Rank-aware threshold: rank_cutoff={best_R}; "
        f"threshold_low={best_t_lo:.6f}; threshold_high={threshold_global:.6f}; "
        f"calibration macro F0.5={cal_metrics_rank['f05_macro']:.4f}",
        flush=True,
    )

    # Freeze both thresholds — validation labels never consulted.
    frozen_threshold_high = threshold_global
    frozen_threshold_low = best_t_lo
    frozen_R = best_R

    if not cal_ids.isdisjoint(val_ids):
        raise RuntimeError("Calibration/validation leakage before inference")

    # ── Validation: global threshold (v2 baseline reproduced) ────────────────
    inference_started = time.perf_counter()
    pval = predict_proba(model, Xval, scaler=scaler)
    validation_inference_seconds = time.perf_counter() - inference_started

    val_id_list = sorted(val_ids)
    val_gt = {sid: all_gt[sid] for sid in val_id_list}

    predictions_global = predict_sets(
        val_id_list, val_groups, val_cids, pval, frozen_threshold_high,
    )
    metrics_global = macro_f05(predictions_global, val_gt)

    # ── Validation: rank-aware threshold ─────────────────────────────────────
    predictions_rank = predict_sets_rank_aware(
        val_id_list, val_groups, val_cids, val_ranks, pval,
        frozen_threshold_high, frozen_R, frozen_threshold_low,
    )
    metrics_rank = macro_f05(predictions_rank, val_gt)

    # ── Compute detailed stats for rank-aware result ──────────────────────────
    tp = fp = fn = exact = 0
    false_pos = []
    false_neg = []
    for sid in val_id_list:
        truth, pred = val_gt[sid], predictions_rank[sid]
        tp += len(truth & pred)
        fp += len(pred - truth)
        fn += len(truth - pred)
        exact += int(pred == truth)
        false_pos.extend((sid, cid) for cid in pred - truth)
        false_neg.extend((sid, cid) for cid in truth - pred)

    val_true = sum(len(val_gt[sid]) for sid in val_id_list)
    val_top200 = 0
    model_failures = 0
    blocking_failures = 0
    group_rows = matrices["validation"][2]
    cids_rows = matrices["validation"][4]
    for sid in val_id_list:
        start, end = group_rows[sid]
        top_ids = set(cids_rows[start:end])
        top_hits = len(val_gt[sid] & top_ids)
        val_top200 += top_hits
        blocking_failures += len(val_gt[sid] - top_ids)
        model_failures += top_hits - len(val_gt[sid] & predictions_rank[sid])

    if tp + fp != sum(len(predictions_rank[s]) for s in val_id_list):
        raise RuntimeError("Pair-level metric accounting mismatch")
    if blocking_failures + model_failures != fn:
        raise RuntimeError(
            "Blocking/model failure accounting does not equal false negatives"
        )

    macro_counts = {
        "singleton": sum(not val_gt[sid] for sid in val_id_list),
        "single":    sum(len(val_gt[sid]) == 1 for sid in val_id_list),
        "multi":     sum(len(val_gt[sid]) > 1 for sid in val_id_list),
    }
    total_seconds = time.perf_counter() - wall_start
    max_memory = max(max_memory or 0, peak_memory_mb() or 0) or max_memory

    # ── Report ────────────────────────────────────────────────────────────────
    print("\n=== HELD-OUT TRAINING-DATA VALIDATION (v5 rank-aware threshold) ===")
    print(
        f"Sampled S1s: India={sample_counts['india']}; "
        f"US={sample_counts['us']}; total={expected_total}"
    )
    print(
        f"Split S1s: train={len(train_ids)}; calibration={len(cal_ids)}; "
        f"validation={len(val_ids)}; overlap=0"
    )
    print(
        f"Validation categories: singleton={macro_counts['singleton']}; "
        f"single-match={macro_counts['single']}; multi-match={macro_counts['multi']}"
    )
    print(
        f"Frozen thresholds (calibration-only): "
        f"rank_cutoff={frozen_R}; "
        f"threshold_low={frozen_threshold_low:.6f}; "
        f"threshold_high={frozen_threshold_high:.6f}"
    )
    print(
        f"Calibration macro F0.5: "
        f"global={cal_metrics_global['f05_macro']:.4f}  "
        f"rank-aware={cal_metrics_rank['f05_macro']:.4f}"
    )
    print(
        f"Validation: macro F0.5={metrics_rank['f05_macro']:.4f}; "
        f"macro precision={metrics_rank['precision_macro']:.4f}; "
        f"macro recall={metrics_rank['recall_macro']:.4f}"
    )
    print(f"Validation exact-set recovery: {exact}/{len(val_id_list)}")
    print(f"Pair totals: TP={tp:,}; FP={fp:,}; FN={fn:,}")
    print(
        f"Blocking recall at top-200: {val_top200}/{val_true} "
        f"({val_top200/val_true if val_true else 1:.2%})"
    )
    print(
        f"Blocking failures (true pair absent from top-200): {blocking_failures}"
    )
    print(
        f"Matching-model failures (true pair in top-200 but rejected): "
        f"{model_failures}"
    )
    print(f"False positives={len(false_pos)}; false negatives={len(false_neg)}")
    print(
        f"Features: {len(FEATURES)} total "
        f"({len(FEATURES)-13} pairwise + 13 existing blocker/rank features)"
    )

    print("\n=== COMPARISON vs CAP=200 LR BASELINE (v2) ===")
    BL_F05, BL_PREC, BL_REC = 0.7388, 0.8971, 0.6519
    BL_EXACT, BL_TP, BL_FP, BL_FN = 24, 194, 24, 101
    BL_BLK, BL_MOD = 30, 71
    m = metrics_rank
    print(f"  {'Metric':<28} {'v2 baseline':>12} {'v5 rank-aware':>14} {'delta':>8}")
    print(f"  {'-'*64}")
    print(f"  {'macro F0.5':<28} {BL_F05:>12.4f} {m['f05_macro']:>14.4f} {m['f05_macro']-BL_F05:>+8.4f}")
    print(f"  {'macro precision':<28} {BL_PREC:>12.4f} {m['precision_macro']:>14.4f} {m['precision_macro']-BL_PREC:>+8.4f}")
    print(f"  {'macro recall':<28} {BL_REC:>12.4f} {m['recall_macro']:>14.4f} {m['recall_macro']-BL_REC:>+8.4f}")
    print(f"  {'exact-set recovery':<28} {BL_EXACT}/86       {exact}/86          {exact-BL_EXACT:>+8d}")
    print(f"  {'TP':<28} {BL_TP:>12d} {tp:>14d} {tp-BL_TP:>+8d}")
    print(f"  {'FP':<28} {BL_FP:>12d} {fp:>14d} {fp-BL_FP:>+8d}")
    print(f"  {'FN':<28} {BL_FN:>12d} {fn:>14d} {fn-BL_FN:>+8d}")
    print(f"  {'blocking recall':<28} {'89.83%':>12} {val_top200/val_true if val_true else 1:>14.2%} {val_top200/val_true - 0.8983 if val_true else 0:>+8.2%}")
    print(f"  {'blocking failures':<28} {BL_BLK:>12d} {blocking_failures:>14d} {blocking_failures-BL_BLK:>+8d}")
    print(f"  {'model failures':<28} {BL_MOD:>12d} {model_failures:>14d} {model_failures-BL_MOD:>+8d}")

    # Also show reproduced global-threshold result for sanity check.
    print(
        f"\nReproduced global-threshold result on same model: "
        f"F0.5={metrics_global['f05_macro']:.4f} "
        f"(expected 0.7388; delta={metrics_global['f05_macro']-0.7388:+.4f})"
    )

    verdict = (
        "IMPROVEMENT" if m["f05_macro"] > BL_F05
        else "NO IMPROVEMENT — revert to v2 baseline"
    )
    print(f"\nVerdict: {verdict}")

    print("\nRUNTIME")
    print(
        f"  Candidate generation total: {candidate_generation_seconds:.1f}s"
    )
    print(
        f"    index: {index_seconds_total:.1f}s; "
        f"query+rank: {query_seconds_total:.1f}s; "
        f"cache: {cache_seconds_total:.1f}s"
    )
    print(f"  Feature extraction: {feature_seconds:.1f}s")
    print(f"  Model training: {training_seconds:.2f}s")
    print(
        f"  Inference (cal + val): "
        f"{cal_inference_seconds+validation_inference_seconds:.2f}s"
    )
    print(f"  Threshold calibration (global + rank-aware grid): {threshold_seconds:.2f}s")
    print(f"  Total wall time: {total_seconds:.1f}s")
    if max_memory is not None:
        print(f"  Peak/approximate process memory: {max_memory:.1f} MiB")
    print(
        "\nCorrectness checks: candidate pools deduplicated; candidate IDs unique; "
        "top-200 generated-only and deterministic; no GT injection; "
        "S1 splits disjoint; thresholds selected only from calibration labels."
    )
    print(
        "Scope: training files only; no test inference, model persistence, "
        "production changes, or submission files."
    )


if __name__ == "__main__":
    main()
