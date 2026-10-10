#!/usr/bin/env python3
"""Held-out experimental matching model using the parity-fixed Address Rescue.

Training, calibration, and validation are split by S1 entity. Ground truth is
used for sample stratification and then for labels/metrics only; it is never
passed to the candidate generator and no true pair is added to a candidate set.
This script does not write models or submission files and does not modify the
production pipeline.
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
from src.pipeline.run_pipeline import CompactBlockingIndex, load_ground_truth, stream_records  # noqa: E402
from src.preprocessing.normalization import normalize_record  # noqa: E402

COUNTRIES = ("india", "us")
REQUESTED_SAMPLE_PER_COUNTRY = 200
FALLBACK_SAMPLE_PER_COUNTRY = 100
SEED = 42
PROGRESS_EVERY = 50


def peak_memory_mb():
    """Return process peak working set when psutil exposes it; otherwise None."""
    try:
        import psutil
        info = psutil.Process().memory_info()
        peak = getattr(info, "peak_wset", None) or getattr(info, "peak_rss", None)
        return peak / (1024 * 1024) if peak is not None else info.rss / (1024 * 1024)
    except Exception:
        return None


def choose_sample(gt, country):
    """Try 200 S1s; fall back to 100 only if the labelled population is smaller."""
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
    """Build the unchanged seven-strategy blocker once for this country."""
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
                print(f"  {country}: indexed {len(ids):,} S2+S3 candidates ({time.perf_counter()-started:.1f}s)", flush=True)
    elapsed = time.perf_counter() - started
    # The same `seen` set is the index-time uniqueness check; IDs are inserted once.
    ids_unique = len(ids) == len(seen)
    del seen
    return idx, ids, elapsed, ids_unique


def load_sampled_s1(country, wanted):
    records = {}
    for eid, name, address, ctry in stream_records(SPLITS / f"train_S1_{country}.tsv"):
        if eid in wanted:
            records[eid] = normalize_record(eid, name, address, ctry)
    missing = wanted - records.keys()
    if missing:
        raise RuntimeError(f"{country}: could not load {len(missing)} sampled S1 rows")
    return records


def generate_country_candidates(country, s1_records):
    """Use the corrected optimized aggregator/ranker; GT is not read here."""
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
            raise RuntimeError(f"{sid}: optimized uncapped pool contains duplicate positions")
        top, _ = query_index.rank(pool)
        query_seconds += time.perf_counter() - t0
        pool_positions = set(int(pos) for pos in pool.tolist())
        top_positions = [int(pos) for pos in top]
        top_ids = [candidate_ids[pos] for pos in top_positions]
        if len(top_ids) != len(set(top_ids)):
            raise RuntimeError(f"{sid}: selected top-200 contains duplicate candidate IDs")
        if len(top_ids) > CAP or not set(top_positions).issubset(pool_positions):
            raise RuntimeError(f"{sid}: invalid top-200 selection or candidate injection detected")
        if not set(top_positions).issubset(pool_positions):
            raise RuntimeError(f"{sid}: selected candidate was absent from generated pool")
        repeated_top, _ = query_index.rank(pool)
        if [int(pos) for pos in repeated_top] != top_positions:
            raise RuntimeError(f"{sid}: top-200 ranking is not deterministic")
        # Cache presence is checked by comparing raw blocker components to the index
        # buffers before clearing them; the generated top list stores those components.
        rows = []
        for rank, pos in enumerate(top_positions, 1):
            blocker = tuple(int(v) for v in query_index.features[pos].tolist())
            score = int(np.asarray(blocker, dtype=np.int32) @ WEIGHTS)
            if score != address_rescue_score(blocker):
                raise RuntimeError(f"{sid}/{candidate_ids[pos]}: Address Rescue score mismatch")
            rows.append((candidate_ids[pos], blocker, score, rank))
        selected[sid] = rows
        pool_positions_by_sid[sid] = np.asarray(pool, dtype=np.int32).copy()
        query_index.clear(touched)
        if number % PROGRESS_EVERY == 0 or number == len(s1_records):
            print(f"  {country}: generated candidates for {number}/{len(s1_records)} S1s", flush=True)
    total_seconds = time.perf_counter() - gen_started
    return selected, pool_positions_by_sid, candidate_ids, index_seconds, query_seconds, total_seconds, query_index.cache_seconds, ids_unique


def load_selected_candidate_records(country, selected):
    needed = {cid for rows in selected.values() for cid, _, _, _ in rows}
    records = {}
    for source in ("S2", "S3"):
        for eid, name, address, ctry in stream_records(SPLITS / f"train_{source}_{country}.tsv"):
            if eid in needed and eid not in records:
                records[eid] = normalize_record(eid, name, address, ctry)
    missing = needed - records.keys()
    if missing:
        raise RuntimeError(f"{country}: {len(missing)} selected candidate records could not be reloaded")
    return records


def macro_threshold(cal_ids, groups, candidate_ids, probabilities, gt):
    """Choose threshold using calibration S1 labels only; deterministic tie rule."""
    candidates = np.unique(np.quantile(probabilities, np.linspace(0.0, 1.0, 401)))
    candidates = np.unique(np.concatenate(([0.0], candidates, [1.000001])))
    best = None
    for threshold in candidates:
        predictions = predict_sets(cal_ids, groups, candidate_ids, probabilities, float(threshold))
        metrics = macro_f05(predictions, {sid: gt[sid] for sid in cal_ids})
        key = (metrics["f05_macro"], metrics["precision_macro"], float(threshold))
        if best is None or key > best[0]:
            best = (key, float(threshold), metrics)
    return best[1], best[2]


def main():
    wall_start = time.perf_counter()
    peak_mb = peak_memory_mb()
    all_gt = {}
    split_ids = {name: set() for name in ("train", "calibration", "validation")}
    categories = {}
    split_arrays = {
        name: {"X": [], "y": [], "sids": [], "cids": [], "groups": {}}
        for name in split_ids
    }
    sample_counts = {}
    blocking = {"true": 0, "uncapped": 0, "top200": 0, "never_generated": 0, "cap_dropped": 0}
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
                raise RuntimeError(f"S1 leakage: {len(overlap)} IDs repeated in {split_name}")
            split_ids[split_name].update(ids)
        if (splits["train"] & splits["calibration"] or splits["train"] & splits["validation"]
                or splits["calibration"] & splits["validation"]):
            raise RuntimeError(f"{country}: S1 appears in multiple splits")
        wanted = set().union(*splits.values())
        print(f"  category sample={quotas}; split counts={{{', '.join(f'{k}: {len(v)}' for k,v in splits.items())}}}", flush=True)
        s1_records = load_sampled_s1(country, wanted)

        selected, pools, candidate_ids, index_s, query_s, gen_s, cache_s, ids_unique = generate_country_candidates(
            country, s1_records,
        )
        if not ids_unique:
            raise RuntimeError(f"{country}: candidate IDs are not unique in the blocker index")
        candidate_generation_seconds += gen_s
        index_seconds_total += index_s
        query_seconds_total += query_s
        cache_seconds_total += cache_s
        max_memory = max(max_memory or 0, peak_memory_mb() or 0) or max_memory
        print(f"  {country}: index={index_s:.1f}s; query+rank={query_s:.1f}s; posting cache={cache_s:.1f}s", flush=True)

        # GT is first consulted after all candidate pools and top-200 lists exist.
        truth_by_sid = {sid: set(gt[sid]) for sid in wanted}
        sample_truth_ids = set().union(*truth_by_sid.values()) if truth_by_sid else set()
        truth_positions = {
            eid: pos for pos, eid in enumerate(candidate_ids)
            if eid in sample_truth_ids
        }
        for sid in sorted(wanted):
            truth = truth_by_sid[sid]
            pool_positions = set(pools[sid].tolist())
            pool_ids = {candidate_ids[pos] for pos in pool_positions}
            top_ids = {row[0] for row in selected[sid]}
            blocking["true"] += len(truth)
            raw_found = sum(truth_positions.get(eid, -1) in pool_positions for eid in truth)
            capped_found = len(truth & top_ids)
            uncapped_by_sid[sid] = raw_found
            blocking["uncapped"] += raw_found
            blocking["top200"] += capped_found
            blocking["never_generated"] += len(truth - pool_ids)
            blocking["cap_dropped"] += raw_found - capped_found
            all_gt[sid] = truth
        del pools

        # Candidate source rows are reloaded only for the already selected top-200.
        feature_started = time.perf_counter()
        candidate_records = load_selected_candidate_records(country, selected)
        sid_to_split = {sid: split_name for split_name, ids in splits.items() for sid in ids}
        for sid in sorted(wanted):
            split_name = sid_to_split[sid]
            data = split_arrays[split_name]
            start = len(data["y"])
            truth = truth_by_sid[sid]  # labels assigned after candidate generation
            for cid, blocker, score, rank in selected[sid]:
                vector = make_feature_vector(s1_records[sid], candidate_records[cid], blocker, rank)
                if len(vector) != len(FEATURES):
                    raise RuntimeError(f"{sid}/{cid}: feature width mismatch {len(vector)} vs {len(FEATURES)}")
                data["X"].append(vector)
                data["y"].append(int(cid in truth))
                data["sids"].append(sid)
                data["cids"].append(cid)
            data["groups"][sid] = (start, len(data["y"]))
        feature_seconds += time.perf_counter() - feature_started
        max_memory = max(max_memory or 0, peak_memory_mb() or 0) or max_memory
        print(f"  {country}: extracted pair features in {time.perf_counter()-feature_started:.1f}s", flush=True)
        del selected, candidate_records, s1_records, candidate_ids, truth_positions, sample_truth_ids, truth_by_sid, gt, splits
        gc.collect()

    # Global leakage/scope checks before fitting.
    train_ids, cal_ids, val_ids = (split_ids[k] for k in ("train", "calibration", "validation"))
    if train_ids & cal_ids or train_ids & val_ids or cal_ids & val_ids:
        raise RuntimeError("S1 leakage detected across train/calibration/validation")
    expected_total = sum(sample_counts.values())
    if len(train_ids | cal_ids | val_ids) != expected_total:
        raise RuntimeError("Sample/split count mismatch")
    if not cal_ids.isdisjoint(val_ids):
        raise RuntimeError("Validation S1s overlap calibration S1s")

    matrices = {}
    for name, data in split_arrays.items():
        X = np.asarray(data["X"], dtype=np.float32)
        y = np.asarray(data["y"], dtype=np.int8)
        if X.ndim != 2 or X.shape[1] != len(FEATURES) or not np.isfinite(X).all():
            raise RuntimeError(f"{name}: invalid/non-finite feature matrix")
        if y.size and not set(np.unique(y)).issubset({0, 1}):
            raise RuntimeError(f"{name}: invalid labels")
        matrices[name] = (X, y, data["groups"], data["sids"], data["cids"])
        print(f"\n{name}: S1={len(split_ids[name])}; pairs={len(y):,}; positive={int(y.sum()):,}; negative={len(y)-int(y.sum()):,}", flush=True)
    Xtr, ytr, train_groups, train_row_sids, train_cids = matrices["train"]
    Xcal, ycal, cal_groups, cal_row_sids, cal_cids = matrices["calibration"]
    Xval, yval, val_groups, val_row_sids, val_cids = matrices["validation"]
    if len(np.unique(ytr)) != 2:
        raise RuntimeError("Training split does not contain both positive and negative candidate pairs")

    print("\nFitting class-weighted Logistic Regression", flush=True)
    training_started = time.perf_counter()
    model, scaler = train_logistic_regression(Xtr, ytr, feature_names=FEATURES)
    training_seconds = time.perf_counter() - training_started

    # Predict calibration probabilities, then freeze a threshold using calibration only.
    inference_started = time.perf_counter()
    pcal = predict_proba(model, Xcal, scaler=scaler)
    cal_inference_seconds = time.perf_counter() - inference_started
    threshold_started = time.perf_counter()
    cal_id_list = sorted(cal_ids)
    threshold, cal_metrics = macro_threshold(cal_id_list, cal_groups, cal_cids, pcal, all_gt)
    threshold_seconds = time.perf_counter() - threshold_started
    if not np.isfinite(threshold):
        raise RuntimeError("Calibration produced an invalid threshold")
    frozen_threshold = threshold
    if not cal_ids.isdisjoint(val_ids):
        raise RuntimeError("Calibration/validation leakage before inference")

    # Validation labels are not used to choose the already-frozen threshold.
    inference_started = time.perf_counter()
    pval = predict_proba(model, Xval, scaler=scaler)
    validation_inference_seconds = time.perf_counter() - inference_started
    val_id_list = sorted(val_ids)
    predictions = predict_sets(val_id_list, val_groups, val_cids, pval, frozen_threshold)
    val_gt = {sid: all_gt[sid] for sid in val_id_list}
    metrics = macro_f05(predictions, val_gt)
    tp = fp = fn = 0
    exact = 0
    false_pos = []
    false_neg = []
    for sid in val_id_list:
        truth, pred = val_gt[sid], predictions[sid]
        tp += len(truth & pred)
        fp += len(pred - truth)
        fn += len(truth - pred)
        exact += int(pred == truth)
        false_pos.extend((sid, cid) for cid in pred - truth)
        false_neg.extend((sid, cid) for cid in truth - pred)

    val_true = sum(len(val_gt[sid]) for sid in val_id_list)
    val_uncapped = sum(uncapped_by_sid[sid] for sid in val_id_list)
    val_top200 = 0
    model_failures = 0
    blocking_failures = 0
    # Recover candidate memberships from the labels in the top-200 feature rows.
    val_rows = matrices["validation"]
    group_rows = val_rows[2]
    sids_rows, cids_rows = val_rows[3], val_rows[4]
    for sid in val_id_list:
        start, end = group_rows[sid]
        top_ids = set(cids_rows[start:end])
        top_hits = len(val_gt[sid] & top_ids)
        val_top200 += top_hits
        # Blocking category A includes anything absent from the retained top-200.
        blocking_failures += len(val_gt[sid] - top_ids)
        model_failures += top_hits - len(val_gt[sid] & predictions[sid])
    # Keep top-200 as the blocking boundary for model diagnosis; also report
    # uncapped pool recall to distinguish never-generated from cap-dropped pairs.

    # Integrity: all labels came from ground truth membership in generated top-200 IDs.
    if tp + fp != sum(len(predictions[s]) for s in val_id_list):
        raise RuntimeError("Pair-level metric accounting mismatch")
    if blocking_failures + model_failures != fn:
        raise RuntimeError("Blocking/model failure accounting does not equal false negatives")
    if threshold != frozen_threshold:
        raise RuntimeError("Calibration threshold changed after being frozen")

    macro_counts = {
        "singleton": sum(not val_gt[sid] for sid in val_id_list),
        "single": sum(len(val_gt[sid]) == 1 for sid in val_id_list),
        "multi": sum(len(val_gt[sid]) > 1 for sid in val_id_list),
    }
    total_seconds = time.perf_counter() - wall_start
    max_memory = max(max_memory or 0, peak_memory_mb() or 0) or max_memory

    print("\n=== HELD-OUT TRAINING-DATA VALIDATION ===")
    print(f"Sampled S1s: India={sample_counts['india']}; US={sample_counts['us']}; total={expected_total}")
    print(f"Split S1s: train={len(train_ids)}; calibration={len(cal_ids)}; validation={len(val_ids)}; overlap=0")
    print(f"Validation categories: singleton={macro_counts['singleton']}; single-match={macro_counts['single']}; multi-match={macro_counts['multi']}")
    print(f"Calibration-only frozen threshold: {frozen_threshold:.6f}; calibration macro F0.5={cal_metrics['f05_macro']:.4f}")
    print(f"Validation: macro F0.5={metrics['f05_macro']:.4f}; macro precision={metrics['precision_macro']:.4f}; macro recall={metrics['recall_macro']:.4f}")
    print(f"Validation exact-set recovery: {exact}/{len(val_id_list)}")
    print(f"Pair totals: TP={tp:,}; FP={fp:,}; FN={fn:,}")
    print(f"Blocking recall at top-200: {val_top200}/{val_true} ({val_top200/val_true if val_true else 1:.2%})")
    print(f"Blocking failures (true pair absent from top-200): {blocking_failures}")
    print(f"Matching-model failures (true pair in top-200 but rejected): {model_failures}")
    print(f"False positives={len(false_pos)}; false negatives={len(false_neg)}")
    print(f"Features: {len(FEATURES)} total ({len(FEATURES)-13} pairwise + 13 existing blocker/rank features)")
    print("\nRUNTIME")
    print(f"  Candidate generation total (index + optimized queries): {candidate_generation_seconds:.1f}s")
    print(f"    index construction: {index_seconds_total:.1f}s; query+ranking: {query_seconds_total:.1f}s; posting-cache setup: {cache_seconds_total:.1f}s")
    print(f"  Feature extraction (selected-record reload + pairs): {feature_seconds:.1f}s")
    print(f"  Model training: {training_seconds:.2f}s")
    print(f"  Inference (calibration + validation): {cal_inference_seconds+validation_inference_seconds:.2f}s")
    print(f"  Threshold calibration: {threshold_seconds:.2f}s")
    print(f"  Total wall time: {total_seconds:.1f}s")
    print(f"  Peak/approximate process memory: {max_memory:.1f} MiB" if max_memory is not None else "  Peak/approximate process memory: unavailable")
    print("\nCorrectness checks: candidate pools deduplicated; candidate IDs unique; top-200 generated-only and deterministic; no GT injection; S1 splits disjoint; threshold selected only from calibration labels.")
    print("Scope: training files only; no test inference, model persistence, production changes, or submission files.")


if __name__ == "__main__":
    main()
