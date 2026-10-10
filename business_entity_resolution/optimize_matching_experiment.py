#!/usr/bin/env python3
"""Benchmark blocker/ranking optimizations without changing production code.

The baseline is the deterministic Address Rescue path in
experiment_matching_model.py. Both implementations share a single
CompactBlockingIndex per country so index construction is not repeated.
The optimized path converts only postings touched by the fixed 40-query
population to NumPy arrays, aggregates signal counts into reusable buffers,
and uses a deterministic partial top-k with candidate ID as the final key.
"""
from __future__ import annotations

import heapq
import sys
import time
from collections import defaultdict
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
    CAP, address_rescue_score, score_candidate_features,
)
from src.pipeline.run_pipeline import (  # noqa: E402
    CompactBlockingIndex, load_ground_truth, stream_records,
)
from src.preprocessing.normalization import normalize_record  # noqa: E402

SPLITS = ROOT / "splits"
COUNTRIES = ("india", "us")
N_PER_COUNTRY = 20
N_DIAGNOSTIC = 5
WEIGHTS = np.asarray([8, 7, 2, 8, 6, 6, 1], dtype=np.int32)
SIGNAL_NAMES = (
    "first_token", "name_prefix", "shared_name_tokens", "shared_address_numbers",
    "shared_address_tokens", "consonant_skeleton", "shared_consonant_ngrams",
)
TARGET_ID = "S1-716574008"
INDEX_PROGRESS_EVERY = 500_000


def read_fixed_queries(country: str):
    """Take the first 20 non-singleton S1 records in source-file order."""
    s1_path = SPLITS / f"train_S1_{country}.tsv"
    gt_path = SPLITS / f"train_GT_{country}.tsv"
    gt = load_ground_truth(gt_path)
    chosen = []
    for eid, name, address, ctry in stream_records(s1_path):
        matches = gt.get(eid, set())
        if matches:
            chosen.append((normalize_record(eid, name, address, ctry), matches))
            if len(chosen) == N_PER_COUNTRY:
                break
    if len(chosen) != N_PER_COUNTRY:
        raise RuntimeError(f"{country}: expected {N_PER_COUNTRY} matched S1s; got {len(chosen)}")
    return chosen, gt


def build_index(country: str):
    """Build one production blocker index and the matching ID array."""
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
            idx.add(len(ids), normalize_record(eid, name, address, ctry))
            ids.append(eid)
            if len(ids) % INDEX_PROGRESS_EVERY == 0:
                print(f"  {country}: indexed {len(ids):,} candidates, {time.perf_counter()-started:.1f}s", flush=True)
    elapsed = time.perf_counter() - started
    print(f"  {country}: index ready, {len(ids):,} candidates, {elapsed:.1f}s", flush=True)
    return idx, ids, elapsed


def baseline_one(s1, idx, ids, measure=False):
    t0 = time.perf_counter() if measure else 0.0
    features, pool = score_candidate_features(s1, idx)
    gen_s = time.perf_counter() - t0 if measure else 0.0
    t0 = time.perf_counter() if measure else 0.0
    selected = heapq.nsmallest(
        CAP, pool,
        key=lambda pos: (-address_rescue_score(features[pos]), ids[pos]),
    )
    rank_s = time.perf_counter() - t0 if measure else 0.0
    return features, pool, selected, gen_s, rank_s


def query_keys(s1):
    return {
        "first_token": {s1["ft"]} if s1.get("ft") else set(),
        "name_prefix": {s1["npfx"]} if s1.get("npfx") else set(),
        "name_tokens": set(s1.get("ntoks", ())),
        "addr_nums": set(s1.get("anums", ())),
        "addr_tokens": set(s1.get("atoks", ())) if len(s1.get("atoks", ())) >= 2 else set(),
        "cskel3": {s1["cskel3"]} if len(s1.get("cskel3", "")) >= 3 else set(),
        "cngrams": set(s1.get("cngrams", ())) if len(s1.get("cngrams", ())) >= 2 else set(),
    }


class NumpyQueryIndex:
    """Query-focused array view over the unchanged CompactBlockingIndex."""
    MAPS = (
        ("first_token", 0, "binary"),
        ("name_prefix", 1, "binary"),
        ("name_tokens", 2, "count"),
        ("addr_nums", 3, "count"),
        ("addr_tokens", 4, "count"),
        ("cskel3", 5, "binary"),
        ("cngrams", 6, "count"),
    )

    def __init__(self, idx, ids, queries):
        n = len(ids)
        all_keys = {name: set() for name, _, _ in self.MAPS}
        for s1 in queries:
            for name, keys in query_keys(s1).items():
                all_keys[name].update(keys)
        started = time.perf_counter()
        self.postings = {}
        for name, _, _ in self.MAPS:
            source = getattr(idx, name)
            self.postings[name] = {
                key: np.fromiter(source.get(key, ()), dtype=np.int32)
                for key in all_keys[name]
            }
        self.features = np.zeros((n, 7), dtype=np.uint16)
        self.touched_mask = np.zeros(n, dtype=np.bool_)
        self.n = n
        self.ids = ids
        self.cache_seconds = time.perf_counter() - started

    def _mark_and_apply(self, posting, col, kind, touched):
        if posting.size == 0:
            return
        new = posting[~self.touched_mask[posting]]
        if new.size:
            # One posting list can contain the same position more than once
            # (for example, repeated address-number tokens). Keep its signal
            # multiplicity for np.add.at below, but record each candidate only
            # once in the uncapped pool / touched list.
            new = np.unique(new)
            touched.extend(new.tolist())
            self.touched_mask[posting] = True
        if kind == "binary":
            self.features[posting, col] = 1
        else:
            np.add.at(self.features[:, col], posting, 1)

    def generate(self, s1):
        touched = []
        keys = query_keys(s1)
        for name, col, kind in self.MAPS:
            postmap = self.postings[name]
            for key in keys[name]:
                self._mark_and_apply(postmap.get(key, _EMPTY), col, kind, touched)
        if not touched:
            return np.empty(0, dtype=np.int32), touched
        positions = np.asarray(touched, dtype=np.int32)
        f = self.features[positions]
        keep = (f[:, 0] > 0) | (f[:, 1] > 0) | (f[:, 2] > 0) | (f[:, 3] > 0) | (f[:, 5] > 0) | (f[:, 4] >= 2) | (f[:, 6] >= 2)
        pool = positions[keep]
        return pool, touched

    def rank(self, pool):
        if pool.size == 0:
            return [], np.empty(0, dtype=np.int32)
        scores = self.features[pool].astype(np.int32) @ WEIGHTS
        take = min(CAP, pool.size)
        if pool.size <= CAP:
            chosen_local = np.arange(pool.size)
        else:
            threshold = np.partition(scores, pool.size - take)[pool.size - take]
            above = np.flatnonzero(scores > threshold)
            ties = np.flatnonzero(scores == threshold)
            remaining = take - above.size
            if remaining:
                tie_positions = pool[ties]
                tie_local = heapq.nsmallest(
                    remaining, range(ties.size),
                    key=lambda i: self.ids[int(tie_positions[i])],
                )
                chosen_local = np.concatenate((above, ties[np.asarray(tie_local, dtype=np.int32)]))
            else:
                chosen_local = above
        selected_positions = pool[chosen_local]
        selected_scores = scores[chosen_local]
        order = sorted(
            range(selected_positions.size),
            key=lambda i: (-int(selected_scores[i]), self.ids[int(selected_positions[i])]),
        )
        ranked = selected_positions[np.asarray(order, dtype=np.int32)]
        return ranked.tolist(), ranked

    def clear(self, touched):
        pos = np.asarray(touched, dtype=np.int32)
        self.features[pos] = 0
        self.touched_mask[pos] = False


_EMPTY = np.empty(0, dtype=np.int32)


def run_country(country):
    print(f"\n{country.upper()}: selecting fixed diagnostic population", flush=True)
    queries, gt = read_fixed_queries(country)
    query_records = [s1 for s1, _ in queries]
    idx, ids, index_seconds = build_index(country)
    truth_by_id = {s1["id"]: truth for s1, truth in queries}

    print(f"  {country}: baseline candidate generation and ranking", flush=True)
    baseline = []
    queries = queries[:N_DIAGNOSTIC]  # One-shot diagnostic; do not expand before cause is known.
    query_records = [s1 for s1, _ in queries]
    baseline = []
    for i, (s1, truth) in enumerate(queries, 1):
        result = baseline_one(s1, idx, ids, measure=False)
        baseline.append((s1, truth, result))
        print(f"  {country}: baseline {i}/5", flush=True)

    print(f"  {country}: preparing query-specific posting cache", flush=True)
    optimized = NumpyQueryIndex(idx, ids, query_records)
    print(f"  {country}: posting cache ready in {optimized.cache_seconds:.2f}s", flush=True)
    optimized_results = []
    for i, (s1, truth, (base_features, base_pool, base_top, _, _)) in enumerate(baseline, 1):
        pool, touched = optimized.generate(s1)
        top, top_array = optimized.rank(pool)

        pool_parity = set(pool.tolist()) == base_pool
        if not pool_parity:
            raise RuntimeError(f"STOP: candidate pool mismatch for {s1['id']}: baseline={len(base_pool)}, optimized={pool.size}")
        if top != base_top:
            base_ids = [ids[p] for p in base_top]
            top_ids = [ids[p] for p in top]
            mismatch = next(i for i, (a, b) in enumerate(zip(base_top, top)) if a != b)
            def describe(pos, fs):
                values = tuple(int(v) for v in fs)
                parts = tuple(int(v) * int(w) for v, w in zip(values, WEIGHTS.tolist()))
                return {"id": ids[pos], "score": sum(parts), "components": values, "weighted_components": parts}
            lo, hi = max(0, mismatch - 1), min(CAP, mismatch + 2)
            print(f"\nPARITY FAILURE: S1={s1['id']} rank={mismatch+1}", flush=True)
            print(f"candidate pool parity={pool_parity}; baseline top200={len(base_ids)}; optimized top200={len(top_ids)}", flush=True)
            for rank in range(lo, hi):
                bp, op = base_top[rank], top[rank]
                baseline_desc = describe(bp, base_features[bp])
                opt_desc = describe(op, optimized.features[op])
                same_score = baseline_desc["score"] == opt_desc["score"]
                print(f"  position {rank+1}: baseline={baseline_desc}; optimized={opt_desc}; identical_scores={same_score}", flush=True)
            base_selected, opt_selected = set(base_top), set(top)
            print(f"top200 candidate-set parity={base_selected == opt_selected}", flush=True)
            raise RuntimeError("STOP: ranking parity differs; do not process further S1s until this cause is resolved")
        # Verify every blocker count and score for the selected positions.
        if any(tuple(optimized.features[p].tolist()) != tuple(base_features[p]) for p in top_array):
            raise RuntimeError(f"STOP: blocker feature mismatch within top 200 for {s1['id']}")
        base_recall = sum(ids[p] in truth for p in base_top)
        opt_recall = sum(ids[p] in truth for p in top)
        if base_recall != opt_recall:
            raise RuntimeError(f"STOP: true-match recall mismatch for {s1['id']}")
        optimized_results.append((pool.size, base_recall, len(truth)))
        optimized.clear(touched)
        print(f"  {country}: query {i}/5 candidate pool, ordered top-200, and recall parity passed", flush=True)

    print(f"  {country}: parity passed on all five; now timing the same five queries", flush=True)
    measured_baseline = []
    measured_optimized = []
    for s1, truth, (_, _, expected_top, _, _) in baseline:
        measured_baseline.append(baseline_one(s1, idx, ids, measure=True))
        t0 = time.perf_counter()
        pool, touched = optimized.generate(s1)
        gen_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        top, _ = optimized.rank(pool)
        rank_s = time.perf_counter() - t0
        if set(pool.tolist()) != measured_baseline[-1][1] or top != expected_top:
            raise RuntimeError(f"STOP: parity changed during timed pass for {s1['id']}")
        measured_optimized.append((gen_s, rank_s))
        optimized.clear(touched)
    base_gen = sum(x[3] for x in measured_baseline)
    base_rank = sum(x[4] for x in measured_baseline)
    opt_gen = sum(x[0] for x in measured_optimized)
    opt_rank = sum(x[1] for x in measured_optimized)
    base_total = base_gen + base_rank
    opt_query_total = opt_gen + opt_rank
    opt_total_including_cache = opt_query_total + optimized.cache_seconds
    ntrue = sum(x[2] for x in optimized_results)
    recovered = sum(x[1] for x in optimized_results)
    return {
        "country": country,
        "n_queries": len(queries),
        "n_candidates": len(ids),
        "index_seconds": index_seconds,
        "cache_seconds": optimized.cache_seconds,
        "baseline_gen": base_gen,
        "baseline_rank": base_rank,
        "baseline_total": base_total,
        "optimized_gen": opt_gen,
        "optimized_rank": opt_rank,
        "optimized_query_total": opt_query_total,
        "optimized_total_including_cache": opt_total_including_cache,
        "speedup_query": base_total / opt_query_total if opt_query_total else float("inf"),
        "speedup_including_cache": base_total / opt_total_including_cache if opt_total_including_cache else float("inf"),
        "pool_size_sum": sum(x[0] for x in optimized_results),
        "truth": ntrue,
        "recovered": recovered,
        "misses": ntrue - recovered,
    }


def diagnose_target():
    """Compare only the previously failing India S1; no other query is processed."""
    country = "india"
    gt = load_ground_truth(SPLITS / f"train_GT_{country}.tsv")
    target = None
    for eid, name, address, ctry in stream_records(SPLITS / f"train_S1_{country}.tsv"):
        if eid == TARGET_ID:
            target = normalize_record(eid, name, address, ctry)
            break
    if target is None:
        raise RuntimeError(f"Could not find {TARGET_ID} in the India S1 file")
    truth = gt.get(TARGET_ID, set())
    print(f"Target {TARGET_ID}; ground-truth matches={len(truth)}", flush=True)

    idx, ids, index_seconds = build_index(country)
    print("Building baseline signals and uncapped pool", flush=True)
    base_features, base_pool = score_candidate_features(target, idx)
    print("Building optimized signals and uncapped pool", flush=True)
    optimized = NumpyQueryIndex(idx, ids, [target])
    opt_pool, touched = optimized.generate(target)
    base_ids = {ids[p] for p in base_pool}
    opt_ids = {ids[int(p)] for p in opt_pool}
    pool_equal = base_ids == opt_ids
    missing_optimized = sorted(base_ids - opt_ids)
    missing_baseline = sorted(opt_ids - base_ids)
    duplicate_count = len(ids) - len(set(ids))
    id_type_ok = all(isinstance(eid, str) for eid in ids)
    id_whitespace = [eid for eid in ids if eid != eid.strip()]
    print("\nUNCAPPED POOL")
    print(f"  baseline size={len(base_pool):,}; optimized size={opt_pool.size:,}; set equality={pool_equal}")
    print(f"  missing from optimized ({len(missing_optimized)}): {missing_optimized}")
    print(f"  missing from baseline ({len(missing_baseline)}): {missing_baseline}")
    if not pool_equal:
        print("STOP: candidate pools differ; no ranking comparison will be treated as valid.", flush=True)
        return

    base_top = heapq.nsmallest(
        CAP, base_pool,
        key=lambda pos: (-address_rescue_score(base_features[pos]), ids[pos]),
    )
    opt_top, _ = optimized.rank(opt_pool)
    opt_top = [int(pos) for pos in opt_top]
    base_top_ids = [ids[p] for p in base_top]
    opt_top_ids = [ids[p] for p in opt_top]
    first_diff = next((i for i, (a, b) in enumerate(zip(base_top_ids, opt_top_ids)) if a != b), None)

    def vector_baseline(pos):
        return tuple(int(x) for x in base_features[pos])

    def vector_optimized(pos):
        return tuple(int(x) for x in optimized.features[pos].tolist())

    def detail(pos):
        eid = ids[pos]
        bvec = vector_baseline(pos)
        ovec = vector_optimized(pos)
        bscore = address_rescue_score(base_features[pos])
        oscore = int(np.asarray(ovec, dtype=np.int32) @ WEIGHTS)
        return {
            "id": eid,
            "baseline_score": bscore,
            "optimized_score": oscore,
            "baseline_components": dict(zip(SIGNAL_NAMES, bvec)),
            "optimized_components": dict(zip(SIGNAL_NAMES, ovec)),
            "all_components_identical": bvec == ovec,
            "scores_identical": bscore == oscore,
            "id_is_string": isinstance(eid, str),
            "id_repr": repr(eid),
        }

    print("\nORDERING")
    print(f"  baseline top-200={len(base_top_ids)}; optimized top-200={len(opt_top_ids)}")
    print(f"  ordered top-200 parity={base_top_ids == opt_top_ids}")
    print(f"  top-200 set parity={set(base_top_ids) == set(opt_top_ids)}")
    print(f"  first differing rank={None if first_diff is None else first_diff + 1}")

    if first_diff is not None:
        involved = set()
        for ranked in (base_top, opt_top):
            for p in range(max(0, first_diff - 1), min(len(ranked), first_diff + 2)):
                involved.add(int(ranked[p]))
        for pos in sorted(involved, key=lambda p: ids[p]):
            print(f"  candidate details: {detail(pos)}")
        for label, ranked in (("baseline", base_top), ("optimized", opt_top)):
            at = first_diff
            print(f"  {label} neighbors:")
            for p in range(max(0, at - 1), min(len(ranked), at + 2)):
                print(f"    rank {p+1}: {detail(int(ranked[p]))}")
    else:
        print("  No differing position exists in these two ordered top-200 lists.")

    # Exact signal equality across the full uncapped pool, independent of order.
    component_diffs = []
    score_diffs = []
    for pos in base_pool:
        pos = int(pos)
        bv, ov = vector_baseline(pos), vector_optimized(pos)
        if bv != ov:
            component_diffs.append(ids[pos])
        if address_rescue_score(base_features[pos]) != int(np.asarray(ov, dtype=np.int32) @ WEIGHTS):
            score_diffs.append(ids[pos])
    print("\nIMPLEMENTATION CHECKS")
    print(f"  full-pool component mismatches={len(component_diffs)}")
    print(f"  full-pool score mismatches={len(score_diffs)}")
    print("  score type: baseline integer; optimized integer dot product (int32); no floating-point scores")
    print(f"  duplicate candidate IDs={duplicate_count}; all candidate IDs strings={id_type_ok}; IDs with whitespace={len(id_whitespace)}")
    print("  set/dict iteration affects only integer accumulation order; no ordering key uses set/dict order")
    print("  both ranking keys are (-exact integer score, candidate ID string)")
    print("  Address Rescue weights/features: (8,7,2,8,6,6,1) over the same seven blocker counts")

    baseline_recall = sum(eid in truth for eid in base_top_ids)
    optimized_recall = sum(eid in truth for eid in opt_top_ids)
    print("\nPARITY SUMMARY")
    print(f"  uncapped pool parity={'PASS' if pool_equal else 'FAIL'}")
    print(f"  ordered ranking parity={'PASS' if base_top_ids == opt_top_ids else 'FAIL'}")
    print(f"  top-200 parity={'PASS' if set(base_top_ids) == set(opt_top_ids) else 'FAIL'}")
    print(f"  true-match recall parity={'PASS' if baseline_recall == optimized_recall else 'FAIL'} ({baseline_recall}/{len(truth)} vs {optimized_recall}/{len(truth)})")
    print(f"  index build time={index_seconds:.2f}s (not part of ranking parity) ")
    optimized.clear(touched)


def main():
    diagnose_target()
    return
    wall = time.perf_counter()
    reports = [run_country("india")]
    print(f"\n=== OPTIMIZATION BENCHMARK ({sum(r['n_queries'] for r in reports)} matched S1s) ===")
    for r in reports:
        n = r["n_queries"]
        print(f"\n{r['country'].upper()} ({n} S1, {r['n_candidates']:,} indexed candidates)")
        print(f"  Shared index build (excluded from both timings): {r['index_seconds']:.2f}s")
        print(f"  Baseline candidate generation:                 {r['baseline_gen']:.2f}s")
        print(f"  Optimized candidate generation:                {r['optimized_gen']:.2f}s")
        print(f"  Baseline Address Rescue ranking:               {r['baseline_rank']:.2f}s")
        print(f"  Optimized top-200 ranking:                     {r['optimized_rank']:.2f}s")
        print(f"  Baseline total:                                {r['baseline_total']:.2f}s")
        print(f"  Baseline total per S1:                        {r['baseline_total']/n:.3f}s")
        print(f"  Optimized query total:                         {r['optimized_query_total']:.2f}s")
        print(f"  Optimized query total per S1:                 {r['optimized_query_total']/n:.3f}s")
        print(f"  Optimized posting-cache setup:                {r['cache_seconds']:.2f}s")
        print(f"  Optimized total incl. cache setup:            {r['optimized_total_including_cache']:.2f}s")
        print(f"  Speedup (query-only / incl. cache):           {r['speedup_query']:.2f}x / {r['speedup_including_cache']:.2f}x")
        print(f"  Candidate pool parity:                         PASS")
        print(f"  Ordered top-200 parity:                        PASS")
        print(f"  True-match recall parity:                      PASS ({r['recovered']}/{r['truth']}, {r['misses']} missed)")
        print(f"  Mean uncapped pool size:                       {r['pool_size_sum']/n:.1f}")

    totals = {
        k: sum(r[k] for r in reports)
        for k in ("baseline_gen", "baseline_rank", "baseline_total", "optimized_gen", "optimized_rank", "optimized_query_total", "cache_seconds", "optimized_total_including_cache", "truth", "recovered", "misses")
    }
    print("\nCOMBINED")
    print(f"  Baseline candidate generation: {totals['baseline_gen']:.2f}s")
    print(f"  Optimized candidate generation: {totals['optimized_gen']:.2f}s")
    print(f"  Baseline ranking: {totals['baseline_rank']:.2f}s")
    print(f"  Optimized ranking: {totals['optimized_rank']:.2f}s")
    print(f"  Baseline total: {totals['baseline_total']:.2f}s")
    print(f"  Optimized query total: {totals['optimized_query_total']:.2f}s")
    print(f"  Cache setup overhead: {totals['cache_seconds']:.2f}s")
    print(f"  Optimized total incl. cache: {totals['optimized_total_including_cache']:.2f}s")
    print(f"  Speedup: {totals['baseline_total']/totals['optimized_query_total']:.2f}x query-only; {totals['baseline_total']/totals['optimized_total_including_cache']:.2f}x incl. setup")
    print(f"  Recall: {totals['recovered']}/{totals['truth']} ({totals['recovered']/totals['truth']:.2%}); misses={totals['misses']}")
    print(f"\nWall time including shared index construction: {time.perf_counter()-wall:.1f}s")


if __name__ == "__main__":
    main()
