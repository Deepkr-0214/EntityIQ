#!/usr/bin/env python3
"""CAP sweep diagnostic: test candidate cap values 200, 300, 500.

Uses the IDENTICAL 400 sampled S1s, seeds, and 60/20/20 split as
experiment_matching_model_v2.py.  Only validation S1s are analysed.

For each S1 the full Address Rescue ranked list is produced once from the
uncapped pool.  CAP=300 and CAP=500 results are obtained by slicing that
same list, which proves they are strict prefixes of the CAP=200 ranking.

Ground truth is consulted only after all ranking is complete.
No model is trained.  No submission files are written.  Production unchanged.
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

from experiment_matching_model import CAP as BASELINE_CAP, stratified_sample_and_split  # noqa: E402
from optimize_matching_experiment import NumpyQueryIndex, SPLITS, WEIGHTS  # noqa: E402
from src.pipeline.run_pipeline import CompactBlockingIndex, load_ground_truth, stream_records  # noqa: E402
from src.preprocessing.normalization import normalize_record  # noqa: E402

COUNTRIES = ("india", "us")
REQUESTED_SAMPLE = 200
SEED = 42
PROGRESS_EVERY = 50
TEST_CAPS = (200, 300, 500)


# ---------------------------------------------------------------------------
# Helpers — identical to experiment_matching_model_v2 / experiment_blocking_failures
# ---------------------------------------------------------------------------

def choose_sample(gt, country):
    return stratified_sample_and_split(
        gt, country, REQUESTED_SAMPLE,
        SEED + (0 if country == "india" else 1),
    )


def build_country_index(country: str):
    idx = CompactBlockingIndex()
    ids: list[str] = []
    seen: set[str] = set()
    t0 = time.perf_counter()
    for source in ("S2", "S3"):
        path = SPLITS / f"train_{source}_{country}.tsv"
        for eid, name, address, ctry in stream_records(path):
            if eid in seen:
                continue
            seen.add(eid)
            idx.add(len(ids), normalize_record(eid, name, address, ctry))
            ids.append(eid)
            if len(ids) % 500_000 == 0:
                print(f"  {country}: indexed {len(ids):,} ({time.perf_counter()-t0:.1f}s)", flush=True)
    return idx, ids, time.perf_counter() - t0


def load_sampled_s1(country: str, wanted: set[str]) -> dict:
    records = {}
    for eid, name, address, ctry in stream_records(SPLITS / f"train_S1_{country}.tsv"):
        if eid in wanted:
            records[eid] = normalize_record(eid, name, address, ctry)
    return records


def address_rescue_top_k(pool: np.ndarray, local_feat: np.ndarray,
                          ids: list[str], k: int) -> list[int]:
    """Return top-k global positions from pool using Address Rescue ranking.

    local_feat[i] corresponds to pool[i].  Tie-break: ascending candidate ID.
    Identical scoring to NumpyQueryIndex.rank() but parameterised on k.
    """
    n = len(pool)
    if n == 0:
        return []
    scores = local_feat.astype(np.int32) @ WEIGHTS
    take = min(k, n)
    if n <= k:
        chosen_local = np.arange(n, dtype=np.int32)
    else:
        threshold = np.partition(scores, n - take)[n - take]
        above = np.flatnonzero(scores > threshold)
        ties = np.flatnonzero(scores == threshold)
        remaining = take - above.size
        if remaining > 0:
            tie_global = pool[ties]
            tie_order = sorted(range(ties.size), key=lambda i: ids[int(tie_global[i])])
            chosen_local = np.concatenate(
                (above, ties[np.asarray(tie_order[:remaining], dtype=np.int32)])
            )
        else:
            chosen_local = above
    sel_scores = scores[chosen_local]
    sel_global = pool[chosen_local]
    order = sorted(
        range(len(chosen_local)),
        key=lambda i: (-int(sel_scores[i]), ids[int(sel_global[i])]),
    )
    return [int(sel_global[i]) for i in order]


def full_ranked_list(pool: np.ndarray, local_feat: np.ndarray,
                     ids: list[str]) -> list[int]:
    """Return ALL pool positions sorted by Address Rescue score descending,
    tie-broken by candidate ID ascending.  Slicing this list at k gives the
    identical result as address_rescue_top_k(..., k) — strict prefix property.
    """
    n = len(pool)
    if n == 0:
        return []
    scores = local_feat.astype(np.int32) @ WEIGHTS
    order = sorted(range(n), key=lambda i: (-int(scores[i]), ids[int(pool[i])]))
    return [int(pool[i]) for i in order]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    wall_start = time.perf_counter()

    assert BASELINE_CAP == 200, f"Expected baseline CAP=200, got {BASELINE_CAP}"
    assert TEST_CAPS[0] == BASELINE_CAP, "First test CAP must equal baseline"

    # ---- reproduce identical splits ----------------------------------------
    all_gt: dict[str, set] = {}
    val_ids: set[str] = set()
    val_s1_records: dict[str, dict] = {}
    country_of: dict[str, str] = {}

    for country in COUNTRIES:
        gt = load_ground_truth(SPLITS / f"train_GT_{country}.tsv")
        splits, _, quotas = choose_sample(gt, country)
        print(f"\n{country.upper()}: split counts={{{', '.join(f'{k}:{len(v)}' for k,v in splits.items())}}}", flush=True)
        wanted = set().union(*splits.values())
        for sid in wanted:
            all_gt[sid] = set(gt.get(sid, set()))
        val_ids.update(splits["validation"])
        s1_recs = load_sampled_s1(country, wanted)
        val_s1_records.update({sid: s1_recs[sid] for sid in splits["validation"]})
        for sid in splits["validation"]:
            country_of[sid] = country

    val_id_list = sorted(val_ids)
    val_true_total = sum(len(all_gt[s]) for s in val_id_list)
    print(f"\nValidation S1s: {len(val_id_list)}  |  Total true matches: {val_true_total}", flush=True)

    # ---- generate uncapped pools + full ranked lists -----------------------
    # ranked_list[sid] = full Address Rescue ordering of all pool positions
    # pool_size[sid]   = uncapped pool size
    ranked_list: dict[str, list[int]] = {}
    pool_size: dict[str, int] = {}
    cids_of: dict[str, list[str]] = {}   # sid -> candidate ID list for that country

    gen_seconds = 0.0
    rank_seconds = 0.0

    for country in COUNTRIES:
        val_sids_country = [sid for sid in val_id_list if country_of[sid] == country]
        print(f"\n{country.upper()}: building index", flush=True)
        idx, cids, index_s = build_country_index(country)
        print(f"  index ready: {len(cids):,} candidates in {index_s:.1f}s", flush=True)

        s1_recs_country = {sid: val_s1_records[sid] for sid in val_sids_country}
        t_cache = time.perf_counter()
        query_index = NumpyQueryIndex(idx, cids, list(s1_recs_country.values()))
        print(f"  posting cache ready in {query_index.cache_seconds:.2f}s", flush=True)

        for i, sid in enumerate(val_sids_country, 1):
            t0 = time.perf_counter()
            pool, touched = query_index.generate(val_s1_records[sid])
            gen_seconds += time.perf_counter() - t0

            feat_copy = query_index.features[pool].copy()  # (pool_size, 7)

            t0 = time.perf_counter()
            ranked = full_ranked_list(pool, feat_copy, cids)
            rank_seconds += time.perf_counter() - t0

            ranked_list[sid] = ranked
            pool_size[sid] = len(pool)
            cids_of[sid] = cids  # shared reference per country — safe (read-only after this)

            query_index.clear(touched)
            if i % PROGRESS_EVERY == 0 or i == len(val_sids_country):
                print(f"  {country}: ranked {i}/{len(val_sids_country)}", flush=True)

        del idx, query_index
        gc.collect()

    # ---- strict prefix verification ----------------------------------------
    # For every S1, verify that top-k for smaller CAP is a prefix of top-k for
    # larger CAP within the full ranked list.  This is true by construction
    # (slicing), but we assert it explicitly.
    print("\nVerifying strict prefix property ...", flush=True)
    prefix_violations = 0
    for sid in val_id_list:
        ranked = ranked_list[sid]
        for smaller, larger in ((200, 300), (200, 500), (300, 500)):
            top_small = ranked[:smaller]
            top_large = ranked[:larger]
            if top_large[:len(top_small)] != top_small:
                prefix_violations += 1
                print(f"  PREFIX VIOLATION: {sid} CAP={smaller} not prefix of CAP={larger}", flush=True)
    if prefix_violations == 0:
        print("  Strict prefix check PASSED for all S1s across all CAP pairs.", flush=True)
    else:
        raise RuntimeError(f"Strict prefix check FAILED: {prefix_violations} violations")

    # ---- measure recall at each CAP (GT consulted here for the first time) --
    print("\n" + "=" * 70, flush=True)
    print("CAP SWEEP RESULTS", flush=True)
    print("=" * 70, flush=True)

    # Baseline CAP=200 top-k sets — used to compute newly recovered / lost counts
    baseline_top_ids: dict[str, set[str]] = {}
    for sid in val_id_list:
        cids = cids_of[sid]
        baseline_top_ids[sid] = {cids[p] for p in ranked_list[sid][:200]}

    cap_stats: dict[int, dict] = {}
    for cap in TEST_CAPS:
        recovered = 0
        newly_recovered = 0   # in top-cap but not in top-200 (only meaningful for cap>200)
        lost = 0              # in top-200 but not in top-cap (should be 0 by prefix property)
        total_candidates = 0
        max_candidates = 0
        for sid in val_id_list:
            cids = cids_of[sid]
            top_ids = {cids[p] for p in ranked_list[sid][:cap]}
            truth = all_gt[sid]
            recovered += len(truth & top_ids)
            total_candidates += len(top_ids)
            max_candidates = max(max_candidates, len(top_ids))
            if cap > 200:
                newly_recovered += len(truth & (top_ids - baseline_top_ids[sid]))
                lost += len(truth & (baseline_top_ids[sid] - top_ids))
        cap_stats[cap] = {
            "recovered": recovered,
            "newly_recovered": newly_recovered,
            "lost": lost,
            "total_candidates": total_candidates,
            "max_candidates": max_candidates,
            "avg_candidates": total_candidates / len(val_id_list),
            "blocking_failures": val_true_total - recovered,
            "blocking_recall": recovered / val_true_total if val_true_total else 0.0,
        }

    # ---- report ------------------------------------------------------------
    print(f"\n{'CAP':<6} {'Recovered':>9} {'BlkRecall':>10} {'Failures':>9} "
          f"{'NewRecov':>9} {'Lost':>5} {'AvgCands':>9} {'MaxCands':>9}")
    print("-" * 72)
    for cap in TEST_CAPS:
        s = cap_stats[cap]
        nr = s["newly_recovered"] if cap > 200 else "-"
        lo = s["lost"] if cap > 200 else "-"
        print(f"  {cap:<4} {s['recovered']:>9}  {s['blocking_recall']:>9.2%}  "
              f"{s['blocking_failures']:>9}  {str(nr):>9}  {str(lo):>5}  "
              f"{s['avg_candidates']:>9.1f}  {s['max_candidates']:>9}")

    print(f"\nColumn definitions:")
    print(f"  Recovered   = validation true matches in top-CAP")
    print(f"  BlkRecall   = Recovered / {val_true_total}")
    print(f"  Failures    = {val_true_total} - Recovered  (blocking failures at this CAP)")
    print(f"  NewRecov    = true matches gained vs CAP=200  (strict prefix: Lost is always 0)")
    print(f"  Lost        = true matches in CAP=200 but not in this CAP  (must be 0)")
    print(f"  AvgCands    = mean candidate count per S1 (capped at pool size)")
    print(f"  MaxCands    = maximum candidate count across all validation S1s")

    # ---- per-CAP detail for newly recovered pairs --------------------------
    for cap in TEST_CAPS:
        if cap == 200:
            continue
        print(f"\nNewly recovered pairs at CAP={cap} (not in CAP=200 top-list):")
        found_any = False
        for sid in val_id_list:
            cids = cids_of[sid]
            top_ids = {cids[p] for p in ranked_list[sid][:cap]}
            truth = all_gt[sid]
            new_hits = truth & (top_ids - baseline_top_ids[sid])
            for mid in sorted(new_hits):
                # Find rank of this match in the full ranked list
                rank = next(
                    (r + 1 for r, p in enumerate(ranked_list[sid]) if cids[p] == mid),
                    None,
                )
                # Feature vector of this match
                pool_local_idx = next(
                    (i for i, p in enumerate(ranked_list[sid]) if cids[p] == mid),
                    None,
                )
                print(f"  S1={sid}  match={mid}  ar_rank={rank}", flush=True)
                found_any = True
        if not found_any:
            print(f"  (none)", flush=True)

    # ---- pool size distribution for context --------------------------------
    print(f"\nUncapped pool size distribution (validation S1s):")
    sizes = sorted(pool_size[sid] for sid in val_id_list)
    n = len(sizes)
    print(f"  min={sizes[0]:,}  p25={sizes[n//4]:,}  median={sizes[n//2]:,}  "
          f"p75={sizes[3*n//4]:,}  p90={sizes[int(0.9*n)]:,}  max={sizes[-1]:,}")
    large_pools = sum(1 for s in sizes if s > 500)
    print(f"  S1s with pool > 500: {large_pools}/{n}  "
          f"(CAP=500 is binding for {large_pools} S1s)")

    # ---- timing summary ----------------------------------------------------
    total_seconds = time.perf_counter() - wall_start
    print(f"\nTiming:")
    print(f"  Candidate generation (NumpyQueryIndex.generate): {gen_seconds:.2f}s")
    print(f"  Full ranked-list construction (all S1s):         {rank_seconds:.2f}s")
    print(f"  Total wall time:                                  {total_seconds:.1f}s")
    print(f"  Note: index construction dominates; ranking is negligible.")
    print(f"  Feature extraction time: N/A — no model trained in this script.")

    # ---- recommendation ----------------------------------------------------
    print(f"\n=== RECOMMENDATION ===")
    r200 = cap_stats[200]["recovered"]
    r300 = cap_stats[300]["recovered"]
    r500 = cap_stats[500]["recovered"]
    gain_300 = r300 - r200
    gain_500 = r500 - r200
    avg_200 = cap_stats[200]["avg_candidates"]
    avg_300 = cap_stats[300]["avg_candidates"]
    avg_500 = cap_stats[500]["avg_candidates"]
    print(f"  CAP=200 baseline: {r200}/{val_true_total} recovered ({cap_stats[200]['blocking_recall']:.2%})")
    print(f"  CAP=300: +{gain_300} true matches, avg candidates {avg_200:.1f} -> {avg_300:.1f} "
          f"(+{avg_300-avg_200:.1f} per S1)")
    print(f"  CAP=500: +{gain_500} true matches, avg candidates {avg_200:.1f} -> {avg_500:.1f} "
          f"(+{avg_500-avg_200:.1f} per S1)")
    print(f"  Strict prefix property: VERIFIED — CAP=300/500 are exact extensions of CAP=200.")
    print(f"  No ranking weights were changed.  No GT was used during ranking.")
    print(f"\nScope: training data only; no model training; no submission files; production unchanged.")


if __name__ == "__main__":
    main()
