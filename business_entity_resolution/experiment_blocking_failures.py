#!/usr/bin/env python3
"""Blocking failure diagnostic for the validation split.

Uses the IDENTICAL 400 sampled S1s and 60/20/20 split as experiment_matching_model_v2.py
(same stratified_sample_and_split call, same seeds).  Only validation S1s are analysed.

For each validation S1 the uncapped candidate pool is reproduced.  True matches are
classified as:
  (A) in uncapped pool but outside top-200  — cap-dropped
  (B) completely absent from uncapped pool  — never generated

Five deterministic ranking strategies are then compared on the validation set:
  1. address_rescue   — existing WEIGHTS=[8,7,2,8,6,6,1] dot product
  2. consensus        — sum of all seven binary/count signals (equal weight)
  3. name_priority    — first_token*8 + name_prefix*7 + shared_name_tokens*4
                        + consonant_skeleton*6 + shared_consonant_ngrams*2
  4. address_priority — shared_address_numbers*8 + shared_address_tokens*6
                        + name_prefix*3 + first_token*2
  5. combined         — address_rescue + 0.5*(name_token_jaccard_proxy)
                        where proxy = shared_name_tokens / max(1, max_shared_name)
                        (uses only blocker counts, no pairwise features, no GT)

All strategies use the same uncapped pool; only the top-200 selection changes.
GT is consulted only after candidate generation to measure recall.

No models are trained.  No submission files are written.  Production is not modified.
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
    CAP, stratified_sample_and_split,
)
from optimize_matching_experiment import NumpyQueryIndex, SPLITS, WEIGHTS  # noqa: E402
from src.pipeline.run_pipeline import CompactBlockingIndex, load_ground_truth, stream_records  # noqa: E402
from src.preprocessing.normalization import normalize_record  # noqa: E402

COUNTRIES = ("india", "us")
REQUESTED_SAMPLE = 200
SEED = 42
PROGRESS_EVERY = 50

# ---------------------------------------------------------------------------
# Ranking strategy definitions
# Each strategy is a function (features_row: np.ndarray[7]) -> int score.
# Higher score = higher priority.  Tie-break is always candidate ID (ascending).
# features_row columns: [first_token, name_prefix, shared_name_tokens,
#                         shared_address_numbers, shared_address_tokens,
#                         consonant_skeleton, shared_consonant_ngrams]
# ---------------------------------------------------------------------------

def _score_address_rescue(f: np.ndarray) -> int:
    """Existing production weights [8,7,2,8,6,6,1]."""
    return int(f @ WEIGHTS)


def _score_consensus(f: np.ndarray) -> int:
    """Sum of all seven signals with equal weight 1."""
    return int(f.sum())


def _score_name_priority(f: np.ndarray) -> int:
    """Emphasise name signals; de-emphasise address."""
    # first_token*8 + name_prefix*7 + shared_name_tokens*4
    # + consonant_skeleton*6 + shared_consonant_ngrams*2
    w = np.array([8, 7, 4, 0, 0, 6, 2], dtype=np.int32)
    return int(f @ w)


def _score_address_priority(f: np.ndarray) -> int:
    """Emphasise address signals; keep name as secondary."""
    # shared_address_numbers*8 + shared_address_tokens*6
    # + name_prefix*3 + first_token*2
    w = np.array([2, 3, 0, 8, 6, 0, 0], dtype=np.int32)
    return int(f @ w)


def _score_combined(f: np.ndarray, max_shared_name: int) -> int:
    """address_rescue + scaled shared_name_tokens bonus.

    The bonus is a proxy for name-token overlap density using only blocker
    counts.  It does not use pairwise features or ground truth.
    Scaled to [0,4] to avoid dominating the address_rescue score.
    """
    base = int(f @ WEIGHTS)
    shared_name = int(f[2])
    bonus = int(4 * shared_name / max_shared_name) if max_shared_name > 0 else 0
    return base + bonus


STRATEGY_NAMES = [
    "address_rescue",
    "consensus",
    "name_priority",
    "address_priority",
    "combined",
]


def top_k_by_strategy(pool: np.ndarray, local_features: np.ndarray, ids: list[str],
                       max_shared_name: int, k: int = CAP) -> dict[str, list[int]]:
    """Return top-k position lists (indices into `pool`) for each strategy.

    `local_features` has shape (len(pool), 7) — row i corresponds to pool[i].
    Returns dict mapping strategy name -> list of global positions (values from pool).
    """
    n = len(pool)
    local_idx = np.arange(n, dtype=np.int32)  # 0..n-1
    feat = local_features.astype(np.int32)     # (n, 7)

    results: dict[str, list[int]] = {}
    for name in STRATEGY_NAMES:
        if name == "address_rescue":
            scores = feat @ WEIGHTS
        elif name == "consensus":
            scores = feat.sum(axis=1)
        elif name == "name_priority":
            w = np.array([8, 7, 4, 0, 0, 6, 2], dtype=np.int32)
            scores = feat @ w
        elif name == "address_priority":
            w = np.array([2, 3, 0, 8, 6, 0, 0], dtype=np.int32)
            scores = feat @ w
        else:  # combined
            base = feat @ WEIGHTS
            shared_name = feat[:, 2]
            bonus = (4 * shared_name / max_shared_name).astype(np.int32) if max_shared_name > 0 else np.zeros(n, dtype=np.int32)
            scores = base + bonus

        take = min(k, n)
        if n <= k:
            chosen_local = local_idx
        else:
            threshold = np.partition(scores, n - take)[n - take]
            above = np.flatnonzero(scores > threshold)
            ties = np.flatnonzero(scores == threshold)
            remaining = take - above.size
            if remaining > 0:
                tie_global = pool[ties]
                tie_order = sorted(range(ties.size), key=lambda i: ids[int(tie_global[i])])
                chosen_local = np.concatenate((above, ties[np.asarray(tie_order[:remaining], dtype=np.int32)]))
            else:
                chosen_local = above

        sel_scores = scores[chosen_local]
        sel_global = pool[chosen_local]
        order = sorted(range(len(chosen_local)),
                       key=lambda i: (-int(sel_scores[i]), ids[int(sel_global[i])]))
        results[name] = [int(sel_global[i]) for i in order]
    return results


# ---------------------------------------------------------------------------
# Index / candidate generation helpers (mirrors v2 exactly)
# ---------------------------------------------------------------------------

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
    elapsed = time.perf_counter() - t0
    return idx, ids, elapsed


def load_sampled_s1(country: str, wanted: set[str]) -> dict:
    records = {}
    for eid, name, address, ctry in stream_records(SPLITS / f"train_S1_{country}.tsv"):
        if eid in wanted:
            records[eid] = normalize_record(eid, name, address, ctry)
    return records


def choose_sample(gt, country):
    try:
        return stratified_sample_and_split(
            gt, country, REQUESTED_SAMPLE,
            SEED + (0 if country == "india" else 1),
        )
    except ValueError as exc:
        raise RuntimeError(f"{country}: cannot reproduce 200-S1 sample: {exc}") from exc


# ---------------------------------------------------------------------------
# Main diagnostic
# ---------------------------------------------------------------------------

def main():
    wall_start = time.perf_counter()

    # ---- reproduce splits (identical to v2) --------------------------------
    all_gt: dict[str, set] = {}
    val_ids: set[str] = set()
    val_s1_records: dict[str, dict] = {}

    for country in COUNTRIES:
        print(f"\n=== {country.upper()} ===", flush=True)
        gt = load_ground_truth(SPLITS / f"train_GT_{country}.tsv")
        splits, cat_map, quotas = choose_sample(gt, country)
        print(f"  split counts: { {k: len(v) for k, v in splits.items()} }", flush=True)
        wanted = set().union(*splits.values())
        for sid in wanted:
            all_gt[sid] = set(gt.get(sid, set()))
        val_ids.update(splits["validation"])
        s1_recs = load_sampled_s1(country, wanted)
        val_s1_records.update({sid: s1_recs[sid] for sid in splits["validation"]})

    print(f"\nValidation S1s: {len(val_ids)}", flush=True)
    val_id_list = sorted(val_ids)

    # ---- per-country: generate uncapped pools for validation S1s only ------
    # We still need to build the full index per country, but only query val S1s.
    country_of: dict[str, str] = {}
    for country in COUNTRIES:
        gt = load_ground_truth(SPLITS / f"train_GT_{country}.tsv")
        splits, _, _ = choose_sample(gt, country)
        for sid in splits["validation"]:
            country_of[sid] = country

    # Collect per-S1 uncapped pool data
    # pool_data[sid] = {"pool": np.ndarray positions, "features": NumpyQueryIndex.features view copy}
    pool_data: dict[str, dict] = {}
    candidate_ids_by_country: dict[str, list[str]] = {}

    for country in COUNTRIES:
        val_sids_country = [sid for sid in val_id_list if country_of[sid] == country]
        print(f"\n{country.upper()}: building index for {len(val_sids_country)} validation S1s", flush=True)
        idx, cids, index_s = build_country_index(country)
        candidate_ids_by_country[country] = cids
        print(f"  index ready: {len(cids):,} candidates in {index_s:.1f}s", flush=True)

        s1_recs_country = {sid: val_s1_records[sid] for sid in val_sids_country}
        query_index = NumpyQueryIndex(idx, cids, list(s1_recs_country.values()))
        print(f"  posting cache ready in {query_index.cache_seconds:.2f}s", flush=True)

        for i, sid in enumerate(val_sids_country, 1):
            pool, touched = query_index.generate(val_s1_records[sid])
            # Copy feature rows for this pool before clearing
            feat_copy = query_index.features[pool].copy()  # shape (pool_size, 7)
            pool_data[sid] = {
                "pool": pool.copy(),
                "features": feat_copy,
                "cids": cids,
            }
            query_index.clear(touched)
            if i % PROGRESS_EVERY == 0 or i == len(val_sids_country):
                print(f"  {country}: generated uncapped pools {i}/{len(val_sids_country)}", flush=True)

        del idx, query_index
        gc.collect()

    # ---- classify blocking failures ----------------------------------------
    # Category A: in uncapped pool, outside top-200 (address_rescue ranking)
    # Category B: completely absent from uncapped pool

    cat_a: list[tuple[str, str]] = []  # (s1_id, match_id)
    cat_b: list[tuple[str, str]] = []

    for sid in val_id_list:
        truth = all_gt[sid]
        if not truth:
            continue
        pd = pool_data[sid]
        pool = pd["pool"]
        cids = pd["cids"]
        pool_id_set = {cids[int(p)] for p in pool}

        # address_rescue top-200
        feat = pd["features"]
        scores_ar = feat.astype(np.int32) @ WEIGHTS
        take = min(CAP, len(pool))
        if len(pool) <= CAP:
            top200_positions = pool
        else:
            threshold = np.partition(scores_ar, len(pool) - take)[len(pool) - take]
            above = np.flatnonzero(scores_ar > threshold)
            ties = np.flatnonzero(scores_ar == threshold)
            remaining = take - above.size
            if remaining > 0:
                tie_pos = pool[ties]
                tie_order = sorted(range(ties.size), key=lambda i: cids[int(tie_pos[i])])
                chosen = np.concatenate((above, ties[np.asarray(tie_order[:remaining], dtype=np.int32)]))
            else:
                chosen = above
            top200_positions = pool[chosen]
        top200_id_set = {cids[int(p)] for p in top200_positions}

        for mid in truth:
            if mid in top200_id_set:
                pass  # recovered by existing strategy
            elif mid in pool_id_set:
                cat_a.append((sid, mid))  # cap-dropped
            else:
                cat_b.append((sid, mid))  # never generated

    print(f"\n=== BLOCKING FAILURE CLASSIFICATION ===")
    print(f"Validation true matches total: {sum(len(all_gt[s]) for s in val_id_list)}")
    print(f"Category A (in uncapped pool, outside top-200): {len(cat_a)}")
    print(f"Category B (completely absent from uncapped pool): {len(cat_b)}")
    print(f"Total blocking failures: {len(cat_a) + len(cat_b)}")

    if cat_b:
        print(f"\nCategory B details (never generated — irreducible without changing blocker):")
        for sid, mid in cat_b:
            print(f"  S1={sid}  missed_match={mid}")

    if cat_a:
        print(f"\nCategory A details (cap-dropped — ranking strategies can help):")
        for sid, mid in cat_a:
            pd = pool_data[sid]
            pool = pd["pool"]
            cids = pd["cids"]
            feat = pd["features"]
            # Find position of this match in the pool
            match_positions = [i for i, p in enumerate(pool) if cids[int(p)] == mid]
            if match_positions:
                local_idx = match_positions[0]
                f_row = feat[local_idx]
                ar_score = int(f_row.astype(np.int32) @ WEIGHTS)
                # Rank of this candidate under address_rescue
                scores_ar = feat.astype(np.int32) @ WEIGHTS
                rank_ar = int((scores_ar > ar_score).sum()) + 1
                print(f"  S1={sid}  match={mid}  pool_size={len(pool)}  "
                      f"ar_score={ar_score}  ar_rank={rank_ar}  "
                      f"features={tuple(int(x) for x in f_row)}")

    # ---- strategy comparison -----------------------------------------------
    print(f"\n=== RANKING STRATEGY COMPARISON (validation S1s only) ===")
    print(f"Baseline: address_rescue WEIGHTS={WEIGHTS.tolist()}")
    print(f"All strategies use the same uncapped pool; only top-200 selection differs.")
    print(f"GT consulted only after candidate generation.\n")

    # Compute max_shared_name across all validation pools (for combined strategy normalisation)
    max_shared_name_global = max(
        (int(pool_data[sid]["features"][:, 2].max()) if len(pool_data[sid]["pool"]) > 0 else 0)
        for sid in val_id_list
    ) if val_id_list else 1
    max_shared_name_global = max(max_shared_name_global, 1)

    # Per-strategy accumulators
    stats: dict[str, dict] = {
        name: {
            "recovered": 0,          # true matches in top-200
            "top200_total": 0,        # sum of top-200 sizes
            "cat_a_recovered": 0,     # cat-A pairs now in top-200
            "cat_a_lost": 0,          # pairs that were in baseline top-200 but dropped
        }
        for name in STRATEGY_NAMES
    }

    # Baseline top-200 per sid (address_rescue) — precompute for lost-pair accounting
    baseline_top200: dict[str, set[str]] = {}
    for sid in val_id_list:
        pd = pool_data[sid]
        pool = pd["pool"]
        cids = pd["cids"]
        feat = pd["features"]
        if len(pool) == 0:
            baseline_top200[sid] = set()
            continue
        tops = top_k_by_strategy(pool, feat, cids, max_shared_name_global)
        baseline_top200[sid] = {cids[p] for p in tops["address_rescue"]}

    for sid in val_id_list:
        truth = all_gt[sid]
        pd = pool_data[sid]
        pool = pd["pool"]
        cids = pd["cids"]
        feat = pd["features"]

        if len(pool) == 0:
            for name in STRATEGY_NAMES:
                stats[name]["top200_total"] += 0
            continue

        tops = top_k_by_strategy(pool, feat, cids, max_shared_name_global)

        for name in STRATEGY_NAMES:
            top_id_set = {cids[p] for p in tops[name]}
            stats[name]["recovered"] += len(truth & top_id_set)
            stats[name]["top200_total"] += len(top_id_set)
            # cat-A pairs recovered by this strategy
            for s1id, mid in cat_a:
                if s1id == sid and mid in top_id_set:
                    stats[name]["cat_a_recovered"] += 1
            # pairs that baseline had but this strategy lost
            if name != "address_rescue":
                lost = baseline_top200[sid] - top_id_set
                stats[name]["cat_a_lost"] += len(truth & lost)

    val_true_total = sum(len(all_gt[s]) for s in val_id_list)
    original_blocking_failures = len(cat_a) + len(cat_b)

    print(f"{'Strategy':<20} {'Recovered':>9} {'BlkRecall':>10} {'CatA_Rcvd':>10} "
          f"{'NewLost':>8} {'AvgTop200':>10} {'Deterministic':>14}")
    print("-" * 85)
    for name in STRATEGY_NAMES:
        s = stats[name]
        recovered = s["recovered"]
        br = recovered / val_true_total if val_true_total else 0.0
        cat_a_rcvd = s["cat_a_recovered"]
        new_lost = s["cat_a_lost"] if name != "address_rescue" else 0
        avg_top200 = s["top200_total"] / len(val_id_list) if val_id_list else 0.0
        print(f"  {name:<18} {recovered:>9}  {br:>9.2%}  {cat_a_rcvd:>10}  "
              f"{new_lost:>8}  {avg_top200:>9.1f}  {'YES':>14}")

    print(f"\nNotes:")
    print(f"  Recovered       = validation true matches in top-200")
    print(f"  BlkRecall       = Recovered / {val_true_total} total validation true matches")
    print(f"  CatA_Rcvd       = category-A pairs (cap-dropped) now in top-200")
    print(f"  NewLost         = true matches that baseline had but this strategy dropped")
    print(f"  AvgTop200       = mean top-200 size (always <= {CAP})")
    print(f"  Deterministic   = ranking uses no randomness (tie-break = candidate ID)")
    print(f"  Category B ({len(cat_b)} pairs) are irreducible by any ranking strategy.")

    # ---- per-strategy breakdown for cat-A pairs ----------------------------
    print(f"\n=== CATEGORY A PAIR RECOVERY BY STRATEGY ===")
    print(f"(Which of the {len(cat_a)} cap-dropped pairs does each strategy recover?)\n")
    for name in STRATEGY_NAMES:
        recovered_pairs = []
        for sid, mid in cat_a:
            pd = pool_data[sid]
            pool = pd["pool"]
            cids = pd["cids"]
            feat = pd["features"]
            tops = top_k_by_strategy(pool, feat, cids, max_shared_name_global)
            top_id_set = {cids[p] for p in tops[name]}
            if mid in top_id_set:
                recovered_pairs.append((sid, mid))
        print(f"  {name}: {len(recovered_pairs)}/{len(cat_a)} cat-A pairs recovered")
        for sid, mid in recovered_pairs:
            print(f"    S1={sid}  match={mid}")

    # ---- diagnostic summary ------------------------------------------------
    print(f"\n=== DIAGNOSTIC SUMMARY ===")
    print(f"Validation S1s: {len(val_id_list)}")
    print(f"Validation true matches: {val_true_total}")
    print(f"Blocking failures: {original_blocking_failures} total "
          f"({len(cat_a)} cap-dropped, {len(cat_b)} never-generated)")
    print(f"\nStrategy ranking (by cat-A pairs recovered, then by new pairs lost):")
    ranked = sorted(
        [(name, stats[name]["cat_a_recovered"],
          stats[name]["cat_a_lost"] if name != "address_rescue" else 0,
          stats[name]["recovered"])
         for name in STRATEGY_NAMES],
        key=lambda x: (-x[1], x[2], -x[3]),
    )
    for rank, (name, cat_a_rcvd, new_lost, total_rcvd) in enumerate(ranked, 1):
        br = total_rcvd / val_true_total if val_true_total else 0.0
        print(f"  {rank}. {name:<20}  cat-A recovered={cat_a_rcvd}/{len(cat_a)}  "
              f"new_lost={new_lost}  total_recovered={total_rcvd}  blocking_recall={br:.2%}")

    print(f"\nRecommendation:")
    best = ranked[0]
    best_name, best_cat_a, best_lost, best_total = best
    if best_name == "address_rescue":
        print(f"  No alternative strategy improves on address_rescue for cat-A recovery.")
        print(f"  The {len(cat_a)} cap-dropped failures are near the score boundary;")
        print(f"  increasing CAP or adding a new blocking signal is the only path forward.")
    else:
        delta = best_cat_a - stats["address_rescue"]["cat_a_recovered"]
        print(f"  '{best_name}' recovers {delta} additional cat-A pair(s) vs address_rescue")
        print(f"  at the cost of {best_lost} newly lost true match(es).")
        if best_lost == 0:
            print(f"  Net gain is positive with zero regression — promising for a future experiment.")
            print(f"  Suggested next step: run a full train/calibration/validation experiment")
            print(f"  replacing address_rescue ranking with '{best_name}' and measure F0.5 impact.")
        else:
            print(f"  Net gain={best_cat_a - best_lost}. Evaluate whether the trade-off is acceptable")
            print(f"  in a full train/calibration/validation experiment before adopting.")

    print(f"\nTotal wall time: {time.perf_counter()-wall_start:.1f}s")
    print("Scope: training data only; no test inference; no model training; "
          "no submission files; production unchanged.")


if __name__ == "__main__":
    main()
