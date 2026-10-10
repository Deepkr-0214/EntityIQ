#!/usr/bin/env python3
"""Experiment with deterministic ranking of existing blocking candidates only."""
import sys
import time
from collections import defaultdict
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.normalization import normalize_record
from src.pipeline.run_pipeline import CompactBlockingIndex, stream_records, load_ground_truth

SPLIT_DIR = PROJECT_ROOT / "splits"
SAMPLE_SIZE = 20
CAP = 200
INDEX_PROGRESS_EVERY = 500_000
EVAL_PROGRESS_EVERY = 5
MAX_MISS_EXAMPLES = 30


def signals_and_score(s1, match):
    """Calculate only the key matches/overlaps used by CompactBlockingIndex."""
    signals = []
    strategies = 0
    score = 0

    if s1.get("ft") and s1["ft"] == match.get("ft"):
        signals.append("first_token")
        strategies += 1
        score += 10
    if s1.get("npfx") and s1["npfx"] == match.get("npfx"):
        signals.append("name_prefix")
        strategies += 1
        score += 10

    shared_name_tokens = len(s1.get("ntoks", set()) & match.get("ntoks", set()))
    if shared_name_tokens:
        signals.append(f"name_tokens={shared_name_tokens}")
        strategies += 1
        score += 4 + 3 * shared_name_tokens

    shared_numbers = len(set(s1.get("anums", [])) & set(match.get("anums", [])))
    if shared_numbers:
        signals.append(f"address_numbers={shared_numbers}")
        strategies += 1
        score += 4 + 2 * shared_numbers

    shared_address_tokens = len(s1.get("atoks", set()) & match.get("atoks", set()))
    if shared_address_tokens >= 2:
        signals.append(f"address_tokens={shared_address_tokens}")
        strategies += 1
        score += 4 + 2 * shared_address_tokens

    cs3 = s1.get("cskel3", "")
    if cs3 and len(cs3) >= 3 and cs3 == match.get("cskel3", ""):
        signals.append("consonant_skeleton")
        strategies += 1
        score += 8

    shared_cngrams = len(s1.get("cngrams", set()) & match.get("cngrams", set()))
    if shared_cngrams >= 2:
        signals.append(f"consonant_ngrams={shared_cngrams}")
        strategies += 1
        score += 4 + min(shared_cngrams, 20)

    # Reward consensus across distinct strategies, then use weighted signal
    # strength above. This only ranks candidates; it adds no new candidates.
    score += 3 * strategies
    return score, strategies, signals


def rank_existing_candidates(s1, idx, candidate_ids):
    """Reproduce the seven existing candidate rules and score their union."""
    scores = defaultdict(int)

    ft = s1.get("ft", "")
    if ft:
        for pos in idx.first_token.get(ft, ()):
            scores[pos] += 13  # first-token signal plus strategy-consensus bonus

    npfx = s1.get("npfx", "")
    if npfx:
        for pos in idx.name_prefix.get(npfx, ()):
            scores[pos] += 13

    name_counts = defaultdict(int)
    for token in s1.get("ntoks", set()):
        for pos in idx.name_tokens.get(token, ()):
            name_counts[pos] += 1
    for pos, overlap in name_counts.items():
        scores[pos] += 7 + 3 * overlap
    del name_counts

    number_counts = defaultdict(int)
    for number in set(s1.get("anums", [])):
        for pos in idx.addr_nums.get(number, ()):
            number_counts[pos] += 1
    for pos, overlap in number_counts.items():
        scores[pos] += 7 + 2 * overlap
    del number_counts

    address_counts = defaultdict(int)
    atoks = s1.get("atoks", set())
    if len(atoks) >= 2:
        for token in atoks:
            for pos in idx.addr_tokens.get(token, ()):
                address_counts[pos] += 1
    for pos, overlap in address_counts.items():
        if overlap >= 2:
            scores[pos] += 7 + 2 * overlap
    del address_counts

    cs3 = s1.get("cskel3", "")
    if cs3 and len(cs3) >= 3:
        for pos in idx.cskel3.get(cs3, ()):
            scores[pos] += 11

    cngram_counts = defaultdict(int)
    cngrams = s1.get("cngrams", set())
    if len(cngrams) >= 2:
        for cng in cngrams:
            for pos in idx.cngrams.get(cng, ()):
                cngram_counts[pos] += 1
    for pos, overlap in cngram_counts.items():
        if overlap >= 2:
            scores[pos] += 7 + min(overlap, 20)
    del cngram_counts

    # The keys must be exactly the existing unbounded blocker union.
    ranked = sorted(scores, key=lambda pos: (-scores[pos], candidate_ids[pos]))
    return ranked, scores


def diagnose_country(country):
    paths = [SPLIT_DIR / f"train_{source}_{country}.tsv" for source in ("S1", "S2", "S3", "GT")]
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
    s1_path, s2_path, s3_path, gt_path = paths

    gt = load_ground_truth(gt_path)
    matched_gt = {sid: mids for sid, mids in gt.items() if mids}
    s1_recs = []
    for eid, name, addr, ctry in stream_records(s1_path):
        if eid in matched_gt:
            s1_recs.append(normalize_record(eid, name, addr, ctry))
            if len(s1_recs) >= SAMPLE_SIZE:
                break
    wanted_match_ids = set()
    for rec in s1_recs:
        wanted_match_ids.update(matched_gt.get(rec["id"], set()))

    print(f"\n{country.upper()}: building index for {len(s1_recs)} sampled S1s", flush=True)
    index_t0 = time.perf_counter()
    idx = CompactBlockingIndex()
    candidate_ids = []
    seen_ids = set()
    gt_positions = {}
    gt_records = {}
    for path in (s2_path, s3_path):
        for eid, name, addr, ctry in stream_records(path):
            if eid in seen_ids:
                continue
            seen_ids.add(eid)
            record = normalize_record(eid, name, addr, ctry)
            pos = len(candidate_ids)
            candidate_ids.append(eid)
            idx.add(pos, record)
            if eid in wanted_match_ids:
                gt_positions[eid] = pos
                gt_records[eid] = record
            if len(candidate_ids) % INDEX_PROGRESS_EVERY == 0:
                print(f"  Indexed {len(candidate_ids):,} in {time.perf_counter()-index_t0:.1f}s", flush=True)
    index_seconds = time.perf_counter() - index_t0
    print(f"  Indexed {len(candidate_ids):,} candidates in {index_seconds:.1f}s", flush=True)

    counts = {"true": 0, "uncapped": 0, "current": 0, "ranked": 0}
    current_sizes = []
    ranked_sizes = []
    current_query_seconds = 0.0
    uncapped_query_seconds = 0.0
    ranking_seconds = 0.0
    key_parity_failures = 0
    current_misses = []
    ranked_misses = []

    for n, s1 in enumerate(s1_recs, 1):
        truth = matched_gt.get(s1["id"], set())
        t0 = time.perf_counter()
        current = idx.query(s1, max_cands=CAP)
        current_query_seconds += time.perf_counter() - t0
        t0 = time.perf_counter()
        uncapped = idx.query(s1, max_cands=len(candidate_ids) + 1)
        uncapped_query_seconds += time.perf_counter() - t0
        t0 = time.perf_counter()
        ranked, scores = rank_existing_candidates(s1, idx, candidate_ids)
        ranking_seconds += time.perf_counter() - t0

        # Validate that the experimental scorer neither drops nor adds a
        # candidate relative to CompactBlockingIndex.query's uncapped union.
        if set(ranked) != uncapped:
            key_parity_failures += 1

        ranked_cap = set(ranked[:CAP])
        current_sizes.append(len(current))
        ranked_sizes.append(len(ranked_cap))
        truth_positions = {gt_positions[mid] for mid in truth if mid in gt_positions}
        ranked_rank = {}
        for rank, pos in enumerate(ranked, 1):
            if pos in truth_positions:
                ranked_rank[pos] = rank

        for mid in truth:
            counts["true"] += 1
            pos = gt_positions.get(mid)
            in_uncapped = pos is not None and pos in uncapped
            in_current = pos is not None and pos in current
            in_ranked = pos is not None and pos in ranked_cap
            counts["uncapped"] += int(in_uncapped)
            counts["current"] += int(in_current)
            counts["ranked"] += int(in_ranked)
            if in_uncapped and not in_current:
                current_misses.append((s1, mid, pos, len(uncapped)))
            if in_uncapped and not in_ranked:
                match = gt_records.get(mid)
                score, strategy_count, signals = signals_and_score(s1, match) if match else (None, 0, [])
                ranked_misses.append((s1["id"], mid, score, ranked_rank.get(pos), len(uncapped), signals))

        if n % EVAL_PROGRESS_EVERY == 0 or n == len(s1_recs):
            print(f"  Evaluated {country.upper()}: {n}/{len(s1_recs)}", flush=True)

    total = counts["true"]
    print(f"  Sample: {len(s1_recs)} S1, {total} true matches; index {index_seconds:.1f}s", flush=True)
    for name in ("uncapped", "current", "ranked"):
        print(f"  {name:8s} recall: {counts[name]}/{total} "
              f"({100*counts[name]/total if total else 100:.2f}%)", flush=True)
    print(f"  Avg capped candidates: current={sum(current_sizes)/len(current_sizes):.1f}, "
          f"ranked={sum(ranked_sizes)/len(ranked_sizes):.1f}", flush=True)
    print(f"  Eval timings: current-query={current_query_seconds:.2f}s, "
          f"uncapped-query={uncapped_query_seconds:.2f}s, ranking+sort={ranking_seconds:.2f}s, "
          f"total={current_query_seconds+uncapped_query_seconds+ranking_seconds:.2f}s", flush=True)
    print(f"  Uncapped-union parity failures: {key_parity_failures}", flush=True)

    if ranked_misses:
        print("  Ranked-cap misses: S1_ID\tTRUE_MATCH_ID\tSCORE\tRANK\tTOTAL_CANDIDATES\tSIGNALS", flush=True)
        for row in ranked_misses[:MAX_MISS_EXAMPLES]:
            print(f"    {row[0]}\t{row[1]}\t{row[2]}\t{row[3]}\t{row[4]}\t{','.join(row[5])}", flush=True)
        if len(ranked_misses) > MAX_MISS_EXAMPLES:
            print(f"    ... {len(ranked_misses)-MAX_MISS_EXAMPLES} more ranked-cap misses omitted", flush=True)
    else:
        print("  Ranked-cap misses: none in this sample", flush=True)

    return {
        "country": country,
        "sampled": len(s1_recs),
        "counts": counts,
        "current_sizes": current_sizes,
        "ranked_sizes": ranked_sizes,
        "index_seconds": index_seconds,
        "current_query_seconds": current_query_seconds,
        "uncapped_query_seconds": uncapped_query_seconds,
        "ranking_seconds": ranking_seconds,
        "ranked_misses": ranked_misses,
        "parity_failures": key_parity_failures,
    }


if __name__ == "__main__":
    all_results = [diagnose_country(country) for country in ("india", "us")]
    total_true = sum(r["counts"]["true"] for r in all_results)
    print("\nCOMBINED RECALL COMPARISON", flush=True)
    for name in ("uncapped", "current", "ranked"):
        recovered = sum(r["counts"][name] for r in all_results)
        print(f"  {name:8s}: {recovered}/{total_true} "
              f"({100*recovered/total_true if total_true else 100:.2f}%)", flush=True)
    n_s1 = sum(r["sampled"] for r in all_results)
    avg_current = sum(sum(r["current_sizes"]) for r in all_results) / n_s1
    avg_ranked = sum(sum(r["ranked_sizes"]) for r in all_results) / n_s1
    print(f"  Average candidates/S1: current={avg_current:.1f}, ranked={avg_ranked:.1f}", flush=True)
    print(f"  Index time: {sum(r['index_seconds'] for r in all_results):.1f}s total", flush=True)
    print(f"  Query+ranking evaluation: {sum(r['current_query_seconds'] + r['uncapped_query_seconds'] + r['ranking_seconds'] for r in all_results):.1f}s total", flush=True)
    print(f"  Production-vs-experiment union parity failures: {sum(r['parity_failures'] for r in all_results)}", flush=True)
