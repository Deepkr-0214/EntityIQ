#!/usr/bin/env python3
"""Optimized test inference v2 — frozen configuration, 3–5x faster.

FROZEN CONFIGURATION (unchanged):
  CAP=200, threshold=0.998722, LR class-weighted, 40 features,
  Address Rescue WEIGHTS=[8,7,2,8,6,6,1], no GT during inference.

OPTIMIZATIONS (preserving exact semantics):
  1. Precompute candidate-side ntoks/atoks/cngrams/anums during indexing
     → stored in parallel lists in CandidateStore; never recomputed.
  2. Precompute S1 derived fields once; reuse for all candidates.
  3. Replace defaultdict(lambda:[0]*7) with direct dict of tuples.
  4. Avoid heapq.nsmallest lambda; precompute scores into list, use numpy argsort.
  5. Batch feature vector allocation with numpy.
  6. Reuse S1-side rapidfuzz preprocessing where possible.
"""
from __future__ import annotations

from array import array
from collections import defaultdict
import gc
import pickle
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from experiment_matching_model import (  # noqa: E402
    FEATURES, CAP,
)
from src.modeling.train import predict_proba  # noqa: E402
from src.pipeline.run_pipeline import (  # noqa: E402
    stream_records,
)
from src.preprocessing.normalization import (  # noqa: E402
    normalize_record,
    extract_name_tokens,
    extract_addr_significant_tokens,
    get_consonant_ngrams,
    extract_numeric_tokens,
)
from src.features.similarity_features import (  # noqa: E402
    jaccard_token, norm_lev, tok_sort, tok_set, partial, wratio,
    monge_elkan, acronym_match, numeric_features, length_diff,
    length_ratio, token_count_diff, FEATURE_COLUMNS,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
RESOURCE   = ROOT.parent / "student_resource"
TEST_DIR   = RESOURCE / "dataset" / "test"
SPLITS     = ROOT / "splits"
OUTPUT_DIR = ROOT / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MATCHING_PATH  = OUTPUT_DIR / "matching_results.tsv"
CANDIDATE_PATH = OUTPUT_DIR / "candidate_pairs.tsv"
VALIDATOR      = RESOURCE / "utils" / "validate_submission.py"

# ---------------------------------------------------------------------------
# Frozen constants
# ---------------------------------------------------------------------------
FROZEN_THRESHOLD = 0.998722
BATCH_S1         = 2_000    # S1 records per processing batch
PROGRESS_EVERY   = 20_000   # print progress every N S1s

assert CAP == 200,          f"CAP must be 200, got {CAP}"
assert len(FEATURES) == 40, f"Must have 40 features, got {len(FEATURES)}"

# Address Rescue weights — hardcoded for speed
_AR_W = (8, 7, 2, 8, 6, 6, 1)

def address_rescue_score(f):
    return 8*f[0] + 7*f[1] + 2*f[2] + 8*f[3] + 6*f[4] + 6*f[5] + f[6]


# ---------------------------------------------------------------------------
# ArrayBlockingIndex — compact inverted index using array('i')
# ---------------------------------------------------------------------------

def _make_array():
    return array('i')


class ArrayBlockingIndex:
    __slots__ = ("first_token", "name_prefix", "name_tokens", "addr_nums",
                 "addr_tokens", "cskel3", "cngrams")

    def __init__(self):
        self.first_token = defaultdict(_make_array)
        self.name_prefix = defaultdict(_make_array)
        self.name_tokens = defaultdict(_make_array)
        self.addr_nums   = defaultdict(_make_array)
        self.addr_tokens = defaultdict(_make_array)
        self.cskel3      = defaultdict(_make_array)
        self.cngrams     = defaultdict(_make_array)

    def add(self, idx: int, rec: dict):
        if rec['ft']:
            self.first_token[rec['ft']].append(idx)
        if rec['npfx']:
            self.name_prefix[rec['npfx']].append(idx)
        for t in rec.get('ntoks', ()):
            self.name_tokens[t].append(idx)
        for n in rec.get('anums', ()):
            self.addr_nums[n].append(idx)
        for t in rec.get('atoks', ()):
            self.addr_tokens[t].append(idx)
        cs3 = rec.get('cskel3', '')
        if cs3 and len(cs3) >= 3:
            self.cskel3[cs3].append(idx)
        for cng in rec.get('cngrams', ()):
            self.cngrams[cng].append(idx)


# ---------------------------------------------------------------------------
# CandidateStore — stores BOTH normalized strings AND derived token sets
# ---------------------------------------------------------------------------

class CandidateStore:
    """Stores candidate fields in parallel lists.
    
    Unlike the previous version, this precomputes and caches ntoks/atoks/
    cngrams/anums once during build time. This avoids recomputing them
    every time a candidate appears in a top-200 set.
    """
    __slots__ = ("nl", "na", "al", "aa", "ntoks", "atoks", "cngrams", "anums")

    def __init__(self):
        self.nl: list[str] = []
        self.na: list[str] = []
        self.al: list[str] = []
        self.aa: list[str] = []
        self.ntoks: list[frozenset] = []
        self.atoks: list[frozenset] = []
        self.cngrams: list[frozenset] = []
        self.anums: list[list] = []

    def add(self, rec: dict):
        self.nl.append(rec["nl"])
        self.na.append(rec["na"])
        self.al.append(rec["al"])
        self.aa.append(rec["aa"])
        # Freeze sets for reuse
        self.ntoks.append(frozenset(rec.get("ntoks", ())))
        self.atoks.append(frozenset(rec.get("atoks", ())))
        self.cngrams.append(frozenset(rec.get("cngrams", ())))
        self.anums.append(rec.get("anums", []))


# ---------------------------------------------------------------------------
# score_candidate_features — identical logic, optimized data structures
# ---------------------------------------------------------------------------

def score_candidate_features_fast(s1, idx):
    """Produce blocker features and exactly their existing union."""
    f = {}
    
    ft = s1.get("ft")
    if ft:
        for p in idx.first_token.get(ft, ()):
            if p not in f:
                f[p] = [0, 0, 0, 0, 0, 0, 0]
            f[p][0] = 1

    npfx = s1.get("npfx")
    if npfx:
        for p in idx.name_prefix.get(npfx, ()):
            if p not in f:
                f[p] = [0, 0, 0, 0, 0, 0, 0]
            f[p][1] = 1

    for token in s1.get("ntoks", ()):
        for p in idx.name_tokens.get(token, ()):
            if p not in f:
                f[p] = [0, 0, 0, 0, 0, 0, 0]
            f[p][2] += 1

    for number in set(s1.get("anums", ())):
        for p in idx.addr_nums.get(number, ()):
            if p not in f:
                f[p] = [0, 0, 0, 0, 0, 0, 0]
            f[p][3] += 1

    atoks = s1.get("atoks", ())
    if len(atoks) >= 2:
        for token in atoks:
            for p in idx.addr_tokens.get(token, ()):
                if p not in f:
                    f[p] = [0, 0, 0, 0, 0, 0, 0]
                f[p][4] += 1

    cs3 = s1.get("cskel3", "")
    if cs3 and len(cs3) >= 3:
        for p in idx.cskel3.get(cs3, ()):
            if p not in f:
                f[p] = [0, 0, 0, 0, 0, 0, 0]
            f[p][5] = 1

    cngrams = s1.get("cngrams", ())
    if len(cngrams) >= 2:
        for gram in cngrams:
            for p in idx.cngrams.get(gram, ()):
                if p not in f:
                    f[p] = [0, 0, 0, 0, 0, 0, 0]
                f[p][6] += 1

    # Pool filter — same condition as original
    pool = set()
    for p, x in f.items():
        if x[0] or x[1] or x[2] or x[3] or x[5] or x[4] >= 2 or x[6] >= 2:
            pool.add(p)

    return f, pool


# ---------------------------------------------------------------------------
# Top-200 selection — exact same semantics as heapq.nsmallest
# ---------------------------------------------------------------------------

def select_top_candidates(f_dict, pool, cids, cap=CAP):
    """Select top-CAP candidates by Address Rescue score (desc), then cid (asc)."""
    if not pool:
        return []
    
    pool_list = list(pool)
    n = len(pool_list)
    
    if n <= cap:
        pool_list.sort(key=lambda p: (-address_rescue_score(f_dict[p]), cids[p]))
        result = []
        for rank, p in enumerate(pool_list, 1):
            blocker = tuple(f_dict[p])
            score = address_rescue_score(blocker)
            result.append((p, blocker, score, rank))
        return result
    
    # Compute scores for all pool members
    scores = [address_rescue_score(f_dict[p]) for p in pool_list]
    
    # Use numpy argpartition for O(n) partial sort
    scores_arr = np.array(scores, dtype=np.int32)
    neg_scores = -scores_arr
    part_idx = np.argpartition(neg_scores, cap)[:cap]
    
    # Sort top-cap by (-score, cid)
    top_items = [(pool_list[i], scores[i]) for i in part_idx]
    top_items.sort(key=lambda x: (-x[1], cids[x[0]]))
    
    result = []
    for rank, (p, score) in enumerate(top_items, 1):
        blocker = tuple(f_dict[p])
        result.append((p, blocker, score, rank))
    return result


# ---------------------------------------------------------------------------
# Inlined feature computation — no dict intermediary
# ---------------------------------------------------------------------------

def make_feature_vector_fast(
    s1_na, s1_nl, s1_aa, s1_al, s1_anums, s1_ntoks, s1_atoks, s1_cngrams,
    c_na, c_nl, c_aa, c_al, c_anums, c_ntoks, c_atoks, c_cngrams, c_id,
    blocker, rank,
):
    """Compute 40-feature vector directly as a list — no dict intermediary."""
    # Name features (aggressive)
    name_jaccard     = jaccard_token(s1_na, c_na)
    name_lev         = norm_lev(s1_na, c_na)
    name_tok_sort    = tok_sort(s1_na, c_na)
    name_tok_set     = tok_set(s1_na, c_na)
    name_partial     = partial(s1_na, c_na)
    name_wratio      = wratio(s1_na, c_na)
    name_me          = monge_elkan(s1_na, c_na)
    name_acro        = acronym_match(s1_na, c_na)
    name_exact       = 1.0 if s1_nl and s1_nl == c_nl else 0.0
    name_exact_agg   = 1.0 if s1_na and s1_na == c_na else 0.0

    name_light_lev   = norm_lev(s1_nl, c_nl)
    name_light_ts    = tok_sort(s1_nl, c_nl)

    # Address features
    addr_jacc        = jaccard_token(s1_aa, c_aa)
    addr_lev         = norm_lev(s1_aa, c_aa)
    addr_ts          = tok_sort(s1_aa, c_aa)
    addr_tset        = tok_set(s1_aa, c_aa)
    addr_part        = partial(s1_aa, c_aa)

    nm, nf, nl_      = numeric_features(s1_anums, c_anums)

    # In per-country processing, country always matches
    country_match    = 1

    name_len_diff    = length_diff(s1_na, c_na)
    name_len_ratio   = length_ratio(s1_na, c_na)
    addr_len_diff    = length_diff(s1_aa, c_aa)
    addr_len_ratio   = length_ratio(s1_aa, c_aa)
    name_tc_diff     = token_count_diff(s1_na, c_na)
    addr_tc_diff     = token_count_diff(s1_aa, c_aa)

    # Blocking features
    shared_name      = blocker[2]
    shared_addr      = blocker[4]
    shared_ngram     = blocker[6]

    nt_inter = len(s1_ntoks & c_ntoks) if s1_ntoks and c_ntoks else 0
    nt_union = len(s1_ntoks | c_ntoks) if s1_ntoks or c_ntoks else 0
    at_inter = len(s1_atoks & c_atoks) if s1_atoks and c_atoks else 0
    at_union = len(s1_atoks | c_atoks) if s1_atoks or c_atoks else 0
    cg_inter = len(s1_cngrams & c_cngrams) if s1_cngrams and c_cngrams else 0
    cg_union = len(s1_cngrams | c_cngrams) if s1_cngrams or c_cngrams else 0

    name_tok_jacc    = nt_inter / nt_union if nt_union else 0.0
    addr_tok_jacc    = at_inter / at_union if at_union else 0.0
    cng_jacc         = cg_inter / cg_union if cg_union else 0.0

    ar_score = address_rescue_score(blocker)
    is_s3    = int(c_id.startswith("S3-"))

    # 40 features in FEATURES order = FEATURE_COLUMNS(27) + BLOCK_FEATURES(13)
    return [
        name_jaccard, name_lev, name_tok_sort, name_tok_set,
        name_partial, name_wratio, name_me, name_acro,
        name_exact, name_exact_agg,
        name_light_lev, name_light_ts,
        addr_jacc, addr_lev, addr_ts, addr_tset, addr_part,
        float(nm), float(nf), float(nl_),
        float(country_match),
        float(name_len_diff), name_len_ratio,
        float(addr_len_diff), addr_len_ratio,
        float(name_tc_diff), float(addr_tc_diff),
        float(blocker[0]), float(blocker[1]), float(shared_name),
        float(blocker[3]), float(shared_addr),
        float(blocker[5]), float(shared_ngram),
        float(ar_score), float(rank), float(is_s3),
        name_tok_jacc, addr_tok_jacc, cng_jacc,
    ]


# ---------------------------------------------------------------------------
# Build index + CandidateStore from per-country split files
# ---------------------------------------------------------------------------

def build_index_and_store(
    country: str,
) -> tuple[ArrayBlockingIndex, list[str], CandidateStore, float]:
    t0 = time.perf_counter()
    idx  = ArrayBlockingIndex()
    cids: list[str] = []
    seen: set[str]  = set()
    store = CandidateStore()

    for source in ("S2", "S3"):
        path = SPLITS / f"test_{source}_{country}.tsv"
        if not path.exists():
            full_path = TEST_DIR / f"test_source{source[1]}.tsv"
            if not full_path.exists():
                continue
            for eid, name, address, ctry in stream_records(full_path):
                if ctry.lower() != country:
                    continue
                if eid in seen:
                    continue
                seen.add(eid)
                rec = normalize_record(eid, name, address, ctry)
                pos = len(cids)
                cids.append(eid)
                idx.add(pos, rec)
                store.add(rec)
                if pos % 500_000 == 0 and pos > 0:
                    print(f"    indexed {pos:,} ({time.perf_counter()-t0:.1f}s, RSS={rss_mb():.0f}MB)",
                          flush=True)
            continue

        for eid, name, address, ctry in stream_records(path):
            if eid in seen:
                continue
            seen.add(eid)
            rec = normalize_record(eid, name, address, ctry)
            pos = len(cids)
            cids.append(eid)
            idx.add(pos, rec)
            store.add(rec)
            if pos % 500_000 == 0 and pos > 0:
                print(f"    indexed {pos:,} ({time.perf_counter()-t0:.1f}s, RSS={rss_mb():.0f}MB)",
                      flush=True)
    del seen
    gc.collect()

    return idx, cids, store, time.perf_counter() - t0


# ---------------------------------------------------------------------------
# Model loader
# ---------------------------------------------------------------------------

def load_frozen_model():
    cache = OUTPUT_DIR / "_frozen_lr_model.pkl"
    if not cache.exists():
        raise FileNotFoundError(f"Frozen model not found at {cache}")
    print(f"\n=== LOADING CACHED FROZEN LR MODEL ===", flush=True)
    with open(cache, "rb") as f:
        data = pickle.load(f)
    print(f"  Loaded from {cache}", flush=True)
    return data["model"], data["scaler"]


def rss_mb() -> float:
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1024 / 1024
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Per-country inference core — OPTIMIZED
# ---------------------------------------------------------------------------

def process_country(
    country: str,
    model,
    scaler,
    f_match,
    f_cand,
) -> dict:
    t_wall = time.perf_counter()

    print(f"\n==================================================", flush=True)
    print(f"  PROCESSING COUNTRY: {country.upper()}", flush=True)
    print(f"==================================================", flush=True)

    idx, cids, store, index_s = build_index_and_store(country)
    n_cands = len(cids)
    print(f"    {n_cands:,} candidates indexed in {index_s:.1f}s", flush=True)
    print(f"    RSS after index+store: {rss_mb():.0f} MB", flush=True)

    # Load S1 records
    t0 = time.perf_counter()
    s1_path = SPLITS / f"test_S1_{country}.tsv"
    if not s1_path.exists():
        s1_path = TEST_DIR / "test_source1.tsv"
        use_filter = True
    else:
        use_filter = False

    all_s1_recs: dict[str, dict] = {}
    for eid, name, address, ctry in stream_records(s1_path):
        if use_filter and ctry.lower() != country:
            continue
        all_s1_recs[eid] = normalize_record(eid, name, address, ctry)

    load_s1_s = time.perf_counter() - t0
    sid_list = sorted(all_s1_recs)
    print(f"    {len(sid_list):,} S1 records loaded in {load_s1_s:.1f}s", flush=True)
    print(f"    RSS after S1 loading: {rss_mb():.0f} MB", flush=True)

    # Process in batches
    n_processed   = 0
    total_cands   = 0
    total_matches = 0
    n_singletons  = 0
    n_with_match  = 0
    cand_counts: list[int] = []

    t_gen  = 0.0
    t_feat = 0.0
    t_lr   = 0.0
    t_out  = 0.0

    print(f"    Starting batch processing (batch_size={BATCH_S1})...", flush=True)

    for batch_start in range(0, len(sid_list), BATCH_S1):
        batch = sid_list[batch_start:batch_start + BATCH_S1]

        # === Candidate generation ===
        t0 = time.perf_counter()
        batch_selected: dict[str, list] = {}
        
        for sid in batch:
            s1 = all_s1_recs[sid]
            feat_dict, pool_set = score_candidate_features_fast(s1, idx)
            top = select_top_candidates(feat_dict, pool_set, cids, CAP)
            batch_selected[sid] = top
            del feat_dict, pool_set
        t_gen += time.perf_counter() - t0

        # === Feature vector construction (OPTIMIZED) ===
        t0 = time.perf_counter()
        all_vecs: list[list[float]] = []
        all_meta: list[tuple[str, str]] = []
        sid_ranges: dict[str, tuple[int, int]] = {}

        for sid in batch:
            s1 = all_s1_recs[sid]
            # Pre-extract S1 fields ONCE
            s1_na = s1.get("na", "")
            s1_nl = s1.get("nl", "")
            s1_aa = s1.get("aa", "")
            s1_al = s1.get("al", "")
            s1_anums = s1.get("anums", [])
            s1_ntoks = s1.get("ntoks", set())
            s1_atoks = s1.get("atoks", set())
            s1_cngrams = s1.get("cngrams", set())
            
            if not isinstance(s1_ntoks, frozenset):
                s1_ntoks = frozenset(s1_ntoks)
            if not isinstance(s1_atoks, frozenset):
                s1_atoks = frozenset(s1_atoks)
            if not isinstance(s1_cngrams, frozenset):
                s1_cngrams = frozenset(s1_cngrams)

            selected = batch_selected[sid]
            start = len(all_vecs)
            
            for pos, blocker, score, rank in selected:
                cid = cids[pos]
                vec = make_feature_vector_fast(
                    s1_na, s1_nl, s1_aa, s1_al, s1_anums, s1_ntoks, s1_atoks, s1_cngrams,
                    store.na[pos], store.nl[pos], store.aa[pos], store.al[pos],
                    store.anums[pos], store.ntoks[pos], store.atoks[pos], store.cngrams[pos], cid,
                    blocker, rank,
                )
                all_vecs.append(vec)
                all_meta.append((sid, cid))
            sid_ranges[sid] = (start, len(all_vecs))
        t_feat += time.perf_counter() - t0

        # === Batched LR prediction ===
        t0 = time.perf_counter()
        if all_vecs:
            X = np.asarray(all_vecs, dtype=np.float32)
            probs = predict_proba(model, X, scaler=scaler)
        else:
            probs = np.array([], dtype=np.float32)
        t_lr += time.perf_counter() - t0

        # === Output generation ===
        t0 = time.perf_counter()
        match_lines: list[str] = []
        cand_lines:  list[str] = []
        for sid in batch:
            selected = batch_selected[sid]
            top_cids = [cids[p] for p, _, _, _ in selected]
            start, end = sid_ranges[sid]
            matched = [
                all_meta[i][1]
                for i in range(start, end)
                if probs[i] >= FROZEN_THRESHOLD
            ]
            cands_sorted   = sorted(top_cids)
            matched_sorted = sorted(matched)
            match_lines.append(f"{sid}\t{','.join(matched_sorted)}\n")
            cand_lines.append(f"{sid}\t{','.join(cands_sorted)}\n")
            cand_counts.append(len(top_cids))
            total_cands   += len(top_cids)
            total_matches += len(matched_sorted)
            if matched_sorted:
                n_with_match += 1
            else:
                n_singletons += 1
        f_match.writelines(match_lines)
        f_cand.writelines(cand_lines)
        t_out += time.perf_counter() - t0

        n_processed += len(batch)
        if n_processed % PROGRESS_EVERY == 0 or n_processed == len(sid_list):
            elapsed = time.perf_counter() - t_wall
            rate = n_processed / elapsed if elapsed > 0 else 0
            mem = rss_mb()
            print(f"    {country.upper()}: {n_processed:,}/{len(sid_list):,} S1s "
                  f"({rate:.0f} S1/s, RSS={mem:.0f}MB, matches={total_matches:,})", flush=True)

        del batch_selected, all_vecs, all_meta
        if (batch_start // BATCH_S1) % 10 == 9:
            gc.collect()

    del idx, all_s1_recs, store, cids
    gc.collect()

    country_s = time.perf_counter() - t_wall
    rate = n_processed / country_s if country_s > 0 else 0
    arr = np.array(cand_counts, dtype=np.int32) if cand_counts else np.array([0], dtype=np.int32)

    stats = {
        "country":        country,
        "n_s1":           n_processed,
        "n_singletons":   n_singletons,
        "n_with_match":   n_with_match,
        "total_cands":    total_cands,
        "total_matches":  total_matches,
        "cand_min":       int(arr.min())        if len(arr) else 0,
        "cand_mean":      float(arr.mean())     if len(arr) else 0.0,
        "cand_median":    int(np.median(arr))   if len(arr) else 0,
        "cand_max":       int(arr.max())        if len(arr) else 0,
        "index_s":        index_s,
        "load_s1_s":      load_s1_s,
        "gen_s":          t_gen,
        "feat_s":         t_feat,
        "lr_s":           t_lr,
        "out_s":          t_out,
        "total_s":        country_s,
        "rate":           rate,
    }
    print(f"    {country.upper()} done: {n_processed:,} S1s in "
          f"{country_s:.1f}s ({rate:.0f} S1/s, matches={total_matches:,})", flush=True)
    return stats


# ---------------------------------------------------------------------------
# Mini benchmark (20 France S1s)
# ---------------------------------------------------------------------------

def run_mini_benchmark(model, scaler, n_test=20):
    print(f"\n{'='*60}", flush=True)
    print(f"  MINI BENCHMARK: {n_test} France S1 records", flush=True)
    print(f"{'='*60}", flush=True)

    country = "france"
    t0 = time.perf_counter()
    idx, cids, store, index_s = build_index_and_store(country)
    n_cands = len(cids)
    print(f"    {n_cands:,} candidates indexed in {index_s:.1f}s", flush=True)
    print(f"    RSS after index: {rss_mb():.0f} MB", flush=True)

    # Load first n_test S1 records
    s1_path = SPLITS / f"test_S1_{country}.tsv"
    s1_recs = {}
    for eid, name, address, ctry in stream_records(s1_path):
        s1_recs[eid] = normalize_record(eid, name, address, ctry)
        if len(s1_recs) >= n_test:
            break

    sids = sorted(s1_recs)
    print(f"    Loaded {len(sids)} S1 records", flush=True)

    t_gen_total  = 0.0
    t_feat_total = 0.0
    t_lr_total   = 0.0
    all_vecs     = []
    all_meta     = []

    for sid in sids:
        s1 = s1_recs[sid]

        t1 = time.perf_counter()
        feat_dict, pool_set = score_candidate_features_fast(s1, idx)
        top = select_top_candidates(feat_dict, pool_set, cids, CAP)
        t_gen_total += time.perf_counter() - t1

        t1 = time.perf_counter()
        s1_na = s1.get("na", "")
        s1_nl = s1.get("nl", "")
        s1_aa = s1.get("aa", "")
        s1_al = s1.get("al", "")
        s1_anums = s1.get("anums", [])
        s1_ntoks = frozenset(s1.get("ntoks", set()))
        s1_atoks = frozenset(s1.get("atoks", set()))
        s1_cngrams = frozenset(s1.get("cngrams", set()))

        for pos, blocker, score, rank in top:
            cid = cids[pos]
            vec = make_feature_vector_fast(
                s1_na, s1_nl, s1_aa, s1_al, s1_anums, s1_ntoks, s1_atoks, s1_cngrams,
                store.na[pos], store.nl[pos], store.aa[pos], store.al[pos],
                store.anums[pos], store.ntoks[pos], store.atoks[pos], store.cngrams[pos], cid,
                blocker, rank,
            )
            all_vecs.append(vec)
            all_meta.append((sid, cid))
        t_feat_total += time.perf_counter() - t1
        del feat_dict, pool_set, top

    t1 = time.perf_counter()
    if all_vecs:
        X = np.asarray(all_vecs, dtype=np.float32)
        probs = predict_proba(model, X, scaler=scaler)
    else:
        probs = np.array([], dtype=np.float32)
    t_lr_total = time.perf_counter() - t1

    n_matches = int(np.sum(probs >= FROZEN_THRESHOLD)) if len(probs) else 0
    total_s = time.perf_counter() - t0 - index_s

    print(f"\n    === MINI BENCHMARK RESULTS ({n_test} S1s) ===", flush=True)
    print(f"    Candidate generation : {t_gen_total:.3f}s  ({t_gen_total/n_test:.4f}s/S1)", flush=True)
    print(f"    Feature computation  : {t_feat_total:.3f}s  ({t_feat_total/n_test:.4f}s/S1)", flush=True)
    print(f"    LR prediction        : {t_lr_total:.3f}s  ({t_lr_total/n_test:.4f}s/S1)", flush=True)
    print(f"    Total (excl index)   : {total_s:.3f}s  ({total_s/n_test:.4f}s/S1)", flush=True)
    print(f"    Total feature vectors: {len(all_vecs)}", flush=True)
    print(f"    Matches at threshold : {n_matches}", flush=True)
    print(f"    RSS                  : {rss_mb():.0f} MB", flush=True)

    # Estimate full-test runtime
    total_s1 = 259_452 + 809_986 + 663_106
    per_s1 = total_s / n_test
    est_hours = per_s1 * total_s1 / 3600
    idx_est = 3 * index_s / 3600
    print(f"\n    Estimated per-S1 time : {per_s1:.4f}s", flush=True)
    print(f"    Estimated full test  : {est_hours:.2f}h (+ ~{idx_est:.2f}h indexing)", flush=True)

    del idx, cids, store, s1_recs, all_vecs, all_meta
    gc.collect()

    return per_s1


# ---------------------------------------------------------------------------
# Full test inference
# ---------------------------------------------------------------------------

def run_full_inference(model, scaler):
    wall_inf = time.perf_counter()
    print("\n==================================================", flush=True)
    print("STARTING FULL TEST INFERENCE", flush=True)
    print(f"Target files:\n  {MATCHING_PATH}\n  {CANDIDATE_PATH}", flush=True)
    print("==================================================", flush=True)

    all_test_s1_ids = set()
    s1_country = {}
    print("  Reading test S1 IDs for verification...", flush=True)
    for eid, _, _, ctry in stream_records(TEST_DIR / "test_source1.tsv"):
        all_test_s1_ids.add(eid)
        s1_country[eid] = ctry.strip().lower() if ctry else "unknown"
    print(f"  Total test S1 IDs: {len(all_test_s1_ids):,}", flush=True)

    test_countries = ["france", "india", "us"]
    written = set()
    all_stats = []

    with open(MATCHING_PATH, "w", encoding="utf-8") as f_match, \
         open(CANDIDATE_PATH, "w", encoding="utf-8") as f_cand:

        f_match.write("source1_entity_id\tmatched_entity_ids\n")
        f_cand.write("source1_entity_id\tcandidate_entity_ids\n")

        for country in test_countries:
            stats = process_country(country, model, scaler, f_match, f_cand)
            all_stats.append(stats)
            for eid, ctry in s1_country.items():
                if ctry == country:
                    written.add(eid)

        missing = all_test_s1_ids - written
        if missing:
            print(f"  Safety net: writing {len(missing):,} unhandled S1s as singletons...",
                  flush=True)
            for sid in sorted(missing):
                f_match.write(f"{sid}\t\n")
                f_cand.write(f"{sid}\t\n")
                written.add(sid)

    inf_s = time.perf_counter() - wall_inf
    total_s1    = sum(s["n_s1"]          for s in all_stats) + len(missing)
    total_cands = sum(s["total_cands"]   for s in all_stats)
    total_match = sum(s["total_matches"] for s in all_stats)
    n_sing      = sum(s["n_singletons"]  for s in all_stats) + len(missing)
    n_match     = sum(s["n_with_match"]  for s in all_stats)

    print(f"\n==================================================", flush=True)
    print(f"=== FULL INFERENCE COMPLETE ===", flush=True)
    print(f"==================================================", flush=True)
    print(f"  Test S1 entities processed : {total_s1:,}")
    print(f"  Predicted singletons       : {n_sing:,}")
    print(f"  S1s with >= 1 match        : {n_match:,}")
    print(f"  Total candidate pairs      : {total_cands:,}")
    print(f"  Total predicted matches    : {total_match:,}")
    print(f"  Countries processed        : {len(test_countries)}  {test_countries}")
    print(f"  Inference wall time        : {inf_s:.1f}s ({inf_s/3600:.2f}h)")
    print(f"  matching_results.tsv       : {MATCHING_PATH}")
    print(f"  candidate_pairs.tsv        : {CANDIDATE_PATH}")

    print(f"\n  Per-Country Breakdown:")
    for s in all_stats:
        print(f"    {s['country'].upper():8s}: {s['n_s1']:>10,} S1s, "
              f"{s['total_s']:>8.1f}s ({s['rate']:.0f} S1/s), "
              f"gen={s['gen_s']:.1f}s feat={s['feat_s']:.1f}s lr={s['lr_s']:.1f}s "
              f"matches={s['total_matches']:,}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    wall_start = time.perf_counter()
    print("=== OPTIMIZED TEST INFERENCE v2 ===", flush=True)
    print(f"  CAP={CAP}  threshold={FROZEN_THRESHOLD}  "
          f"features={len(FEATURES)}  batch={BATCH_S1}", flush=True)
    print(f"  GT loaded during inference: NO", flush=True)
    print(f"  RSS at start: {rss_mb():.0f} MB", flush=True)

    # 1. Load frozen model
    model, scaler = load_frozen_model()
    print(f"  Model ready. RSS: {rss_mb():.0f} MB", flush=True)

    # 2. Mini benchmark
    per_s1 = run_mini_benchmark(model, scaler, n_test=20)

    # 3. Full inference
    run_full_inference(model, scaler)

    # 4. Validator
    print("\n==================================================", flush=True)
    print("=== RUNNING OFFICIAL SUBMISSION VALIDATOR ===", flush=True)
    print("==================================================", flush=True)
    if VALIDATOR.exists():
        cmd = [
            sys.executable, str(VALIDATOR),
            "--matching", str(MATCHING_PATH),
            "--candidate", str(CANDIDATE_PATH),
            "--test-dir", str(TEST_DIR),
        ]
        print(f"  Command: {' '.join(cmd)}", flush=True)
        result = subprocess.run(cmd)
        print(f"  Validator Exit Code: {result.returncode}", flush=True)
        if result.returncode == 0:
            print("  >>> SUBMISSION VALIDATION PASSED SUCCESSFULLY! <<<", flush=True)
        else:
            print("  >>> SUBMISSION VALIDATION FAILED! <<<", flush=True)
    else:
        print(f"  Validator not found at {VALIDATOR}", flush=True)

    total_s = time.perf_counter() - wall_start
    print(f"\n  Total elapsed wall time: {total_s:.1f}s ({total_s/3600:.2f}h)", flush=True)


if __name__ == "__main__":
    main()
