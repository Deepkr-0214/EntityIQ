#!/usr/bin/env python3
"""Final High-Performance Test Inference — Frozen Configuration.

FROZEN CONFIGURATION (strictly preserved):
  - Model: output/_frozen_lr_model.pkl
  - Logistic Regression, class_weight='balanced'
  - 40 features (27 pairwise + 13 blocker)
  - Address Rescue ranking: WEIGHTS = [8, 7, 2, 8, 6, 6, 1]
  - CAP = 200
  - threshold = 0.998722
  - France processed as open-set country
  - No GT used during test inference

TOTAL S1: 1,732,544
  - France: 259,452
  - India:  809,986
  - US:     663,106
"""
from __future__ import annotations

import gc
import os
import pickle
import sys
import time
from array import array
from collections import defaultdict
from pathlib import Path

import numba as nb
import numpy as np
import psutil
from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.pipeline.run_pipeline import stream_records
from src.preprocessing.normalization import (
    normalize_record,
    extract_name_tokens,
    extract_addr_significant_tokens,
    get_consonant_ngrams,
    extract_numeric_tokens,
)
from src.features.similarity_features import (
    jaccard_token, norm_lev, tok_sort, tok_set, partial, wratio,
    monge_elkan, acronym_match, numeric_features, length_diff,
    length_ratio, token_count_diff, FEATURE_COLUMNS,
)
from experiment_matching_model import (
    BLOCK_FEATURES, FEATURES, CAP, address_rescue_score
)
from src.modeling.train import predict_proba

# Paths
SPLITS      = ROOT / "splits"
OUTPUT_DIR  = ROOT / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MATCHING_PATH  = OUTPUT_DIR / "matching_results.tsv"
CANDIDATE_PATH = OUTPUT_DIR / "candidate_pairs.tsv"
FROZEN_MODEL   = OUTPUT_DIR / "_frozen_lr_model.pkl"

# Constants
FROZEN_THRESHOLD = 0.998722
BATCH_SIZE       = 1_000   # Write and report every 1,000 S1 records
COUNTRIES        = ("france", "india", "us")
TOTAL_EXPECTED_S1 = 1_732_544

WEIGHTS = np.asarray([8, 7, 2, 8, 6, 6, 1], dtype=np.int32)

def rss_mb() -> float:
    try:
        return psutil.Process().memory_info().rss / 1024 / 1024
    except Exception:
        return 0.0

# ---------------------------------------------------------------------------
# ArrayBlockingIndex
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
# CandidateStore
# ---------------------------------------------------------------------------
class CandidateStore:
    __slots__ = ("nl", "na", "al", "aa", "anums_cache", "ntoks_cache", "atoks_cache", "cngrams_cache")

    def __init__(self, nl, na, al, aa):
        self.nl = nl
        self.na = na
        self.al = al
        self.aa = aa
        # On-demand cache for top candidates to avoid recomputation
        self.anums_cache = {}
        self.ntoks_cache = {}
        self.atoks_cache = {}
        self.cngrams_cache = {}

    def get_cand_data(self, pos: int):
        nl = self.nl[pos]
        na = self.na[pos]
        al = self.al[pos]
        aa = self.aa[pos]

        if pos in self.anums_cache:
            anums = self.anums_cache[pos]
            ntoks = self.ntoks_cache[pos]
            atoks = self.atoks_cache[pos]
            cngrams = self.cngrams_cache[pos]
        else:
            anums = [n for n in extract_numeric_tokens(al) if len(n) >= 4]
            ntoks = extract_name_tokens(na)
            atoks = extract_addr_significant_tokens(aa)
            cngrams = get_consonant_ngrams(na, 3)
            # Cache if cache size reasonable
            if len(self.anums_cache) < 500_000:
                self.anums_cache[pos] = anums
                self.ntoks_cache[pos] = ntoks
                self.atoks_cache[pos] = atoks
                self.cngrams_cache[pos] = cngrams

        return nl, na, al, aa, anums, ntoks, atoks, cngrams

# ---------------------------------------------------------------------------
# Numba Blocker Kernel
# ---------------------------------------------------------------------------
@nb.njit(fastmath=True)
def _numba_score_and_filter(
    p0, p1, p2_flat, p2_offsets,
    p3_flat, p3_offsets,
    p4_flat, p4_offsets,
    p5, p6_flat, p6_offsets,
    features, touched, pool_buf, scores_buf
):
    num_touched = 0

    # 0: first_token
    for i in range(len(p0)):
        idx = p0[i]
        if features[idx, 0] == 0:
            if (features[idx, 0] | features[idx, 1] | features[idx, 2] |
                features[idx, 3] | features[idx, 4] | features[idx, 5] | features[idx, 6]) == 0:
                touched[num_touched] = idx
                num_touched += 1
            features[idx, 0] = 1

    # 1: name_prefix
    for i in range(len(p1)):
        idx = p1[i]
        if features[idx, 1] == 0:
            if (features[idx, 0] | features[idx, 1] | features[idx, 2] |
                features[idx, 3] | features[idx, 4] | features[idx, 5] | features[idx, 6]) == 0:
                touched[num_touched] = idx
                num_touched += 1
            features[idx, 1] = 1

    # 2: name_tokens
    for t in range(len(p2_offsets) - 1):
        for i in range(p2_offsets[t], p2_offsets[t + 1]):
            idx = p2_flat[i]
            if (features[idx, 0] | features[idx, 1] | features[idx, 2] |
                features[idx, 3] | features[idx, 4] | features[idx, 5] | features[idx, 6]) == 0:
                touched[num_touched] = idx
                num_touched += 1
            features[idx, 2] += 1

    # 3: addr_nums
    for t in range(len(p3_offsets) - 1):
        for i in range(p3_offsets[t], p3_offsets[t + 1]):
            idx = p3_flat[i]
            if (features[idx, 0] | features[idx, 1] | features[idx, 2] |
                features[idx, 3] | features[idx, 4] | features[idx, 5] | features[idx, 6]) == 0:
                touched[num_touched] = idx
                num_touched += 1
            features[idx, 3] += 1

    # 4: addr_tokens
    for t in range(len(p4_offsets) - 1):
        for i in range(p4_offsets[t], p4_offsets[t + 1]):
            idx = p4_flat[i]
            if (features[idx, 0] | features[idx, 1] | features[idx, 2] |
                features[idx, 3] | features[idx, 4] | features[idx, 5] | features[idx, 6]) == 0:
                touched[num_touched] = idx
                num_touched += 1
            features[idx, 4] += 1

    # 5: cskel3
    for i in range(len(p5)):
        idx = p5[i]
        if features[idx, 5] == 0:
            if (features[idx, 0] | features[idx, 1] | features[idx, 2] |
                features[idx, 3] | features[idx, 4] | features[idx, 5] | features[idx, 6]) == 0:
                touched[num_touched] = idx
                num_touched += 1
            features[idx, 5] = 1

    # 6: cngrams
    for t in range(len(p6_offsets) - 1):
        for i in range(p6_offsets[t], p6_offsets[t + 1]):
            idx = p6_flat[i]
            if (features[idx, 0] | features[idx, 1] | features[idx, 2] |
                features[idx, 3] | features[idx, 4] | features[idx, 5] | features[idx, 6]) == 0:
                touched[num_touched] = idx
                num_touched += 1
            features[idx, 6] += 1

    # Filter pool and score
    pool_size = 0
    for i in range(num_touched):
        idx = touched[i]
        f0 = features[idx, 0]
        f1 = features[idx, 1]
        f2 = features[idx, 2]
        f3 = features[idx, 3]
        f4 = features[idx, 4]
        f5 = features[idx, 5]
        f6 = features[idx, 6]

        if (f0 > 0) or (f1 > 0) or (f2 > 0) or (f3 > 0) or (f5 > 0) or (f4 >= 2) or (f6 >= 2):
            score = 8*f0 + 7*f1 + 2*f2 + 8*f3 + 6*f4 + 6*f5 + 1*f6
            pool_buf[pool_size] = idx
            scores_buf[pool_size] = score
            pool_size += 1

    return num_touched, pool_size

@nb.njit(fastmath=True)
def _numba_clear(features, touched, num_touched):
    for i in range(num_touched):
        idx = touched[i]
        features[idx, 0] = 0
        features[idx, 1] = 0
        features[idx, 2] = 0
        features[idx, 3] = 0
        features[idx, 4] = 0
        features[idx, 5] = 0
        features[idx, 6] = 0

_EMPTY_INT32 = np.empty(0, dtype=np.int32)
_EMPTY_OFFSETS = np.array([0], dtype=np.int32)

class NumbaCandidateGenerator:
    def __init__(self, idx: ArrayBlockingIndex, cids: list[str]):
        self.idx = idx
        self.cids = cids
        self.n = len(cids)
        self.features = np.zeros((self.n, 7), dtype=np.uint16)
        self.touched = np.zeros(self.n, dtype=np.int32)
        self.pool_buf = np.zeros(self.n, dtype=np.int32)
        self.scores_buf = np.zeros(self.n, dtype=np.int32)

    def query(self, s1: dict) -> list[tuple[int, tuple, int, int]]:
        ft = s1.get("ft")
        p0 = np.frombuffer(self.idx.first_token[ft], dtype=np.int32) if (ft and ft in self.idx.first_token) else _EMPTY_INT32

        npfx = s1.get("npfx")
        p1 = np.frombuffer(self.idx.name_prefix[npfx], dtype=np.int32) if (npfx and npfx in self.idx.name_prefix) else _EMPTY_INT32

        ntoks = s1.get("ntoks", ())
        if ntoks:
            p2_list = [np.frombuffer(self.idx.name_tokens[t], dtype=np.int32) for t in ntoks if t in self.idx.name_tokens]
            if p2_list:
                p2_flat = np.concatenate(p2_list)
                lens = [len(p) for p in p2_list]
                p2_offsets = np.empty(len(p2_list) + 1, dtype=np.int32)
                p2_offsets[0] = 0
                np.cumsum(lens, out=p2_offsets[1:])
            else:
                p2_flat, p2_offsets = _EMPTY_INT32, _EMPTY_OFFSETS
        else:
            p2_flat, p2_offsets = _EMPTY_INT32, _EMPTY_OFFSETS

        anums = s1.get("anums", ())
        if anums:
            anums_set = set(anums)
            p3_list = [np.frombuffer(self.idx.addr_nums[n], dtype=np.int32) for n in anums_set if n in self.idx.addr_nums]
            if p3_list:
                p3_flat = np.concatenate(p3_list)
                lens = [len(p) for p in p3_list]
                p3_offsets = np.empty(len(p3_list) + 1, dtype=np.int32)
                p3_offsets[0] = 0
                np.cumsum(lens, out=p3_offsets[1:])
            else:
                p3_flat, p3_offsets = _EMPTY_INT32, _EMPTY_OFFSETS
        else:
            p3_flat, p3_offsets = _EMPTY_INT32, _EMPTY_OFFSETS

        atoks = s1.get("atoks", ())
        if len(atoks) >= 2:
            p4_list = [np.frombuffer(self.idx.addr_tokens[t], dtype=np.int32) for t in atoks if t in self.idx.addr_tokens]
            if p4_list:
                p4_flat = np.concatenate(p4_list)
                lens = [len(p) for p in p4_list]
                p4_offsets = np.empty(len(p4_list) + 1, dtype=np.int32)
                p4_offsets[0] = 0
                np.cumsum(lens, out=p4_offsets[1:])
            else:
                p4_flat, p4_offsets = _EMPTY_INT32, _EMPTY_OFFSETS
        else:
            p4_flat, p4_offsets = _EMPTY_INT32, _EMPTY_OFFSETS

        cs3 = s1.get("cskel3", "")
        p5 = np.frombuffer(self.idx.cskel3[cs3], dtype=np.int32) if (cs3 and len(cs3) >= 3 and cs3 in self.idx.cskel3) else _EMPTY_INT32

        cngrams = s1.get("cngrams", ())
        if len(cngrams) >= 2:
            p6_list = [np.frombuffer(self.idx.cngrams[g], dtype=np.int32) for g in cngrams if g in self.idx.cngrams]
            if p6_list:
                p6_flat = np.concatenate(p6_list)
                lens = [len(p) for p in p6_list]
                p6_offsets = np.empty(len(p6_list) + 1, dtype=np.int32)
                p6_offsets[0] = 0
                np.cumsum(lens, out=p6_offsets[1:])
            else:
                p6_flat, p6_offsets = _EMPTY_INT32, _EMPTY_OFFSETS
        else:
            p6_flat, p6_offsets = _EMPTY_INT32, _EMPTY_OFFSETS

        num_touched, pool_size = _numba_score_and_filter(
            p0, p1, p2_flat, p2_offsets,
            p3_flat, p3_offsets,
            p4_flat, p4_offsets,
            p5, p6_flat, p6_offsets,
            self.features, self.touched, self.pool_buf, self.scores_buf
        )

        if pool_size == 0:
            _numba_clear(self.features, self.touched, num_touched)
            return []

        pool = self.pool_buf[:pool_size]
        scores = self.scores_buf[:pool_size]
        take = min(CAP, pool_size)

        if pool_size <= CAP:
            chosen_local = np.arange(pool_size)
        else:
            threshold = np.partition(scores, pool_size - take)[pool_size - take]
            above = np.flatnonzero(scores > threshold)
            ties = np.flatnonzero(scores == threshold)
            remaining = take - above.size
            if remaining:
                import heapq
                tie_positions = pool[ties]
                tie_local = heapq.nsmallest(
                    remaining, range(ties.size),
                    key=lambda i: self.cids[int(tie_positions[i])],
                )
                chosen_local = np.concatenate((above, ties[np.asarray(tie_local, dtype=np.int32)]))
            else:
                chosen_local = above

        selected_positions = pool[chosen_local]
        selected_scores = scores[chosen_local]
        order = sorted(
            range(selected_positions.size),
            key=lambda i: (-int(selected_scores[i]), self.cids[int(selected_positions[i])]),
        )
        ranked = selected_positions[np.asarray(order, dtype=np.int32)]

        top_positions = ranked.tolist()
        result = []
        for rank_idx, pos in enumerate(top_positions, 1):
            blocker = tuple(int(v) for v in self.features[pos].tolist())
            score = int(scores[chosen_local[order[rank_idx - 1]]])
            result.append((pos, blocker, score, rank_idx))

        _numba_clear(self.features, self.touched, num_touched)
        return result

# ---------------------------------------------------------------------------
# Feature Vector Construction
# ---------------------------------------------------------------------------
def make_feature_vector_optimized(
    s1_nl, s1_na, s1_al, s1_aa, s1_anums, s1_ntoks, s1_atoks, s1_cngrams,
    c_nl, c_na, c_al, c_aa, c_anums, c_ntoks, c_atoks, c_cngrams, c_id,
    blocker, rank
) -> list[float]:
    # Name features
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

    # Country match is 1 in per-country processing
    country_match    = 1.0

    name_len_diff    = float(length_diff(s1_na, c_na))
    name_len_ratio   = length_ratio(s1_na, c_na)
    addr_len_diff    = float(length_diff(s1_aa, c_aa))
    addr_len_ratio   = length_ratio(s1_aa, c_aa)
    name_tc_diff     = float(token_count_diff(s1_na, c_na))
    addr_tc_diff     = float(token_count_diff(s1_aa, c_aa))

    # Blocking features
    shared_name      = float(blocker[2])
    shared_addr      = float(blocker[4])
    shared_ngram     = float(blocker[6])

    nt_inter = len(s1_ntoks & c_ntoks) if s1_ntoks and c_ntoks else 0
    nt_union = len(s1_ntoks | c_ntoks) if s1_ntoks or c_ntoks else 0
    at_inter = len(s1_atoks & c_atoks) if s1_atoks and c_atoks else 0
    at_union = len(s1_atoks | c_atoks) if s1_atoks or c_atoks else 0
    cg_inter = len(s1_cngrams & c_cngrams) if s1_cngrams and c_cngrams else 0
    cg_union = len(s1_cngrams | c_cngrams) if s1_cngrams or c_cngrams else 0

    name_tok_jacc    = nt_inter / nt_union if nt_union else 0.0
    addr_tok_jacc    = at_inter / at_union if at_union else 0.0
    cng_jacc         = cg_inter / cg_union if cg_union else 0.0

    ar_score = float(address_rescue_score(blocker))
    is_s3    = 1.0 if c_id.startswith("S3-") else 0.0

    return [
        name_jaccard, name_lev, name_tok_sort, name_tok_set,
        name_partial, name_wratio, name_me, name_acro,
        name_exact, name_exact_agg,
        name_light_lev, name_light_ts,
        addr_jacc, addr_lev, addr_ts, addr_tset, addr_part,
        float(nm), float(nf), float(nl_),
        country_match,
        name_len_diff, name_len_ratio,
        addr_len_diff, addr_len_ratio,
        name_tc_diff, addr_tc_diff,
        float(blocker[0]), float(blocker[1]), shared_name,
        float(blocker[3]), shared_addr,
        float(blocker[5]), shared_ngram,
        ar_score, float(rank), is_s3,
        name_tok_jacc, addr_tok_jacc, cng_jacc,
    ]

# ---------------------------------------------------------------------------
# Build country candidate index & store
# ---------------------------------------------------------------------------
def build_country_index(country: str) -> tuple[ArrayBlockingIndex, list[str], CandidateStore]:
    t0 = time.perf_counter()
    idx = ArrayBlockingIndex()
    cids = []
    seen = set()
    _nl, _na, _al, _aa = [], [], [], []

    for source in ("S2", "S3"):
        path = SPLITS / f"test_{source}_{country}.tsv"
        for eid, name, address, ctry in stream_records(path):
            if eid in seen:
                continue
            seen.add(eid)
            rec = normalize_record(eid, name, address, ctry)
            pos = len(cids)
            cids.append(eid)
            idx.add(pos, rec)
            _nl.append(rec["nl"])
            _na.append(rec["na"])
            _al.append(rec["al"])
            _aa.append(rec["aa"])

    store = CandidateStore(_nl, _na, _al, _aa)
    del seen
    print(f"  {country.upper()} candidate index built: {len(cids):,} candidates in {time.perf_counter()-t0:.1f}s, RSS: {rss_mb():.1f} MB", flush=True)
    return idx, cids, store

# ---------------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("FINAL HIGH-PERFORMANCE TEST INFERENCE")
    print("=" * 60)
    print(f"  expected S1 total = {TOTAL_EXPECTED_S1:,}")
    print(f"  France = 259,452")
    print(f"  India = 809,986")
    print(f"  US = 663,106")
    print(f"  CAP = {CAP}")
    print(f"  threshold = {FROZEN_THRESHOLD}")
    print(f"  model = _frozen_lr_model.pkl")
    print("=" * 60, flush=True)

    # 1. Load frozen model
    if not FROZEN_MODEL.exists():
        raise FileNotFoundError(f"Frozen model not found at {FROZEN_MODEL}")

    with open(FROZEN_MODEL, "rb") as f:
        saved = pickle.load(f)
        model = saved["model"]
        scaler = saved["scaler"]
    print(f"Loaded frozen LR model. Initial RSS: {rss_mb():.1f} MB", flush=True)

    # 2. Check for resume
    completed_s1_match = set()
    completed_s1_cand  = set()

    if MATCHING_PATH.exists() and MATCHING_PATH.stat().st_size > 0:
        with open(MATCHING_PATH, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if parts and parts[0] != "source1_entity_id":
                    completed_s1_match.add(parts[0])

    if CANDIDATE_PATH.exists() and CANDIDATE_PATH.stat().st_size > 0:
        with open(CANDIDATE_PATH, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if parts and parts[0] != "source1_entity_id":
                    completed_s1_cand.add(parts[0])

    completed_s1 = completed_s1_match & completed_s1_cand
    print(f"Resuming: {len(completed_s1):,} S1s already processed.", flush=True)

    # Prepare files (write headers if new)
    if not MATCHING_PATH.exists() or MATCHING_PATH.stat().st_size == 0:
        with open(MATCHING_PATH, "w", encoding="utf-8") as f:
            f.write("source1_entity_id\tmatched_entity_ids\n")

    if not CANDIDATE_PATH.exists() or CANDIDATE_PATH.stat().st_size == 0:
        with open(CANDIDATE_PATH, "w", encoding="utf-8") as f:
            f.write("source1_entity_id\tcandidate_entity_ids\n")

    total_processed = len(completed_s1)
    total_matched = 0
    t_start = time.perf_counter()

    # Open append file handles
    f_match = open(MATCHING_PATH, "a", encoding="utf-8", buffering=1024*1024)
    f_cand  = open(CANDIDATE_PATH, "a", encoding="utf-8", buffering=1024*1024)

    try:
        for country in COUNTRIES:
            print(f"\n==================================================", flush=True)
            print(f"  STARTING COUNTRY: {country.upper()}", flush=True)
            print(f"==================================================", flush=True)

            s1_path = SPLITS / f"test_S1_{country}.tsv"
            if not s1_path.exists():
                print(f"Warning: {s1_path} not found, skipping!", flush=True)
                continue

            # Load S1 records for this country
            t_load = time.perf_counter()
            country_s1_recs = []
            for eid, name, address, ctry in stream_records(s1_path):
                if eid in completed_s1:
                    continue
                rec = normalize_record(eid, name, address, ctry)
                country_s1_recs.append(rec)

            n_country_total = len(country_s1_recs)
            print(f"  {country.upper()}: {n_country_total:,} S1 records to process (loaded in {time.perf_counter()-t_load:.1f}s)", flush=True)

            if n_country_total == 0:
                print(f"  {country.upper()}: All S1s already processed. Moving to next.", flush=True)
                continue

            # Build index for this country
            idx, cids, store = build_country_index(country)
            generator = NumbaCandidateGenerator(idx, cids)

            # Warmup JIT
            generator.query(country_s1_recs[0])

            country_processed = 0
            t_country_start = time.perf_counter()

            # Batch processing
            match_buf = []
            cand_buf  = []

            for i, s1 in enumerate(country_s1_recs, 1):
                sid = s1["id"]
                top_cands = generator.query(s1)

                if not top_cands:
                    cand_buf.append(f"{sid}\t\n")
                    match_buf.append(f"{sid}\t\n")
                else:
                    cand_ids = [cids[pos] for pos, _, _, _ in top_cands]
                    cand_buf.append(f"{sid}\t{' '.join(cand_ids)}\n")

                    # Precompute S1 fields once
                    s1_nl = s1.get("nl", "")
                    s1_na = s1.get("na", "")
                    s1_al = s1.get("al", "")
                    s1_aa = s1.get("aa", "")
                    s1_anums = s1.get("anums", [])
                    s1_ntoks = s1.get("ntoks", set())
                    s1_atoks = s1.get("atoks", set())
                    s1_cngrams = s1.get("cngrams", set())

                    # Featurize top candidates
                    feature_vectors = []
                    for pos, blocker, score, rank in top_cands:
                        c_nl, c_na, c_al, c_aa, c_anums, c_ntoks, c_atoks, c_cngrams = store.get_cand_data(pos)
                        cid = cids[pos]
                        vec = make_feature_vector_optimized(
                            s1_nl, s1_na, s1_al, s1_aa, s1_anums, s1_ntoks, s1_atoks, s1_cngrams,
                            c_nl, c_na, c_al, c_aa, c_anums, c_ntoks, c_atoks, c_cngrams, cid,
                            blocker, rank
                        )
                        feature_vectors.append(vec)

                    # Predict
                    X = np.asarray(feature_vectors, dtype=np.float32)
                    probs = predict_proba(model, X, scaler=scaler)
                    matched_indices = np.flatnonzero(probs >= FROZEN_THRESHOLD)

                    if matched_indices.size > 0:
                        matched_cids = [cand_ids[idx_match] for idx_match in matched_indices]
                        match_buf.append(f"{sid}\t{' '.join(matched_cids)}\n")
                        total_matched += len(matched_cids)
                    else:
                        match_buf.append(f"{sid}\t\n")

                country_processed += 1
                total_processed += 1

                # Flush buffer and print progress
                if i % BATCH_SIZE == 0 or i == n_country_total:
                    f_cand.writelines(cand_buf)
                    f_match.writelines(match_buf)
                    f_cand.flush()
                    f_match.flush()
                    cand_buf.clear()
                    match_buf.clear()

                    elapsed_total = time.perf_counter() - t_start
                    elapsed_country = time.perf_counter() - t_country_start
                    rate_country = country_processed / elapsed_country if elapsed_country > 0 else 0
                    rem_country = n_country_total - country_processed
                    eta_country_s = rem_country / rate_country if rate_country > 0 else 0

                    rem_total = TOTAL_EXPECTED_S1 - total_processed
                    rate_overall = (total_processed - len(completed_s1)) / elapsed_total if elapsed_total > 0 else rate_country
                    eta_total_s = rem_total / rate_overall if rate_overall > 0 else 0

                    print(
                        f"  [{country.upper()}] {country_processed:,}/{n_country_total:,} S1s | "
                        f"Total: {total_processed:,}/{TOTAL_EXPECTED_S1:,} ({total_processed/TOTAL_EXPECTED_S1*100:.1f}%) | "
                        f"Rate: {rate_country:.1f} S1/s | "
                        f"Country ETA: {eta_country_s/60:.1f}m | "
                        f"Total ETA: {eta_total_s/3600:.2f}h | "
                        f"RAM: {rss_mb():.0f} MB",
                        flush=True
                    )

            print(f"  Completed {country.upper()} in {time.perf_counter()-t_country_start:.1f}s", flush=True)

            # Free memory
            del idx, cids, store, generator, country_s1_recs
            gc.collect()
            print(f"  RAM after cleanup: {rss_mb():.1f} MB", flush=True)

    finally:
        f_match.close()
        f_cand.close()

    total_time = time.perf_counter() - t_start
    print("\n" + "=" * 60)
    print("ALL COUNTRIES PROCESSED SUCCESSFULLY!")
    print(f"Total time: {total_time/3600:.2f} hours ({total_time:.1f}s)")
    print(f"Total S1 records processed: {total_processed:,}")
    print(f"Total matches predicted: {total_matched:,}")
    print("=" * 60, flush=True)

if __name__ == "__main__":
    main()
