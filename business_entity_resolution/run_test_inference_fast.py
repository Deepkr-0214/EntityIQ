#!/usr/bin/env python3
"""Memory-safe fast test inference — frozen configuration.

FROZEN CONFIGURATION (unchanged):
  CAP=200, threshold=0.998722, LR class-weighted, 40 features,
  Address Rescue WEIGHTS=[8,7,2,8,6,6,1], no GT during inference.

MEMORY STRATEGY:
  - Per-country split files (splits/test_S{1,2,3}_{country}.tsv) avoid scanning full files.
  - ArrayBlockingIndex uses array('i') for contiguous 32-bit integer postings (4 bytes/entry),
    reducing inverted index memory from 5.4 GB to <700 MB.
  - CandidateStore keeps only 4 normalized text strings (nl, na, al, aa) per candidate.
    Precomputed sets (ntoks, atoks, cngrams) and anums lists are eliminated from candidate storage;
    they are computed on-the-fly ONLY for the top-200 candidates per S1 during featurization.
  - Peak RSS stays under 2.5 GB (safe for 8 GB RAM laptop).
  - Processes France, India, and US sequentially; fully frees country index before next.
  - Runs official validate_submission.py immediately upon completion.
"""
from __future__ import annotations

from array import array
from collections import defaultdict
import gc
import heapq
import pickle
import subprocess
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
    FEATURES, CAP, make_feature_vector, stratified_sample_and_split,
    score_candidate_features, address_rescue_score,
)
from src.modeling.train import predict_proba, train_logistic_regression  # noqa: E402
from src.pipeline.run_pipeline import (  # noqa: E402
    CompactBlockingIndex, load_ground_truth, stream_records,
)
from src.preprocessing.normalization import (  # noqa: E402
    normalize_record,
    extract_name_tokens,
    extract_addr_significant_tokens,
    get_consonant_ngrams,
    extract_numeric_tokens,
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
SEED             = 42
COUNTRIES_TRAIN  = ("india", "us")
REQUESTED_SAMPLE = 200
BATCH_S1         = 2_000    # S1 records per processing batch
PROGRESS_EVERY   = 20_000   # print progress every N S1s

assert CAP == 200,          f"CAP must be 200, got {CAP}"
assert len(FEATURES) == 40, f"Must have 40 features, got {len(FEATURES)}"


# ---------------------------------------------------------------------------
# ArrayBlockingIndex — compact inverted index using array('i')
# ---------------------------------------------------------------------------

def _make_array():
    return array('i')


class ArrayBlockingIndex:
    """Inverted index using array('i') for 32-bit contiguous integer postings.
    Compatible with score_candidate_features().
    """
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
# CandidateStore — lean storage of normalized strings only
# ---------------------------------------------------------------------------

class CandidateStore:
    """Stores candidate record fields as parallel Python lists of strings.
    No set or list objects stored in candidate memory!
    Derived sets and numeric tokens are generated on the fly only for
    the top-200 selected candidates per S1 during feature vector construction.
    """
    __slots__ = ("nl", "na", "al", "aa")

    def __init__(self, nl: list[str], na: list[str], al: list[str], aa: list[str]):
        self.nl = nl
        self.na = na
        self.al = al
        self.aa = aa

    def get_rec(self, pos: int, cid: str, country: str) -> dict:
        nl = self.nl[pos]
        na = self.na[pos]
        al = self.al[pos]
        aa = self.aa[pos]
        return {
            "id":       cid,
            "nl":       nl,
            "na":       na,
            "al":       al,
            "aa":       aa,
            "country":  country,
            "anums":    [n for n in extract_numeric_tokens(al) if len(n) >= 4],
            "ntoks":    extract_name_tokens(na),
            "atoks":    extract_addr_significant_tokens(aa),
            "cngrams":  get_consonant_ngrams(na, 3),
        }


# ---------------------------------------------------------------------------
# Build index + CandidateStore from per-country split files
# ---------------------------------------------------------------------------

def build_index_and_store(
    country: str,
) -> tuple[ArrayBlockingIndex, list[str], CandidateStore, float]:
    """Single pass over per-country S2+S3 splits: build blocking index AND
    populate CandidateStore.
    """
    t0 = time.perf_counter()
    idx  = ArrayBlockingIndex()
    cids: list[str] = []
    seen: set[str]  = set()

    _nl: list[str]  = []
    _na: list[str]  = []
    _al: list[str]  = []
    _aa: list[str]  = []

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
                _nl.append(rec["nl"])
                _na.append(rec["na"])
                _al.append(rec["al"])
                _aa.append(rec["aa"])
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
            _nl.append(rec["nl"])
            _na.append(rec["na"])
            _al.append(rec["al"])
            _aa.append(rec["aa"])
            if pos % 500_000 == 0 and pos > 0:
                print(f"    indexed {pos:,} ({time.perf_counter()-t0:.1f}s, RSS={rss_mb():.0f}MB)",
                      flush=True)
    del seen
    gc.collect()

    store = CandidateStore(_nl, _na, _al, _aa)
    return idx, cids, store, time.perf_counter() - t0


# ---------------------------------------------------------------------------
# Training helpers & Frozen Model Loader
# ---------------------------------------------------------------------------

def load_or_train_model():
    """Load cached model if present, otherwise train and cache to disk."""
    cache = OUTPUT_DIR / "_frozen_lr_model.pkl"
    if cache.exists():
        print(f"\n=== LOADING CACHED FROZEN LR MODEL ===", flush=True)
        with open(cache, "rb") as f:
            data = pickle.load(f)
        print(f"  Loaded from {cache}", flush=True)
        return data["model"], data["scaler"]

    print("\n=== TRAINING FROZEN LR MODEL ===", flush=True)
    gt = load_ground_truth(SPLITS / "train_ground_truth.tsv") if (SPLITS / "train_ground_truth.tsv").exists() \
        else load_ground_truth(ROOT.parent / "student_resource" / "dataset" / "train" / "train_ground_truth.tsv")

    all_X: list[list[float]] = []
    all_y: list[int]         = []

    for country in COUNTRIES_TRAIN:
        print(f"\n  {country.upper()}", flush=True)
        train_ids, _ = stratified_sample_and_split(
            gt, country, REQUESTED_SAMPLE, SEED + (0 if country == "india" else 1)
        )
        idx_t = CompactBlockingIndex()
        cids_train: list[str] = []
        seen_t: set[str] = set()
        t0 = time.perf_counter()

        for source in ("S2", "S3"):
            path = SPLITS / f"train_{source}_{country}.tsv"
            for eid, name, address, ctry in stream_records(path):
                if eid in seen_t:
                    continue
                seen_t.add(eid)
                rec = normalize_record(eid, name, address, ctry)
                idx_t.add(len(cids_train), rec)
                cids_train.append(eid)
        del seen_t

        s1_records: dict = {}
        for eid, name, address, ctry in stream_records(
            SPLITS / f"train_S1_{country}.tsv"
        ):
            if eid in train_ids:
                s1_records[eid] = normalize_record(eid, name, address, ctry)

        selected: dict[str, list] = {}
        for sid in sorted(s1_records):
            features, pool_set = score_candidate_features(s1_records[sid], idx_t)
            top = heapq.nsmallest(
                CAP, pool_set,
                key=lambda p: (-address_rescue_score(features[p]), cids_train[p]),
            )
            rows = []
            for rank, pos in enumerate(top, 1):
                blocker = tuple(int(v) for v in features[pos])
                score   = address_rescue_score(blocker)
                rows.append((cids_train[pos], blocker, score, rank))
            selected[sid] = rows
            del features, pool_set

        needed = {cid for rows in selected.values() for cid, _, _, _ in rows}
        cand_records: dict = {}
        for source in ("S2", "S3"):
            for eid, name, address, ctry in stream_records(
                SPLITS / f"train_{source}_{country}.tsv"
            ):
                if eid in needed and eid not in cand_records:
                    cand_records[eid] = normalize_record(eid, name, address, ctry)

        truth_by_sid = {sid: set(gt.get(sid, set())) for sid in train_ids}
        for sid in sorted(s1_records):
            truth = truth_by_sid[sid]
            for cid, blocker, score, rank in selected[sid]:
                vec = make_feature_vector(
                    s1_records[sid], cand_records[cid], blocker, rank
                )
                all_X.append(vec)
                all_y.append(int(cid in truth))

        del idx_t, selected, cand_records, s1_records
        gc.collect()

    Xtr = np.asarray(all_X, dtype=np.float32)
    ytr = np.asarray(all_y, dtype=np.int8)
    model, scaler = train_logistic_regression(Xtr, ytr, feature_names=FEATURES)
    with open(cache, "wb") as f:
        pickle.dump({"model": model, "scaler": scaler}, f)
    return model, scaler


def rss_mb() -> float:
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1024 / 1024
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Per-country inference core
# ---------------------------------------------------------------------------

def process_country(
    country: str,
    model,
    scaler,
    f_match,
    f_cand,
) -> dict:
    """Process all S1s for one country. Returns timing stats."""
    t_wall = time.perf_counter()

    print(f"\n==================================================", flush=True)
    print(f"  PROCESSING COUNTRY: {country.upper()}", flush=True)
    print(f"==================================================", flush=True)

    idx, cids, store, index_s = build_index_and_store(country)
    n_cands = len(cids)
    print(f"    {n_cands:,} candidates indexed in {index_s:.1f}s", flush=True)
    print(f"    RSS after index+store: {rss_mb():.0f} MB", flush=True)

    # Load S1 records from per-country split
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

        # Candidate generation
        t0 = time.perf_counter()
        batch_top:     dict[str, list[int]]  = {}
        batch_blocker: dict[str, list[tuple]] = {}
        for sid in batch:
            s1 = all_s1_recs[sid]
            feat_dict, pool_set = score_candidate_features(s1, idx)
            top = heapq.nsmallest(
                CAP, pool_set,
                key=lambda p: (-address_rescue_score(feat_dict[p]), cids[p]),
            )
            batch_top[sid]     = top
            batch_blocker[sid] = [
                (tuple(int(v) for v in feat_dict[p]), rank)
                for rank, p in enumerate(top, 1)
            ]
            del feat_dict, pool_set
        t_gen += time.perf_counter() - t0

        # Feature vector construction
        t0 = time.perf_counter()
        all_vecs:   list[list[float]]       = []
        all_meta:   list[tuple[str, str]]   = []
        sid_ranges: dict[str, tuple[int, int]] = {}

        for sid in batch:
            s1          = all_s1_recs[sid]
            top_positions = batch_top[sid]
            blockers      = batch_blocker[sid]
            start = len(all_vecs)
            for pos, (blocker, rank) in zip(top_positions, blockers):
                cid      = cids[pos]
                cand_rec = store.get_rec(pos, cid, country)
                vec      = make_feature_vector(s1, cand_rec, blocker, rank)
                all_vecs.append(vec)
                all_meta.append((sid, cid))
            sid_ranges[sid] = (start, len(all_vecs))
        t_feat += time.perf_counter() - t0

        # Batched LR prediction
        t0 = time.perf_counter()
        if all_vecs:
            X = np.asarray(all_vecs, dtype=np.float32)
            probs = predict_proba(model, X, scaler=scaler)
        else:
            probs = np.array([], dtype=np.float32)
        t_lr += time.perf_counter() - t0

        # Output generation
        t0 = time.perf_counter()
        match_lines: list[str] = []
        cand_lines:  list[str] = []
        for sid in batch:
            top_positions = batch_top[sid]
            top_cids      = [cids[p] for p in top_positions]
            start, end    = sid_ranges[sid]
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

        del batch_top, batch_blocker, all_vecs, all_meta
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
# Full test inference execution
# ---------------------------------------------------------------------------

def run_full_inference(model, scaler):
    wall_inf = time.perf_counter()
    print("\n==================================================", flush=True)
    print("STARTING FULL TEST INFERENCE", flush=True)
    print(f"Target files:\n  {MATCHING_PATH}\n  {CANDIDATE_PATH}", flush=True)
    print("==================================================", flush=True)

    # Collect all test S1 IDs for completeness check
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    wall_start = time.perf_counter()
    print("=== FAST MEMORY-SAFE TEST INFERENCE ===", flush=True)
    print(f"  CAP={CAP}  threshold={FROZEN_THRESHOLD}  "
          f"features={len(FEATURES)}  batch={BATCH_S1}", flush=True)
    print(f"  GT loaded during inference: NO", flush=True)
    print(f"  RSS at start: {rss_mb():.0f} MB", flush=True)

    # 1. Load frozen model artifact
    model, scaler = load_or_train_model()
    print(f"  Model ready. RSS: {rss_mb():.0f} MB", flush=True)

    # 2. Run full inference immediately
    run_full_inference(model, scaler)

    # 3. Run official submission validator
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
