#!/usr/bin/env python3
"""Test inference using the frozen experimental configuration.

Frozen configuration
--------------------
  Blocking  : CompactBlockingIndex (7 strategies, unchanged production logic)
  Ranking   : Address Rescue  WEIGHTS=[8,7,2,8,6,6,1], CAP=200
  Model     : class-weighted Logistic Regression  (C=1.0, max_iter=1000, SEED=42)
  Features  : 40 (27 pairwise + 13 blocker/rank)
  Threshold : 0.998722  (global, calibration-only)
  Training  : ALL 400 sampled S1s (train split only, 236 S1s per country pair)

Countries are discovered dynamically from the test files — France, India, US
and any future country are all handled without code changes.

Ground truth is never loaded.  No external data.  No production modification.
Outputs: output/matching_results.tsv, output/candidate_pairs.tsv
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
    FEATURES, CAP, make_feature_vector, stratified_sample_and_split,
)
from optimize_matching_experiment import NumpyQueryIndex, SPLITS, WEIGHTS  # noqa: E402
from experiment_matching_model import address_rescue_score  # noqa: E402
from src.modeling.train import predict_proba, train_logistic_regression  # noqa: E402
from src.pipeline.run_pipeline import (  # noqa: E402
    CompactBlockingIndex, load_ground_truth, stream_records,
)
from src.preprocessing.normalization import normalize_record  # noqa: E402

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
RESOURCE  = ROOT.parent / "student_resource"
TEST_DIR  = RESOURCE / "dataset" / "test"
TRAIN_DIR = RESOURCE / "dataset" / "train"
OUTPUT_DIR = ROOT / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MATCHING_PATH   = OUTPUT_DIR / "matching_results.tsv"
CANDIDATE_PATH  = OUTPUT_DIR / "candidate_pairs.tsv"
VALIDATOR       = RESOURCE / "utils" / "validate_submission.py"

# ---------------------------------------------------------------------------
# Frozen configuration constants
# ---------------------------------------------------------------------------
FROZEN_THRESHOLD = 0.998722
SEED = 42
COUNTRIES_TRAIN  = ("india", "us")          # countries used for model training
REQUESTED_SAMPLE = 200
PROGRESS_EVERY   = 500


# ---------------------------------------------------------------------------
# Training helpers (identical to v2)
# ---------------------------------------------------------------------------

def choose_sample(gt, country):
    return stratified_sample_and_split(
        gt, country, REQUESTED_SAMPLE,
        SEED + (0 if country == "india" else 1),
    )


def build_index_from_paths(s2_path: Path, s3_path: Path):
    """Build CompactBlockingIndex + candidate ID list from S2 and S3 files."""
    idx = CompactBlockingIndex()
    ids: list[str] = []
    seen: set[str] = set()
    t0 = time.perf_counter()
    for path in (s2_path, s3_path):
        if not path.exists():
            continue
        for eid, name, address, ctry in stream_records(path):
            if eid in seen:
                continue
            seen.add(eid)
            idx.add(len(ids), normalize_record(eid, name, address, ctry))
            ids.append(eid)
            if len(ids) % 500_000 == 0:
                print(f"    indexed {len(ids):,} ({time.perf_counter()-t0:.1f}s)",
                      flush=True)
    return idx, ids, time.perf_counter() - t0


# ---------------------------------------------------------------------------
# Train the frozen LR model on the training split of the 400-S1 sample
# ---------------------------------------------------------------------------

def train_frozen_model():
    """Train LR on the train-split portion of the 400 sampled S1s.

    Uses the identical stratified_sample_and_split call as v2 so the train
    split is exactly the same 236 S1s per country pair (472 total).
    GT is used only for labels — never passed to candidate generation.
    """
    print("\n=== TRAINING FROZEN LR MODEL ===", flush=True)
    all_X: list = []
    all_y: list = []

    for country in COUNTRIES_TRAIN:
        print(f"\n  {country.upper()}: loading training data", flush=True)
        gt = load_ground_truth(SPLITS / f"train_GT_{country}.tsv")
        splits, _, quotas = choose_sample(gt, country)
        train_ids = splits["train"]
        print(f"    train S1s: {len(train_ids)}", flush=True)

        # Build index
        idx = CompactBlockingIndex()
        cids: list[str] = []
        seen: set[str] = set()
        t0 = time.perf_counter()
        for source in ("S2", "S3"):
            for eid, name, address, ctry in stream_records(
                SPLITS / f"train_{source}_{country}.tsv"
            ):
                if eid in seen:
                    continue
                seen.add(eid)
                idx.add(len(cids), normalize_record(eid, name, address, ctry))
                cids.append(eid)
                if len(cids) % 500_000 == 0:
                    print(f"    indexed {len(cids):,} ({time.perf_counter()-t0:.1f}s)",
                          flush=True)
        del seen
        print(f"    index: {len(cids):,} candidates in {time.perf_counter()-t0:.1f}s",
              flush=True)

        # Load train S1 records
        s1_records: dict = {}
        for eid, name, address, ctry in stream_records(
            SPLITS / f"train_S1_{country}.tsv"
        ):
            if eid in train_ids:
                s1_records[eid] = normalize_record(eid, name, address, ctry)

        # Generate top-200 candidates using NumpyQueryIndex
        query_index = NumpyQueryIndex(idx, cids, list(s1_records.values()))
        print(f"    posting cache: {query_index.cache_seconds:.2f}s", flush=True)

        # Load candidate records needed for featurisation
        selected: dict[str, list] = {}
        for number, sid in enumerate(sorted(s1_records), 1):
            pool, touched = query_index.generate(s1_records[sid])
            top, _ = query_index.rank(pool)
            top_positions = [int(p) for p in top]
            rows = []
            for rank, pos in enumerate(top_positions, 1):
                blocker = tuple(int(v) for v in query_index.features[pos].tolist())
                score = int(np.asarray(blocker, dtype=np.int32) @ WEIGHTS)
                rows.append((cids[pos], blocker, score, rank))
            selected[sid] = rows
            query_index.clear(touched)
            if number % PROGRESS_EVERY == 0 or number == len(s1_records):
                print(f"    generated {number}/{len(s1_records)} S1s", flush=True)

        # Reload only needed candidate records
        needed = {cid for rows in selected.values() for cid, _, _, _ in rows}
        cand_records: dict = {}
        for source in ("S2", "S3"):
            for eid, name, address, ctry in stream_records(
                SPLITS / f"train_{source}_{country}.tsv"
            ):
                if eid in needed and eid not in cand_records:
                    cand_records[eid] = normalize_record(eid, name, address, ctry)

        # Featurise
        truth_by_sid = {sid: set(gt.get(sid, set())) for sid in train_ids}
        for sid in sorted(s1_records):
            truth = truth_by_sid[sid]
            for cid, blocker, score, rank in selected[sid]:
                vec = make_feature_vector(
                    s1_records[sid], cand_records[cid], blocker, rank
                )
                all_X.append(vec)
                all_y.append(int(cid in truth))

        del idx, query_index, selected, cand_records, s1_records
        gc.collect()

    Xtr = np.asarray(all_X, dtype=np.float32)
    ytr = np.asarray(all_y, dtype=np.int8)
    print(
        f"\n  Training matrix: {Xtr.shape}  "
        f"positives={int(ytr.sum()):,}  negatives={len(ytr)-int(ytr.sum()):,}",
        flush=True,
    )
    assert Xtr.shape[1] == len(FEATURES), f"Feature width mismatch: {Xtr.shape[1]}"
    assert len(np.unique(ytr)) == 2, "Training data missing positive or negative class"

    t0 = time.perf_counter()
    model, scaler = train_logistic_regression(Xtr, ytr, feature_names=FEATURES)
    print(f"  LR trained in {time.perf_counter()-t0:.2f}s", flush=True)
    return model, scaler


# ---------------------------------------------------------------------------
# Test inference
# ---------------------------------------------------------------------------

def discover_test_countries() -> list[str]:
    """Return sorted list of countries found in test_source1.tsv."""
    countries: set[str] = set()
    for _, _, _, ctry in stream_records(TEST_DIR / "test_source1.tsv"):
        if ctry:
            countries.add(ctry.lower())
    return sorted(countries)


def run_test_inference(model, scaler):
    """Stream test S1s country-by-country; write both output files."""
    print("\n=== TEST INFERENCE ===", flush=True)

    # Collect all test S1 IDs upfront to guarantee 100% coverage
    all_test_s1_ids: set[str] = set()
    s1_country: dict[str, str] = {}
    for eid, _, _, ctry in stream_records(TEST_DIR / "test_source1.tsv"):
        if eid:
            all_test_s1_ids.add(eid)
            s1_country[eid] = ctry.lower()
    print(f"  Total test S1 entities: {len(all_test_s1_ids):,}", flush=True)

    test_countries = sorted(set(s1_country.values()))
    print(f"  Countries: {test_countries}", flush=True)

    written_s1_ids: set[str] = set()
    cand_counts: list[int] = []
    n_singletons = 0
    n_with_match = 0
    total_matches = 0
    total_candidates = 0

    wall_inf = time.perf_counter()

    with open(MATCHING_PATH, "w", encoding="utf-8") as f_match, \
         open(CANDIDATE_PATH, "w", encoding="utf-8") as f_cand:

        f_match.write("source1_entity_id\tmatched_entity_ids\n")
        f_cand.write("source1_entity_id\tcandidate_entity_ids\n")

        for country in test_countries:
            print(f"\n  {country.upper()}: building candidate index", flush=True)
            s2_path = TEST_DIR / "test_source2.tsv"
            s3_path = TEST_DIR / "test_source3.tsv"

            # Build index filtered to this country
            idx = CompactBlockingIndex()
            cids: list[str] = []
            seen: set[str] = set()
            t0 = time.perf_counter()
            for path in (s2_path, s3_path):
                for eid, name, address, ctry in stream_records(path):
                    if ctry.lower() != country:
                        continue
                    if eid in seen:
                        continue
                    seen.add(eid)
                    idx.add(len(cids), normalize_record(eid, name, address, ctry))
                    cids.append(eid)
                    if len(cids) % 500_000 == 0:
                        print(f"    indexed {len(cids):,} ({time.perf_counter()-t0:.1f}s)",
                              flush=True)
            del seen
            print(
                f"    {len(cids):,} candidates in {time.perf_counter()-t0:.1f}s",
                flush=True,
            )

            # Load S1 records for this country
            s1_recs: dict[str, dict] = {}
            for eid, name, address, ctry in stream_records(
                TEST_DIR / "test_source1.tsv"
            ):
                if ctry.lower() == country:
                    s1_recs[eid] = normalize_record(eid, name, address, ctry)
            print(f"    {len(s1_recs):,} S1 entities", flush=True)

            if not s1_recs:
                del idx, cids
                gc.collect()
                continue

            # Build NumpyQueryIndex for this country's S1 batch
            query_index = NumpyQueryIndex(idx, cids, list(s1_recs.values()))
            print(f"    posting cache: {query_index.cache_seconds:.2f}s", flush=True)

            # Process in sub-batches to limit memory
            BATCH = 5_000
            sid_list = sorted(s1_recs)
            n_processed = 0

            for batch_start in range(0, len(sid_list), BATCH):
                batch_sids = sid_list[batch_start: batch_start + BATCH]

                # Generate candidates
                selected: dict[str, list[tuple]] = {}
                cand_id_sets: dict[str, set[str]] = {}
                for sid in batch_sids:
                    pool, touched = query_index.generate(s1_recs[sid])
                    top, _ = query_index.rank(pool)
                    top_positions = [int(p) for p in top]
                    pool_id_set = {cids[int(p)] for p in pool}
                    rows = []
                    for rank, pos in enumerate(top_positions, 1):
                        blocker = tuple(
                            int(v) for v in query_index.features[pos].tolist()
                        )
                        score = int(np.asarray(blocker, dtype=np.int32) @ WEIGHTS)
                        rows.append((cids[pos], blocker, score, rank))
                    selected[sid] = rows
                    cand_id_sets[sid] = {r[0] for r in rows}
                    query_index.clear(touched)

                # Reload candidate records for featurisation
                needed = {
                    cid for rows in selected.values() for cid, _, _, _ in rows
                }
                cand_records: dict[str, dict] = {}
                for path in (s2_path, s3_path):
                    for eid, name, address, ctry in stream_records(path):
                        if eid in needed and eid not in cand_records:
                            cand_records[eid] = normalize_record(
                                eid, name, address, ctry
                            )

                # Featurise
                X_rows: list = []
                row_meta: list[tuple[str, str]] = []  # (sid, cid)
                for sid in batch_sids:
                    for cid, blocker, score, rank in selected[sid]:
                        if cid not in cand_records:
                            continue
                        vec = make_feature_vector(
                            s1_recs[sid], cand_records[cid], blocker, rank
                        )
                        X_rows.append(vec)
                        row_meta.append((sid, cid))

                # Predict
                if X_rows:
                    X = np.asarray(X_rows, dtype=np.float32)
                    probs = predict_proba(model, X, scaler=scaler)
                else:
                    probs = np.array([], dtype=np.float32)

                # Apply frozen threshold and write output
                prob_by_pair: dict[tuple[str, str], float] = {
                    (sid, cid): float(p)
                    for (sid, cid), p in zip(row_meta, probs)
                }
                for sid in batch_sids:
                    cands = sorted(cand_id_sets[sid])
                    matched = sorted(
                        cid for cid in cands
                        if prob_by_pair.get((sid, cid), 0.0) >= FROZEN_THRESHOLD
                    )
                    f_match.write(f"{sid}\t{','.join(matched)}\n")
                    f_cand.write(f"{sid}\t{','.join(cands)}\n")
                    written_s1_ids.add(sid)
                    cand_counts.append(len(cands))
                    total_candidates += len(cands)
                    total_matches += len(matched)
                    if matched:
                        n_with_match += 1
                    else:
                        n_singletons += 1

                n_processed += len(batch_sids)
                if n_processed % (BATCH * 4) == 0 or n_processed == len(sid_list):
                    print(
                        f"    {country.upper()}: {n_processed:,}/{len(sid_list):,} S1s",
                        flush=True,
                    )
                del selected, cand_records, X_rows, row_meta
                gc.collect()

            del idx, query_index, s1_recs, cids
            gc.collect()

        # Write any S1 IDs not yet written (should be zero, but safety net)
        missing = all_test_s1_ids - written_s1_ids
        if missing:
            print(f"  Writing {len(missing):,} missing S1s as singletons", flush=True)
            for sid in sorted(missing):
                f_match.write(f"{sid}\t\n")
                f_cand.write(f"{sid}\t\n")
                written_s1_ids.add(sid)
                cand_counts.append(0)
                n_singletons += 1

    inf_seconds = time.perf_counter() - wall_inf

    # Statistics
    cand_arr = np.array(cand_counts, dtype=np.int32)
    print(f"\n=== TEST INFERENCE COMPLETE ===")
    print(f"  Test S1 entities processed : {len(written_s1_ids):,}")
    print(f"  candidate_pairs.tsv rows   : {len(written_s1_ids):,}")
    print(f"  matching_results.tsv rows  : {len(written_s1_ids):,}")
    print(f"  Predicted singletons       : {n_singletons:,}")
    print(f"  S1s with >= 1 match        : {n_with_match:,}")
    print(f"  Total candidate pairs      : {total_candidates:,}")
    print(f"  Total predicted matches    : {total_matches:,}")
    print(f"  Candidate count stats      : "
          f"min={cand_arr.min()}, "
          f"mean={cand_arr.mean():.1f}, "
          f"median={int(np.median(cand_arr))}, "
          f"max={cand_arr.max()}")
    print(f"  Countries processed        : {len(test_countries)}  {test_countries}")
    print(f"  Inference wall time        : {inf_seconds:.1f}s")
    print(f"  matching_results.tsv       : {MATCHING_PATH}")
    print(f"  candidate_pairs.tsv        : {CANDIDATE_PATH}")

    return inf_seconds


# ---------------------------------------------------------------------------
# DRY-RUN inspection
# ---------------------------------------------------------------------------

def dry_run():
    print("=== DRY-RUN INSPECTION ===", flush=True)
    print(f"  Script                  : {Path(__file__).name}")
    print(f"  test_source1.tsv        : {TEST_DIR / 'test_source1.tsv'}")
    print(f"  test_source2.tsv        : {TEST_DIR / 'test_source2.tsv'}")
    print(f"  test_source3.tsv        : {TEST_DIR / 'test_source3.tsv'}")
    print(f"  Output matching         : {MATCHING_PATH}")
    print(f"  Output candidates       : {CANDIDATE_PATH}")
    print(f"  Frozen CAP              : {CAP}  (must be 200)")
    print(f"  Frozen threshold        : {FROZEN_THRESHOLD}")
    print(f"  Model                   : class-weighted Logistic Regression")
    print(f"  Features                : {len(FEATURES)} (27 pairwise + 13 blocker/rank)")
    print(f"  GT loaded during test   : NO")
    assert CAP == 200, f"CAP is {CAP}, expected 200"
    assert FROZEN_THRESHOLD == 0.998722, f"Threshold is {FROZEN_THRESHOLD}"

    # Count test records and countries
    for label, fname in [("S1", "test_source1.tsv"),
                          ("S2", "test_source2.tsv"),
                          ("S3", "test_source3.tsv")]:
        import collections
        counts: dict = collections.Counter()
        n = 0
        for _, _, _, ctry in stream_records(TEST_DIR / fname):
            counts[ctry.lower()] += 1
            n += 1
        print(f"\n  {fname}:")
        print(f"    Total records : {n:,}")
        for ctry, cnt in sorted(counts.items()):
            print(f"    {ctry:<12}: {cnt:,}")

    france_in_s1 = any(
        ctry.lower() == "france"
        for _, _, _, ctry in stream_records(TEST_DIR / "test_source1.tsv")
    )
    print(f"\n  France in test S1       : {france_in_s1}  (open-set: handled)")
    print(f"  CAP=200 confirmed       : {CAP == 200}")
    print(f"  Threshold confirmed     : {FROZEN_THRESHOLD == 0.998722}")
    print(f"  GT not loaded           : confirmed (no load_ground_truth call in inference)")
    print("\nDRY-RUN COMPLETE — proceeding to full test inference.\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    wall_start = time.perf_counter()

    dry_run()

    model, scaler = train_frozen_model()

    inf_seconds = run_test_inference(model, scaler)

    total_seconds = time.perf_counter() - wall_start
    print(f"\n  Total wall time (train + inference): {total_seconds:.1f}s")

    # Run official validator
    print("\n=== RUNNING OFFICIAL VALIDATOR ===", flush=True)
    if VALIDATOR.exists():
        import subprocess
        result = subprocess.run(
            [sys.executable, str(VALIDATOR),
             "--matching", str(MATCHING_PATH),
             "--candidate", str(CANDIDATE_PATH),
             "--test-dir", str(TEST_DIR)],
            capture_output=False,
        )
        print(f"  Validator exit code: {result.returncode}", flush=True)
    else:
        print(f"  Validator not found at {VALIDATOR}", flush=True)

    print("\nScope: test inference only; no submission; production unchanged.")


if __name__ == "__main__":
    main()
