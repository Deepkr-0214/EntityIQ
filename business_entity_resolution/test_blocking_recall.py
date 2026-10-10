"""Test blocking recall with transliteration + consonant skeleton blocking."""
import sys, os, time
if sys.platform == 'win32':
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path('d:/Amazon/business_entity_resolution')
sys.path.insert(0, str(PROJECT_ROOT))
from src.preprocessing.normalization import normalize_record
from src.pipeline.run_pipeline import (
    CompactBlockingIndex, stream_records, load_ground_truth
)

SPLIT_DIR = PROJECT_ROOT / 'splits'

for ctry in ['india', 'us']:
    print(f"\n{'='*60}")
    print(f"  Testing blocking recall for {ctry.upper()}")
    print(f"{'='*60}")

    s2_path = SPLIT_DIR / f"train_S2_{ctry}.tsv"
    s3_path = SPLIT_DIR / f"train_S3_{ctry}.tsv"
    s1_path = SPLIT_DIR / f"train_S1_{ctry}.tsv"
    gt_path = SPLIT_DIR / f"train_GT_{ctry}.tsv"

    # Build candidate index
    print("  Building candidate index from S2+S3...", end=" ", flush=True)
    t0 = time.time()
    cand_records = []
    cand_id_map = {}
    idx = CompactBlockingIndex()
    for path in (s2_path, s3_path):
        if not path.exists():
            continue
        for eid, name, addr, c in stream_records(path):
            if eid in cand_id_map:
                continue
            rec = normalize_record(eid, name, addr, c)
            pos = len(cand_records)
            cand_records.append(rec)
            cand_id_map[eid] = pos
            idx.add(pos, rec)
    print(f"{len(cand_records):,} candidates in {time.time()-t0:.1f}s")

    # Load GT
    gt = load_ground_truth(gt_path)
    gt_with_matches = {k: v for k, v in gt.items() if v}

    # Sample S1 entities
    s1_recs = []
    count = 0
    for eid, name, addr, c in stream_records(s1_path):
        if eid in gt_with_matches:
            s1_recs.append(normalize_record(eid, name, addr, c))
            count += 1
            if count >= 3000:
                break

    # Test blocking recall
    print(f"  Testing {len(s1_recs)} S1 entities with known matches...")
    t0 = time.time()
    total = found = 0
    missed_examples = []
    for s1_rec in s1_recs:
        s1_id = s1_rec['id']
        true_matches = gt_with_matches.get(s1_id, set())
        cand_int_indices = idx.query(s1_rec, max_cands=200)
        cand_str_ids = {cand_records[i]['id'] for i in cand_int_indices}
        for m in true_matches:
            total += 1
            if m in cand_str_ids:
                found += 1
            elif len(missed_examples) < 5:
                missed_examples.append((s1_id, m))

    recall = found / total if total else 1.0
    elapsed = time.time() - t0
    avg_cands = sum(len(idx.query(r, max_cands=200)) for r in s1_recs[:500]) / min(500, len(s1_recs))
    print(f"  Blocking Recall: {recall:.4f} ({found}/{total})")
    print(f"  Time: {elapsed:.1f}s, Avg candidates/S1: {avg_cands:.0f}")
    if missed_examples:
        print(f"  Missed examples:")
        for s1_id, mid in missed_examples:
            s1r = next((r for r in s1_recs if r['id'] == s1_id), None)
            if mid in cand_id_map:
                mr = cand_records[cand_id_map[mid]]
                print(f"    {s1_id}: '{s1r['na'][:40]}' <-> {mid}: '{mr['na'][:40]}'")

    del cand_records, cand_id_map, idx
    import gc; gc.collect()
