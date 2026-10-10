#!/usr/bin/env python3
"""Small diagnostic for candidate ordering and the production 200-candidate cap."""
import sys
import time
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
INDEX_PROGRESS_EVERY = 500_000
CAP = 200


def matching_keys(s1, match):
    """Return the CompactBlockingIndex strategies that should admit a match."""
    keys = []
    if s1.get("ft") and s1["ft"] == match.get("ft"):
        keys.append("first_token")
    if s1.get("npfx") and s1["npfx"] == match.get("npfx"):
        keys.append("name_prefix")
    if s1.get("ntoks", set()) & match.get("ntoks", set()):
        keys.append("name_token")
    if set(s1.get("anums", [])) & set(match.get("anums", [])):
        keys.append("address_numeric")
    if len(s1.get("atoks", set()) & match.get("atoks", set())) >= 2:
        keys.append("address_tokens>=2")
    cs3 = s1.get("cskel3", "")
    if cs3 and len(cs3) >= 3 and cs3 == match.get("cskel3", ""):
        keys.append("consonant_skeleton")
    if len(s1.get("cngrams", set()) & match.get("cngrams", set())) >= 2:
        keys.append("consonant_ngrams>=2")
    return keys


def diagnose_country(country):
    s2_path = SPLIT_DIR / f"train_S2_{country}.tsv"
    s3_path = SPLIT_DIR / f"train_S3_{country}.tsv"
    s1_path = SPLIT_DIR / f"train_S1_{country}.tsv"
    gt_path = SPLIT_DIR / f"train_GT_{country}.tsv"
    for path in (s1_path, s2_path, s3_path, gt_path):
        if not path.exists():
            raise FileNotFoundError(path)

    gt = load_ground_truth(gt_path)
    gt_with_matches = {sid: mids for sid, mids in gt.items() if mids}
    s1_recs = []
    for eid, name, addr, ctry in stream_records(s1_path):
        if eid in gt_with_matches:
            s1_recs.append(normalize_record(eid, name, addr, ctry))
            if len(s1_recs) >= SAMPLE_SIZE:
                break
    wanted_match_ids = set()
    for rec in s1_recs:
        wanted_match_ids.update(gt_with_matches.get(rec["id"], set()))

    print(f"\n{country.upper()}: sampling {len(s1_recs)} S1s", flush=True)
    index_start = time.perf_counter()
    idx = CompactBlockingIndex()
    cand_ids = []
    seen_ids = set()
    match_positions = {}
    match_records = {}
    for path in (s2_path, s3_path):
        for eid, name, addr, ctry in stream_records(path):
            if eid in seen_ids:
                continue
            seen_ids.add(eid)
            rec = normalize_record(eid, name, addr, ctry)
            pos = len(cand_ids)
            cand_ids.append(eid)
            idx.add(pos, rec)
            if eid in wanted_match_ids:
                match_positions[eid] = pos
                match_records[eid] = rec
            if len(cand_ids) % INDEX_PROGRESS_EVERY == 0:
                print(f"  Indexed {len(cand_ids):,} ({time.perf_counter()-index_start:.1f}s)", flush=True)
    print(f"  Indexed {len(cand_ids):,} candidates in {time.perf_counter()-index_start:.1f}s", flush=True)

    total_true = 0
    capped_found = 0
    uncapped_found = 0
    cap_misses = 0
    uncapped_misses = 0
    parity_mismatches = 0
    print("  COUNTRY\tS1_ID\tTRUE_MATCH_ID\tUNCAPPED_RANK\tUNCAPPED_COUNT\tIN_CAPPED_200\tBLOCKING_KEYS", flush=True)

    for s1_i, s1_rec in enumerate(s1_recs, 1):
        true_ids = gt_with_matches.get(s1_rec["id"], set())
        raw = idx.query(s1_rec, max_cands=len(cand_ids) + 1)
        raw_order = list(raw)
        capped_from_raw = set(raw_order[:CAP])
        production_capped = idx.query(s1_rec, max_cands=CAP)
        if production_capped != capped_from_raw:
            parity_mismatches += 1

        target_positions = {
            match_positions[mid]: mid
            for mid in true_ids
            if mid in match_positions
        }
        target_ranks = {}
        for rank, pos in enumerate(raw_order, 1):
            if pos in target_positions:
                target_ranks[pos] = rank

        for mid in sorted(true_ids):
            total_true += 1
            pos = match_positions.get(mid)
            rank = target_ranks.get(pos) if pos is not None else None
            in_uncapped = rank is not None
            in_capped = pos is not None and pos in production_capped
            if in_uncapped:
                uncapped_found += 1
            else:
                uncapped_misses += 1
            if in_capped:
                capped_found += 1
            else:
                if in_uncapped:
                    cap_misses += 1
                match_rec = match_records.get(mid)
                keys = matching_keys(s1_rec, match_rec) if match_rec else []
                rank_text = str(rank) if rank is not None else "NOT_IN_CANDIDATES"
                print(f"  {country.upper()}\t{s1_rec['id']}\t{mid}\t{rank_text}\t{len(raw)}\tNO\t{','.join(keys) or 'NONE'}", flush=True)

        if s1_i % 5 == 0 or s1_i == len(s1_recs):
            print(f"  Evaluated {country.upper()}: {s1_i}/{len(s1_recs)}; "
                  f"uncapped={len(raw):,} candidates for latest S1", flush=True)

    print(f"  SUMMARY {country.upper()}: S1={len(s1_recs)}, true_matches={total_true}, "
          f"capped_found={capped_found}, uncapped_found={uncapped_found}, "
          f"cap_misses={cap_misses}, uncapped_misses={uncapped_misses}, "
          f"production_slice_parity_mismatches={parity_mismatches}", flush=True)


if __name__ == "__main__":
    for country_name in ("india", "us"):
        diagnose_country(country_name)
