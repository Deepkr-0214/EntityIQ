#!/usr/bin/env python3
"""
Comprehensive Blocking Recall Validation
=========================================
Produces a full diagnostic report covering:
  1. Per-country blocking recall (India, US)
  2. Combined recall
  3. Candidate count statistics (avg, median, max, zero-candidate S1s)
  4. True-match loss analysis (why matches were missed)
  5. max_cands=200 impact analysis (does capping drop true matches?)

Run from the project root:
    python validate_blocking_recall.py
"""
import sys, os, time, gc, statistics
from pathlib import Path
from collections import defaultdict

# Force UTF-8 on Windows
if sys.platform == 'win32':
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

PROJECT_ROOT = Path('d:/Amazon/business_entity_resolution')
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.normalization import normalize_record
from src.pipeline.run_pipeline import (
    CompactBlockingIndex, stream_records, load_ground_truth
)

SPLIT_DIR = PROJECT_ROOT / 'splits'
SAMPLE_SIZE = 3000  # S1 entities per country to test


def safe_print(text):
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode('ascii', errors='replace').decode('ascii'))


def analyze_miss_reason(s1_rec, match_rec):
    """Categorize why a true match was missed by all blocking strategies."""
    reasons = []

    # Name checks
    ft_match = s1_rec['ft'] == match_rec['ft']
    npfx_match = s1_rec['npfx'] == match_rec['npfx']
    tok_overlap = s1_rec['ntoks'] & match_rec['ntoks']
    cskel3_match = (s1_rec.get('cskel3', '') == match_rec.get('cskel3', '') and
                    len(s1_rec.get('cskel3', '')) >= 3)

    # Consonant n-gram overlap
    cng_s1 = s1_rec.get('cngrams', set())
    cng_m = match_rec.get('cngrams', set())
    cng_overlap = cng_s1 & cng_m if isinstance(cng_s1, set) and isinstance(cng_m, set) else set()

    # Address checks
    atoks_overlap = s1_rec.get('atoks', set()) & match_rec.get('atoks', set())
    anums_s1 = set(s1_rec.get('anums', []))
    anums_m = set(match_rec.get('anums', []))
    anum_overlap = anums_s1 & anums_m

    # Determine what blocked us
    if not ft_match:
        reasons.append('first_token_mismatch')
    if not npfx_match:
        reasons.append('name_prefix_mismatch')
    if not tok_overlap:
        reasons.append('no_shared_name_token')
    if not cskel3_match:
        reasons.append('consonant_skeleton_mismatch')
    if len(cng_overlap) < 2:
        reasons.append('insufficient_consonant_ngrams')
    if len(atoks_overlap) < 2:
        reasons.append('insufficient_addr_token_overlap')
    if not anum_overlap:
        reasons.append('no_shared_address_numeric')

    # Higher-level categorization
    categories = []
    s1_name = s1_rec.get('na', '')
    m_name = match_rec.get('na', '')

    # Check if names are very different (possible trade/DBA name issue)
    if not tok_overlap and not ft_match:
        categories.append('COMPLETELY_DIFFERENT_NAME')
    elif not ft_match and tok_overlap:
        categories.append('REORDERED_NAME')
    elif ft_match and not tok_overlap:
        categories.append('PARTIAL_NAME_MATCH')

    # Check for transliteration issues
    s1_orig_name = s1_rec.get('nl', '')
    m_orig_name = match_rec.get('nl', '')
    if s1_orig_name and m_orig_name:
        # Check if one might be transliterated from another script
        s1_has_translit = any(ord(c) > 0x024F for c in s1_orig_name)
        m_has_translit = any(ord(c) > 0x024F for c in m_orig_name)
        if s1_has_translit or m_has_translit:
            categories.append('POSSIBLE_TRANSLITERATION')

    # Check for domain name issue
    if '.' in s1_name or '.' in m_name:
        categories.append('DOMAIN_NAME')

    if not categories:
        if not tok_overlap and len(atoks_overlap) < 2 and not anum_overlap:
            categories.append('NO_OVERLAP_AT_ALL')
        else:
            categories.append('MARGINAL_OVERLAP')

    return reasons, categories


def run_country_validation(country, verbose=True):
    """Run full blocking recall validation for a single country."""
    s2_path = SPLIT_DIR / f"train_S2_{country}.tsv"
    s3_path = SPLIT_DIR / f"train_S3_{country}.tsv"
    s1_path = SPLIT_DIR / f"train_S1_{country}.tsv"
    gt_path = SPLIT_DIR / f"train_GT_{country}.tsv"

    for p in (s2_path, s3_path, s1_path, gt_path):
        if not p.exists():
            print(f"  WARNING: {p} not found, skipping {country}")
            return None

    # Build candidate index
    safe_print(f"\n  Building candidate index from S2+S3 for {country.upper()}...")
    t0 = time.time()
    cand_records = []
    cand_id_map = {}
    idx = CompactBlockingIndex()
    for path in (s2_path, s3_path):
        for eid, name, addr, c in stream_records(path):
            if eid in cand_id_map:
                continue
            rec = normalize_record(eid, name, addr, c)
            pos = len(cand_records)
            cand_records.append(rec)
            cand_id_map[eid] = pos
            idx.add(pos, rec)
    safe_print(f"  {len(cand_records):,} candidates indexed in {time.time()-t0:.1f}s")

    # Load ground truth
    gt = load_ground_truth(gt_path)
    gt_with_matches = {k: v for k, v in gt.items() if v}

    # Sample S1 entities (those with known matches)
    s1_recs = []
    count = 0
    for eid, name, addr, c in stream_records(s1_path):
        if eid in gt_with_matches:
            s1_recs.append(normalize_record(eid, name, addr, c))
            count += 1
            if count >= SAMPLE_SIZE:
                break

    safe_print(f"  Sampled {len(s1_recs)} S1 entities with known matches")

    # --- Test blocking recall ---
    t0 = time.time()
    total = 0
    found = 0
    missed_details = []
    cand_counts = []
    cand_counts_uncapped = []
    zero_cand_count = 0
    capped_count = 0
    capped_true_match_lost = 0

    for s1_rec in s1_recs:
        s1_id = s1_rec['id']
        true_matches = gt_with_matches.get(s1_id, set())

        # Query with max_cands=200 (the production setting)
        cand_int_indices_capped = idx.query(s1_rec, max_cands=200)
        cand_str_ids_capped = {cand_records[i]['id'] for i in cand_int_indices_capped}

        # Also query UNCAPPED to measure max_cands=200 impact
        cand_int_indices_uncapped = idx.query(s1_rec, max_cands=999_999)
        cand_str_ids_uncapped = {cand_records[i]['id'] for i in cand_int_indices_uncapped}

        cand_counts.append(len(cand_str_ids_capped))
        cand_counts_uncapped.append(len(cand_str_ids_uncapped))

        if len(cand_str_ids_capped) == 0:
            zero_cand_count += 1

        if len(cand_str_ids_uncapped) > 200:
            capped_count += 1

        for m in true_matches:
            total += 1
            if m in cand_str_ids_capped:
                found += 1
            else:
                # Was this match findable uncapped?
                found_uncapped = m in cand_str_ids_uncapped

                if found_uncapped:
                    capped_true_match_lost += 1

                # Get match record for analysis
                if m in cand_id_map:
                    match_rec = cand_records[cand_id_map[m]]
                    reasons, categories = analyze_miss_reason(s1_rec, match_rec)
                    missed_details.append({
                        's1_id': s1_id,
                        'match_id': m,
                        's1_name': s1_rec.get('na', '')[:60],
                        'match_name': match_rec.get('na', '')[:60],
                        's1_name_light': s1_rec.get('nl', '')[:60],
                        'match_name_light': match_rec.get('nl', '')[:60],
                        's1_addr': s1_rec.get('aa', '')[:50],
                        'match_addr': match_rec.get('aa', '')[:50],
                        's1_ft': s1_rec.get('ft', ''),
                        'match_ft': match_rec.get('ft', ''),
                        's1_npfx': s1_rec.get('npfx', ''),
                        'match_npfx': match_rec.get('npfx', ''),
                        's1_ntoks': s1_rec.get('ntoks', set()),
                        'match_ntoks': match_rec.get('ntoks', set()),
                        'reasons': reasons,
                        'categories': categories,
                        'found_uncapped': found_uncapped,
                    })
                else:
                    missed_details.append({
                        's1_id': s1_id,
                        'match_id': m,
                        's1_name': s1_rec.get('na', '')[:60],
                        'match_name': '(NOT FOUND IN S2/S3)',
                        'reasons': ['match_id_not_in_candidate_pool'],
                        'categories': ['MISSING_FROM_S2_S3'],
                        'found_uncapped': False,
                    })

    elapsed = time.time() - t0
    recall = found / total if total else 1.0

    result = {
        'country': country,
        'sampled': len(s1_recs),
        'total_true_matches': total,
        'found': found,
        'recall': recall,
        'elapsed': elapsed,
        'cand_counts': cand_counts,
        'cand_counts_uncapped': cand_counts_uncapped,
        'zero_cand_count': zero_cand_count,
        'capped_count': capped_count,
        'capped_true_match_lost': capped_true_match_lost,
        'missed_details': missed_details,
    }

    # Cleanup
    del cand_records, cand_id_map, idx
    gc.collect()

    return result


def print_report(results):
    """Print the comprehensive blocking recall report."""
    print("\n" + "=" * 70)
    print("  BLOCKING RECALL VALIDATION REPORT")
    print("=" * 70)

    combined_total = 0
    combined_found = 0
    all_cand_counts = []
    all_capped_loss = 0
    all_missed = []

    for r in results:
        if r is None:
            continue
        ctry = r['country'].upper()
        print(f"\n{'─' * 60}")
        print(f"  {ctry}")
        print(f"{'─' * 60}")
        print(f"  Sampled:       {r['sampled']:,}")
        print(f"  True matches:  {r['total_true_matches']:,}")
        print(f"  Recovered:     {r['found']:,}")
        print(f"  Recall:        {r['recall']*100:.2f}%")
        print(f"  Time:          {r['elapsed']:.1f}s")

        combined_total += r['total_true_matches']
        combined_found += r['found']
        all_cand_counts.extend(r['cand_counts'])
        all_capped_loss += r['capped_true_match_lost']
        all_missed.extend(r['missed_details'])

    # Combined recall
    combined_recall = combined_found / combined_total if combined_total else 1.0
    print(f"\n{'─' * 60}")
    print(f"  COMBINED")
    print(f"{'─' * 60}")
    print(f"  Total true matches:  {combined_total:,}")
    print(f"  Recovered:           {combined_found:,}")
    print(f"  Combined Recall:     {combined_recall*100:.2f}%")

    # Candidate count statistics
    print(f"\n{'─' * 60}")
    print(f"  CANDIDATE COUNT STATISTICS")
    print(f"{'─' * 60}")
    if all_cand_counts:
        avg_cands = statistics.mean(all_cand_counts)
        med_cands = statistics.median(all_cand_counts)
        max_cands = max(all_cand_counts)
        min_cands = min(all_cand_counts)
        zero_total = sum(1 for c in all_cand_counts if c == 0)
        p90 = sorted(all_cand_counts)[int(0.9 * len(all_cand_counts))]
        p95 = sorted(all_cand_counts)[int(0.95 * len(all_cand_counts))]
        p99 = sorted(all_cand_counts)[min(int(0.99 * len(all_cand_counts)), len(all_cand_counts)-1)]

        print(f"  Average candidates/S1:  {avg_cands:.1f}")
        print(f"  Median:                 {med_cands:.0f}")
        print(f"  P90:                    {p90}")
        print(f"  P95:                    {p95}")
        print(f"  P99:                    {p99}")
        print(f"  Max:                    {max_cands}")
        print(f"  Min:                    {min_cands}")
        print(f"  Zero-candidate S1:      {zero_total}")

        # Distribution buckets
        buckets = [0, 10, 50, 100, 200, 500, 1000, 5000, 999999]
        bucket_labels = ['0', '1-10', '11-50', '51-100', '101-200', '201-500', '501-1000', '1001+']
        print(f"\n  Candidate count distribution:")
        for i in range(len(bucket_labels)):
            lo = buckets[i]
            hi = buckets[i + 1]
            cnt = sum(1 for c in all_cand_counts if lo <= c < hi) if i > 0 else sum(1 for c in all_cand_counts if c == 0)
            if i > 0:
                cnt = sum(1 for c in all_cand_counts if buckets[i-1] < c <= buckets[i]) if i < len(bucket_labels) - 1 else sum(1 for c in all_cand_counts if c > buckets[i-1])
                # Recalculate properly
            pct = cnt / len(all_cand_counts) * 100 if all_cand_counts else 0
            print(f"    {bucket_labels[i]:>8s}: {cnt:>6,} ({pct:5.1f}%)")

    # max_cands=200 impact
    print(f"\n{'─' * 60}")
    print(f"  max_cands=200 IMPACT ANALYSIS")
    print(f"{'─' * 60}")
    total_capped = sum(r['capped_count'] for r in results if r)
    print(f"  S1 entities with >200 raw candidates: {total_capped:,}")
    print(f"  True matches lost due to capping:     {all_capped_loss:,}")
    if combined_total > 0:
        print(f"  Recall loss from capping:             {all_capped_loss/combined_total*100:.3f}%")
    # Uncapped stats
    all_uncapped = []
    for r in results:
        if r:
            all_uncapped.extend(r['cand_counts_uncapped'])
    if all_uncapped:
        print(f"  Avg candidates (uncapped):            {statistics.mean(all_uncapped):.1f}")
        print(f"  Max candidates (uncapped):            {max(all_uncapped)}")

    # True-match loss analysis
    print(f"\n{'─' * 60}")
    print(f"  TRUE-MATCH LOSS ANALYSIS")
    print(f"{'─' * 60}")
    print(f"  Total missed matches: {len(all_missed)}")

    # Categorize misses
    category_counts = defaultdict(int)
    reason_counts = defaultdict(int)
    for m in all_missed:
        for cat in m.get('categories', []):
            category_counts[cat] += 1
        for r in m.get('reasons', []):
            reason_counts[r] += 1

    print(f"\n  Miss categories:")
    for cat, cnt in sorted(category_counts.items(), key=lambda x: -x[1]):
        pct = cnt / len(all_missed) * 100 if all_missed else 0
        print(f"    {cat:40s} {cnt:>5} ({pct:5.1f}%)")

    print(f"\n  Blocking key failures (reason counts):")
    for reason, cnt in sorted(reason_counts.items(), key=lambda x: -x[1]):
        pct = cnt / len(all_missed) * 100 if all_missed else 0
        print(f"    {reason:40s} {cnt:>5} ({pct:5.1f}%)")

    # Show detailed examples of missed matches
    print(f"\n  Detailed missed match examples (up to 20):")
    for i, m in enumerate(all_missed[:20]):
        print(f"\n  [{i+1}] {m['s1_id']} <-> {m['match_id']}")
        safe_print(f"      S1 name (agg):  '{m.get('s1_name', '')}'")
        safe_print(f"      Match name:     '{m.get('match_name', '')}'")
        if 's1_name_light' in m:
            safe_print(f"      S1 name (light):'{m.get('s1_name_light', '')}'")
            safe_print(f"      Match (light):  '{m.get('match_name_light', '')}'")
        if 's1_ft' in m:
            safe_print(f"      S1 ft='{m.get('s1_ft','')}', Match ft='{m.get('match_ft','')}'")
            safe_print(f"      S1 npfx='{m.get('s1_npfx','')}', Match npfx='{m.get('match_npfx','')}'")
        if 's1_ntoks' in m:
            safe_print(f"      S1 ntoks={m.get('s1_ntoks', set())}")
            safe_print(f"      Match ntoks={m.get('match_ntoks', set())}")
        if 's1_addr' in m:
            safe_print(f"      S1 addr: '{m.get('s1_addr', '')}'")
            safe_print(f"      Match addr: '{m.get('match_addr', '')}'")
        safe_print(f"      Categories: {m.get('categories', [])}")
        safe_print(f"      Reasons:    {m.get('reasons', [])}")
        safe_print(f"      Found uncapped: {m.get('found_uncapped', False)}")


    # Final summary in the exact format requested
    print("\n" + "=" * 70)
    print("  SUMMARY (copy-paste format)")
    print("=" * 70)
    for r in results:
        if r is None:
            continue
        print(f"\n{r['country'].upper()}:")
        print(f"  Sampled: {r['sampled']:,}")
        print(f"  True matches: {r['total_true_matches']:,}")
        print(f"  Recovered: {r['found']:,}")
        print(f"  Recall: {r['recall']*100:.2f}%")

    print(f"\nCombined recall: {combined_recall*100:.2f}%")
    if all_cand_counts:
        print(f"\nAverage candidates/S1: {statistics.mean(all_cand_counts):.1f}")
        print(f"Median: {statistics.median(all_cand_counts):.0f}")
        print(f"Max: {max(all_cand_counts)}")
        print(f"Zero-candidate S1: {sum(1 for c in all_cand_counts if c == 0)}")


if __name__ == '__main__':
    print("=" * 70)
    print("  BLOCKING RECALL VALIDATION")
    print(f"  Sample size: {SAMPLE_SIZE} S1 entities per country")
    print(f"  Production max_cands: 200")
    print("=" * 70)

    results = []
    for country in ['india', 'us']:
        result = run_country_validation(country)
        results.append(result)
        gc.collect()

    print_report(results)
