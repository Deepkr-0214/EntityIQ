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
    python -u validate_blocking_recall_optimized.py
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

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.normalization import normalize_record
from src.pipeline.run_pipeline import (
    CompactBlockingIndex, stream_records, load_ground_truth
)

SPLIT_DIR = PROJECT_ROOT / 'splits'
SAMPLE_SIZE = 100  # Start small; increase only after reviewing benchmark results.
RUN_UNCAPPED = True  # Required for uncapped counts and cap-induced loss analysis.
MAX_CANDS = 200
PROGRESS_EVERY = 25
INDEX_PROGRESS_EVERY = 500_000
BASELINE_BENCHMARK_SAMPLES = 10  # Compares old two-query path to shared-query path.


def safe_print(text, flush=False):
    try:
        print(text, flush=flush)
    except UnicodeEncodeError:
        print(text.encode('ascii', errors='replace').decode('ascii'), flush=flush)


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
    """Validate one country, sharing one full query between capped/uncapped metrics."""
    s2_path = SPLIT_DIR / f"train_S2_{country}.tsv"
    s3_path = SPLIT_DIR / f"train_S3_{country}.tsv"
    s1_path = SPLIT_DIR / f"train_S1_{country}.tsv"
    gt_path = SPLIT_DIR / f"train_GT_{country}.tsv"

    for p in (s2_path, s3_path, s1_path, gt_path):
        if not p.exists():
            print(f"  WARNING: {p} not found, skipping {country}")
            return None

    # Load GT and sample first so we retain full normalized records only for
    # true-match candidates needed by miss diagnostics, rather than all records.
    safe_print(f"\n  Loading ground truth and sampling {SAMPLE_SIZE} S1s for {country.upper()}...", flush=True)
    setup_t0 = time.perf_counter()
    gt = load_ground_truth(gt_path)
    gt_with_matches = {k: v for k, v in gt.items() if v}
    s1_recs = []
    for eid, name, addr, c in stream_records(s1_path):
        if eid in gt_with_matches:
            s1_recs.append(normalize_record(eid, name, addr, c))
            if len(s1_recs) >= SAMPLE_SIZE:
                break

    sampled_true_ids = set()
    for rec in s1_recs:
        sampled_true_ids.update(gt_with_matches.get(rec['id'], set()))
    safe_print(f"  Sampled {len(s1_recs)} S1 entities with known matches ({time.perf_counter()-setup_t0:.1f}s)", flush=True)

    # Build the same production index, while keeping only candidate IDs for
    # every record and diagnostic details for sampled ground-truth IDs.
    safe_print(f"  Building candidate index from S2+S3 for {country.upper()}...", flush=True)
    index_t0 = time.perf_counter()
    cand_ids = []
    true_match_positions = {}
    true_match_records = {}
    seen_ids = set()
    idx = CompactBlockingIndex()
    indexed = 0
    for path in (s2_path, s3_path):
        for eid, name, addr, c in stream_records(path):
            if eid in seen_ids:
                continue
            seen_ids.add(eid)
            rec = normalize_record(eid, name, addr, c)
            pos = len(cand_ids)
            cand_ids.append(eid)
            idx.add(pos, rec)
            if eid in sampled_true_ids:
                true_match_positions[eid] = pos
                true_match_records[eid] = rec
            indexed += 1
            if verbose and indexed % INDEX_PROGRESS_EVERY == 0:
                safe_print(f"    Indexed {indexed:,} candidate records ({time.perf_counter()-index_t0:.1f}s)", flush=True)
            del rec
    index_seconds = time.perf_counter() - index_t0
    safe_print(f"  {len(cand_ids):,} candidates indexed in {index_seconds:.1f}s", flush=True)

    # --- Evaluate ---
    eval_t0 = time.perf_counter()
    total = 0
    found = 0
    uncapped_found = 0
    missed_details = []
    cand_counts = []
    cand_counts_uncapped = []
    zero_cand_count = 0
    capped_count = 0
    capped_true_match_lost = 0
    optimized_query_seconds = 0.0
    benchmark_query_seconds = 0.0
    old_capped_query_seconds = 0.0
    benchmark_comparisons = 0

    for sample_i, s1_rec in enumerate(s1_recs, 1):
        s1_id = s1_rec['id']
        true_matches = gt_with_matches.get(s1_id, set())
        query_t0 = time.perf_counter()
        if RUN_UNCAPPED:
            # query() performs all seven blocking lookups before truncation.
            # One full query therefore supplies both views; slicing the same
            # set uses the same ordering/truncation rule as production query.
            raw_indices = idx.query(s1_rec, max_cands=len(cand_ids) + 1)
            cand_int_indices_capped = set(list(raw_indices)[:MAX_CANDS])
            cand_count_uncapped = len(raw_indices)
            cand_counts_uncapped.append(cand_count_uncapped)
            if cand_count_uncapped > MAX_CANDS:
                capped_count += 1
        else:
            raw_indices = None
            cand_int_indices_capped = idx.query(s1_rec, max_cands=MAX_CANDS)
            cand_count_uncapped = None

        one_query_seconds = time.perf_counter() - query_t0
        optimized_query_seconds += one_query_seconds
        if RUN_UNCAPPED and sample_i <= BASELINE_BENCHMARK_SAMPLES:
            benchmark_query_seconds += one_query_seconds

        # A small comparison benchmark recreates the original extra capped
        # query and checks that shared-query slicing matches production.
        if RUN_UNCAPPED and sample_i <= BASELINE_BENCHMARK_SAMPLES:
            old_query_t0 = time.perf_counter()
            old_capped = idx.query(s1_rec, max_cands=MAX_CANDS)
            old_capped_query_seconds += time.perf_counter() - old_query_t0
            if old_capped != cand_int_indices_capped:
                raise RuntimeError(f"Capped candidate parity mismatch for {country} S1 {s1_id}")
            benchmark_comparisons += 1

        cand_counts.append(len(cand_int_indices_capped))
        if len(cand_int_indices_capped) == 0:
            zero_cand_count += 1

        for m in true_matches:
            total += 1
            match_pos = true_match_positions.get(m)
            in_capped = match_pos is not None and match_pos in cand_int_indices_capped
            in_uncapped = match_pos is not None and (match_pos in raw_indices if RUN_UNCAPPED else False)
            if in_uncapped:
                uncapped_found += 1
            if in_capped:
                found += 1
            else:
                if in_uncapped:
                    capped_true_match_lost += 1

                if m in true_match_records:
                    match_rec = true_match_records[m]
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
                        'found_uncapped': in_uncapped,
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

        if verbose and (sample_i % PROGRESS_EVERY == 0 or sample_i == len(s1_recs)):
            elapsed = time.perf_counter() - eval_t0
            per_s1 = elapsed / sample_i
            eta = per_s1 * (len(s1_recs) - sample_i)
            safe_print(f"  Evaluating {country.upper()}: {sample_i}/{len(s1_recs)} "
                       f"({elapsed:.1f}s elapsed, {per_s1:.3f}s/S1, ETA {eta:.1f}s)", flush=True)

    elapsed = time.perf_counter() - eval_t0
    recall = found / total if total else 1.0

    result = {
        'country': country,
        'sampled': len(s1_recs),
        'total_true_matches': total,
        'found': found,
        'uncapped_found': uncapped_found if RUN_UNCAPPED else None,
        'recall': recall,
        'uncapped_recall': (uncapped_found / total if total else 1.0) if RUN_UNCAPPED else None,
        'elapsed': elapsed,
        'index_seconds': index_seconds,
        'setup_seconds': time.perf_counter() - setup_t0 - index_seconds - elapsed,
        'avg_time_per_s1': elapsed / len(s1_recs) if s1_recs else 0.0,
        'optimized_query_seconds': optimized_query_seconds,
        'benchmark_query_seconds': benchmark_query_seconds,
        'old_capped_query_seconds': old_capped_query_seconds,
        'benchmark_comparisons': benchmark_comparisons,
        'cand_counts': cand_counts,
        'cand_counts_uncapped': cand_counts_uncapped,
        'zero_cand_count': zero_cand_count,
        'capped_count': capped_count,
        'capped_true_match_lost': capped_true_match_lost,
        'missed_details': missed_details,
    }

    # Cleanup
    del cand_ids, true_match_positions, true_match_records, seen_ids, idx
    gc.collect()

    return result


def print_report(results):
    """Print the comprehensive blocking recall report."""
    print("\n" + "=" * 70)
    print("  BLOCKING RECALL VALIDATION REPORT")
    print("=" * 70)

    combined_total = 0
    combined_found = 0
    combined_uncapped_found = 0
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
        if r['uncapped_recall'] is not None:
            print(f"  Uncapped recall: {r['uncapped_recall']*100:.2f}%")
        else:
            print("  Uncapped recall: not measured (RUN_UNCAPPED=False)")
        print(f"  Index time:    {r['index_seconds']:.1f}s")
        print(f"  Eval time:     {r['elapsed']:.1f}s")
        print(f"  Avg time/S1:   {r['avg_time_per_s1']:.3f}s")

        combined_total += r['total_true_matches']
        combined_found += r['found']
        combined_uncapped_found += r['uncapped_found'] or 0
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
    if RUN_UNCAPPED:
        combined_uncapped_recall = combined_uncapped_found / combined_total if combined_total else 1.0
        print(f"  Combined uncapped recall: {combined_uncapped_recall*100:.2f}%")

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
    if RUN_UNCAPPED:
        print(f"  S1 entities with >{MAX_CANDS} raw candidates: {total_capped:,}")
        print(f"  True matches lost due to capping:     {all_capped_loss:,}")
    else:
        print("  Cap impact not measured (RUN_UNCAPPED=False)")
    if RUN_UNCAPPED and combined_total > 0:
        print(f"  Recall loss from capping:             {all_capped_loss/combined_total*100:.3f}%")
    # Uncapped stats
    all_uncapped = []
    for r in results:
        if r:
            all_uncapped.extend(r['cand_counts_uncapped'])
    if RUN_UNCAPPED and all_uncapped:
        print(f"  Avg candidates (uncapped):            {statistics.mean(all_uncapped):.1f}")
        print(f"  Max candidates (uncapped):            {max(all_uncapped)}")

    # Performance comparison from identical S1 queries. The optimized path
    # uses one full query and slices it; the prior path made both full and
    # capped queries. This benchmark is limited to the first configured S1s.
    benchmark_n = sum(r['benchmark_comparisons'] for r in results if r)
    extra_capped_q = sum(r['old_capped_query_seconds'] for r in results if r)
    if benchmark_n:
        # Report benchmark-only times from the first N queries, not whole-sample query totals.
        bench_opt = sum(r.get('benchmark_query_seconds', 0.0) for r in results if r)
        old_total = bench_opt + extra_capped_q
        print(f"\n  QUERY PATH BENCHMARK ({benchmark_n} S1s)")
        print(f"  Shared-query time: {bench_opt:.3f}s")
        print(f"  Prior two-query time: {old_total:.3f}s")
        print(f"  Query-only speedup: {old_total/bench_opt:.2f}x" if bench_opt else "  Query-only speedup: n/a")
        print(f"  Total evaluation time: {sum(r['elapsed'] for r in results if r):.1f}s")
        if all(r['sampled'] >= SAMPLE_SIZE for r in results if r):
            projected_3000 = statistics.mean(r['avg_time_per_s1'] for r in results if r) * 3000
            print(f"  Estimated evaluation time for 3,000 S1/country: {projected_3000:.1f}s/country (excluding indexing)")

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
        if r['uncapped_recall'] is not None:
            print(f"  Uncapped recall: {r['uncapped_recall']*100:.2f}%")

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
    print(f"  Uncapped analysis: {RUN_UNCAPPED}")
    print("=" * 70)
    print("Progress is flushed during indexing and every "
          f"{PROGRESS_EVERY} evaluation S1s.", flush=True)

    results = []
    for country in ['india', 'us']:
        result = run_country_validation(country)
        results.append(result)
        gc.collect()

    print_report(results)
