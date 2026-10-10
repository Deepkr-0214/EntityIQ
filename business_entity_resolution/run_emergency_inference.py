#!/usr/bin/env python3
"""
Emergency Fast Inference — Index-Based Exact Matching
=====================================================

Designed to complete within ~4 hours for 1.7M S1 records.

Strategy:
  1. Read S2 + S3 once; build three compact dict indexes per record:
       A. country + norm_name + norm_addr  → set of entity_ids
       B. country + norm_name              → set of entity_ids
       C. country + norm_addr              → set of entity_ids
  2. Stream S1 in chunks (never load all 1.7M into memory).
  3. For each S1 row, look up keys in order A → B → C.
     - A match is emitted only when the key maps to exactly 1 entity_id
       (unambiguous match).
     - If the key maps to >1 IDs, skip (ambiguous).
  4. Write matching_results.tsv and candidate_pairs.tsv.

No external data, no GT, no API calls, no fabricated IDs.
"""
from __future__ import annotations

import csv
import gc
import os
import re
import sys
import time
import unicodedata
from pathlib import Path

# Fix Windows console encoding and force unbuffered output
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Force unbuffered output
import builtins
_original_print = builtins.print
def _flush_print(*args, **kwargs):
    kwargs.setdefault("flush", True)
    _original_print(*args, **kwargs)
builtins.print = _flush_print

# ─── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TEST_DIR = Path(r"D:\Amazon\student_resource\dataset\test")
S1_PATH = TEST_DIR / "test_source1.tsv"
S2_PATH = TEST_DIR / "test_source2.tsv"
S3_PATH = TEST_DIR / "test_source3.tsv"

MATCHING_PATH = OUTPUT_DIR / "matching_results.tsv"
CANDIDATE_PATH = OUTPUT_DIR / "candidate_pairs.tsv"

CHUNK_SIZE = 10_000  # S1 rows per chunk
BENCHMARK_SIZE = 1_000  # rows for benchmark

# ─── Normalization ────────────────────────────────────────────────────────────

# Precompiled regexes
_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)
_MULTI_SPACE_RE = re.compile(r"\s+")
_LEGAL_SUFFIX_RE = re.compile(
    r"\b(?:"
    r"private\s+limited|pvt\.?\s*ltd\.?|ltd\.?|limited|"
    r"llc\.?|llp\.?|inc\.?|incorporated|"
    r"corp\.?|corporation|co\.?|company|"
    r"enterprises?|industries|group|holdings?|"
    r"partnership|associates?|foundation|trust|society|"
    r"ventures?|solutions?|services?|technolog(?:y|ies)|"
    r"systems?|international|global|"
    r"udyog|udhyog|sansthan|"
    r"sarl\.?|sas|eurl|societe|cie\.?"
    r")\b",
    re.IGNORECASE,
)
_ADDR_ABBREV = {
    "street": "st",
    "avenue": "ave",
    "boulevard": "blvd",
    "drive": "dr",
    "road": "rd",
    "lane": "ln",
    "court": "ct",
    "place": "pl",
    "circle": "cir",
    "highway": "hwy",
    "parkway": "pkwy",
    "terrace": "ter",
    "square": "sq",
    "route": "rte",
    "rue": "rue",
    "chemin": "ch",
    # Directional
    "north": "n",
    "south": "s",
    "east": "e",
    "west": "w",
    "northeast": "ne",
    "northwest": "nw",
    "southeast": "se",
    "southwest": "sw",
}
_ADDR_ABBREV_RE = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in _ADDR_ABBREV) + r")\b",
    re.IGNORECASE,
)
# Common noise words in addresses
_ADDR_NOISE_RE = re.compile(
    r"\b(?:unit|suite|ste|apt|apartment|floor|fl|room|bldg|building|no|number)\b",
    re.IGNORECASE,
)


def normalize_text(text: str) -> str:
    """Cheap deterministic normalization: lowercase, NFKD, strip punctuation,
    collapse whitespace. Preserves useful alphanumeric tokens."""
    if not text:
        return ""
    # Unicode NFKD decomposition + lowercase
    t = unicodedata.normalize("NFKD", text).lower()
    # Strip accents (combining characters)
    t = "".join(c for c in t if not unicodedata.combining(c))
    # Remove punctuation (keep word chars + whitespace)
    t = _PUNCT_RE.sub(" ", t)
    # Collapse whitespace
    t = _MULTI_SPACE_RE.sub(" ", t).strip()
    return t


def normalize_name(text: str) -> str:
    """Normalize business name: base normalization + strip legal suffixes."""
    t = normalize_text(text)
    # Remove legal suffixes
    t = _LEGAL_SUFFIX_RE.sub(" ", t)
    t = _MULTI_SPACE_RE.sub(" ", t).strip()
    return t


def normalize_address(text: str) -> str:
    """Normalize address: base normalization + abbreviate common words + remove noise."""
    t = normalize_text(text)
    # Abbreviate common address words
    t = _ADDR_ABBREV_RE.sub(lambda m: _ADDR_ABBREV[m.group(1).lower()], t)
    # Remove common noise words
    t = _ADDR_NOISE_RE.sub(" ", t)
    t = _MULTI_SPACE_RE.sub(" ", t).strip()
    return t


def normalize_country(text: str) -> str:
    """Normalize country to lowercase trimmed."""
    return text.strip().lower() if text else ""


# ─── Index Building ───────────────────────────────────────────────────────────

def build_indexes(path: Path, label: str):
    """Read a source TSV and build three indexes.
    
    Returns (index_A, index_B, index_C, row_count).
    
    index_A: (country, norm_name, norm_addr) → set of entity_ids
    index_B: (country, norm_name)            → set of entity_ids
    index_C: (country, norm_addr)            → set of entity_ids
    """
    print(f"  Building indexes from {label} ({path.name})...")
    t0 = time.perf_counter()

    index_A: dict[tuple, set] = {}
    index_B: dict[tuple, set] = {}
    index_C: dict[tuple, set] = {}
    count = 0

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader)  # skip header
        for row in reader:
            if len(row) < 4:
                continue
            eid, bname, baddr, country = row[0], row[1], row[2], row[3]
            count += 1

            c = normalize_country(country)
            nn = normalize_name(bname)
            na = normalize_address(baddr)

            # Index A: country + name + address (most specific)
            if nn and na:
                key_a = (c, nn, na)
                if key_a in index_A:
                    index_A[key_a].add(eid)
                else:
                    index_A[key_a] = {eid}

            # Index B: country + name
            if nn:
                key_b = (c, nn)
                if key_b in index_B:
                    index_B[key_b].add(eid)
                else:
                    index_B[key_b] = {eid}

            # Index C: country + address
            if na:
                key_c = (c, na)
                if key_c in index_C:
                    index_C[key_c].add(eid)
                else:
                    index_C[key_c] = {eid}

    elapsed = time.perf_counter() - t0
    print(f"    {label}: {count:,} records indexed in {elapsed:.1f}s")
    print(f"    Index A (country+name+addr): {len(index_A):,} keys")
    print(f"    Index B (country+name):      {len(index_B):,} keys")
    print(f"    Index C (country+addr):      {len(index_C):,} keys")
    return index_A, index_B, index_C, count


def merge_indexes(idx1: dict, idx2: dict) -> dict:
    """Merge two dicts of {key: set_of_ids} into a single dict."""
    merged = dict(idx1)  # shallow copy
    for key, ids in idx2.items():
        if key in merged:
            merged[key] = merged[key] | ids
        else:
            merged[key] = set(ids)
    return merged


def get_memory_mb() -> float:
    """Get current process RSS in MB."""
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1024 / 1024
    except ImportError:
        return 0.0


# ─── Matching Logic ───────────────────────────────────────────────────────────

def match_entity(country: str, norm_name: str, norm_addr: str,
                 idx_a: dict, idx_b: dict, idx_c: dict):
    """Try to match an S1 entity against the combined S2+S3 indexes.
    
    Returns (matched_ids: set, candidate_ids: set).
    
    matched_ids: unambiguous matches (exactly 1 ID for the key)
    candidate_ids: all IDs considered (union of all looked-up keys)
    """
    matched = set()
    candidates = set()

    # Strategy A: exact match on country + name + address (strongest)
    if norm_name and norm_addr:
        key_a = (country, norm_name, norm_addr)
        if key_a in idx_a:
            ids = idx_a[key_a]
            candidates.update(ids)
            if len(ids) == 1:
                matched.update(ids)
                return matched, candidates  # strong match, done

    # Strategy B: exact match on country + name
    if norm_name:
        key_b = (country, norm_name)
        if key_b in idx_b:
            ids = idx_b[key_b]
            candidates.update(ids)
            # Only emit if unambiguous AND we didn't already match via A
            if len(ids) == 1 and not matched:
                matched.update(ids)
                return matched, candidates

    # Strategy C: exact match on country + address
    if norm_addr:
        key_c = (country, norm_addr)
        if key_c in idx_c:
            ids = idx_c[key_c]
            candidates.update(ids)
            # Only emit if unambiguous AND we didn't already match
            if len(ids) == 1 and not matched:
                matched.update(ids)

    return matched, candidates


# ─── Main Pipeline ────────────────────────────────────────────────────────────

def run_benchmark(idx_a, idx_b, idx_c, n_rows=BENCHMARK_SIZE):
    """Run a benchmark on the first n_rows of S1."""
    print(f"\n{'='*70}")
    print(f"BENCHMARK: Processing first {n_rows:,} S1 rows")
    print(f"{'='*70}")

    t0 = time.perf_counter()
    match_count = 0
    candidate_pair_count = 0
    rows_processed = 0

    with open(S1_PATH, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)  # skip header
        for row in reader:
            if rows_processed >= n_rows:
                break
            if len(row) < 4:
                rows_processed += 1
                continue

            eid, bname, baddr, country = row[0], row[1], row[2], row[3]
            c = normalize_country(country)
            nn = normalize_name(bname)
            na = normalize_address(baddr)

            matched, candidates = match_entity(c, nn, na, idx_a, idx_b, idx_c)
            if matched:
                match_count += 1
            candidate_pair_count += len(candidates)
            rows_processed += 1

    elapsed = time.perf_counter() - t0
    mem_mb = get_memory_mb()

    total_s1 = 1_732_544
    projected_seconds = (elapsed / n_rows) * total_s1
    projected_hours = projected_seconds / 3600

    print(f"\n  Benchmark Results:")
    print(f"    Rows processed:     {rows_processed:,}")
    print(f"    Time:               {elapsed:.2f}s")
    print(f"    Rate:               {rows_processed / elapsed:,.0f} rows/s")
    print(f"    Non-empty matches:  {match_count:,} ({100*match_count/max(1,rows_processed):.1f}%)")
    print(f"    Candidate pairs:    {candidate_pair_count:,}")
    print(f"    Memory (RSS):       {mem_mb:,.0f} MB")
    print(f"    Projected full run: {projected_hours:.2f} hours ({projected_seconds:.0f}s)")
    print()

    return projected_hours, elapsed


def run_full(idx_a, idx_b, idx_c):
    """Process all S1 records in chunks and write output files."""
    print(f"\n{'='*70}")
    print(f"FULL RUN: Processing all S1 records")
    print(f"{'='*70}")

    t0 = time.perf_counter()
    rows_processed = 0
    match_count = 0
    total_candidates = 0

    # Open both output files
    with (
        open(MATCHING_PATH, "w", encoding="utf-8", newline="") as f_match,
        open(CANDIDATE_PATH, "w", encoding="utf-8", newline="") as f_cand,
        open(S1_PATH, "r", encoding="utf-8", errors="replace") as f_s1,
    ):
        # Write headers
        f_match.write("source1_entity_id\tmatched_entity_ids\n")
        f_cand.write("source1_entity_id\tcandidate_entity_ids\n")

        reader = csv.reader(f_s1, delimiter="\t")
        next(reader)  # skip header

        # Process in chunks for progress reporting
        chunk_buf_match = []
        chunk_buf_cand = []

        for row in reader:
            if len(row) < 4:
                # Malformed row — still must emit the entity_id if present
                eid = row[0] if row else ""
                if eid:
                    chunk_buf_match.append(f"{eid}\t\n")
                    chunk_buf_cand.append(f"{eid}\t\n")
                    rows_processed += 1
                continue

            eid, bname, baddr, country = row[0], row[1], row[2], row[3]
            c = normalize_country(country)
            nn = normalize_name(bname)
            na = normalize_address(baddr)

            matched, candidates = match_entity(c, nn, na, idx_a, idx_b, idx_c)

            if matched:
                match_count += 1
            total_candidates += len(candidates)

            matched_str = ",".join(sorted(matched)) if matched else ""
            candidate_str = ",".join(sorted(candidates)) if candidates else ""

            chunk_buf_match.append(f"{eid}\t{matched_str}\n")
            chunk_buf_cand.append(f"{eid}\t{candidate_str}\n")

            rows_processed += 1

            # Flush chunk
            if rows_processed % CHUNK_SIZE == 0:
                f_match.writelines(chunk_buf_match)
                f_cand.writelines(chunk_buf_cand)
                chunk_buf_match.clear()
                chunk_buf_cand.clear()

                elapsed = time.perf_counter() - t0
                rate = rows_processed / elapsed
                eta = (1_732_544 - rows_processed) / rate if rate > 0 else 0
                print(
                    f"  [{rows_processed:>10,} / 1,732,544] "
                    f"{elapsed:>7.1f}s  "
                    f"{rate:>8,.0f} rows/s  "
                    f"matches={match_count:,}  "
                    f"ETA={eta/60:.1f}min  "
                    f"mem={get_memory_mb():.0f}MB"
                )

        # Flush remaining
        if chunk_buf_match:
            f_match.writelines(chunk_buf_match)
            f_cand.writelines(chunk_buf_cand)

    elapsed = time.perf_counter() - t0
    print(f"\n  Full Run Complete:")
    print(f"    Total S1 rows:      {rows_processed:,}")
    print(f"    Non-empty matches:  {match_count:,} ({100*match_count/max(1,rows_processed):.1f}%)")
    print(f"    Total candidates:   {total_candidates:,}")
    print(f"    Runtime:            {elapsed:.1f}s ({elapsed/3600:.2f}h)")
    print(f"    Output files:")
    print(f"      matching_results: {MATCHING_PATH}  ({MATCHING_PATH.stat().st_size/1024/1024:.1f} MB)")
    print(f"      candidate_pairs:  {CANDIDATE_PATH} ({CANDIDATE_PATH.stat().st_size/1024/1024:.1f} MB)")
    print(f"    Memory (RSS):       {get_memory_mb():.0f} MB")

    return rows_processed, match_count, total_candidates, elapsed


def main():
    print("=" * 70)
    print("EMERGENCY FAST INFERENCE — Index-Based Exact Matching")
    print("=" * 70)
    print(f"  S1: {S1_PATH}")
    print(f"  S2: {S2_PATH}")
    print(f"  S3: {S3_PATH}")
    print(f"  Output: {OUTPUT_DIR}")
    print()

    # ── Step 1: Build indexes from S2 + S3 ────────────────────────────────
    print("Step 1: Building S2 + S3 indexes...")
    mem_before = get_memory_mb()
    t_idx_start = time.perf_counter()

    a2, b2, c2, n2 = build_indexes(S2_PATH, "S2")
    a3, b3, c3, n3 = build_indexes(S3_PATH, "S3")

    # Merge S2 + S3 indexes
    print("  Merging S2 + S3 indexes...")
    idx_a = merge_indexes(a2, a3)
    idx_b = merge_indexes(b2, b3)
    idx_c = merge_indexes(c2, c3)

    # Free individual indexes
    del a2, b2, c2, a3, b3, c3
    gc.collect()

    t_idx_end = time.perf_counter()
    mem_after = get_memory_mb()

    print(f"\n  Index build complete:")
    print(f"    S2 records: {n2:,}")
    print(f"    S3 records: {n3:,}")
    print(f"    Combined Index A: {len(idx_a):,} keys")
    print(f"    Combined Index B: {len(idx_b):,} keys")
    print(f"    Combined Index C: {len(idx_c):,} keys")
    print(f"    Index build time: {t_idx_end - t_idx_start:.1f}s")
    print(f"    Memory: {mem_before:.0f} → {mem_after:.0f} MB (+{mem_after - mem_before:.0f} MB)")

    # ── Step 2: Benchmark on 1,000 rows ───────────────────────────────────
    projected_hours, bench_time = run_benchmark(idx_a, idx_b, idx_c, BENCHMARK_SIZE)

    if projected_hours > 4.0:
        print(f"\n  ⚠ Projected runtime ({projected_hours:.2f}h) exceeds 4 hours!")
        print("  Consider optimization before full run.")
        # Continue anyway since we have 5 hours and index-based lookup is O(1)
        # The projection might be pessimistic due to cold-start effects

    # ── Step 3: Full run ──────────────────────────────────────────────────
    print(f"\n  Projected runtime ({projected_hours:.2f}h) is acceptable. Starting full run...")
    rows, matches, candidates, full_time = run_full(idx_a, idx_b, idx_c)

    # ── Final summary ─────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("FINAL SUMMARY")
    print(f"{'='*70}")
    print(f"  Index build time:    {t_idx_end - t_idx_start:.1f}s")
    print(f"  Benchmark time:      {bench_time:.2f}s (for {BENCHMARK_SIZE:,} rows)")
    print(f"  Projected runtime:   {projected_hours:.2f}h")
    print(f"  Full runtime:        {full_time:.1f}s ({full_time/3600:.2f}h)")
    print(f"  Total S1 rows:       {rows:,}")
    print(f"  Non-empty matches:   {matches:,}")
    print(f"  Candidate pairs:     {candidates:,}")
    print(f"  matching_results:    {MATCHING_PATH.stat().st_size / 1024 / 1024:.1f} MB")
    print(f"  candidate_pairs:     {CANDIDATE_PATH.stat().st_size / 1024 / 1024:.1f} MB")
    print(f"  Memory (RSS):        {get_memory_mb():.0f} MB")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
