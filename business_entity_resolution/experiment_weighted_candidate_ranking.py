#!/usr/bin/env python3
"""Compare deterministic weighted rankings over the existing seven-rule pool."""
import sys
import time
from collections import defaultdict
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from src.preprocessing.normalization import normalize_record
from src.pipeline.run_pipeline import CompactBlockingIndex, stream_records, load_ground_truth

SPLITS = ROOT / "splits"
SAMPLE_SIZE = 20
CAP = 200
PROGRESS_EVERY = 500_000

# Feature order: exact first token, exact four-character name prefix,
# shared significant name tokens, shared >=4-digit address numbers,
# shared significant address tokens (only a blocker at overlap >=2),
# exact three-consonant skeleton prefix, shared consonant trigrams (>=2).
SCHEMES = {
    "balanced": (12, 10, 4, 5, 2, 9, 2),
    "name_priority": (15, 14, 6, 2, 1, 12, 3),
    "address_rescue": (8, 7, 2, 8, 6, 6, 1),
    "consensus": (10, 9, 3, 4, 2, 8, 2),
}


def score_features(s1, idx):
    """Return per-position blocker features and the exact existing union."""
    features = defaultdict(lambda: [0] * 7)
    ft = s1.get("ft", "")
    if ft:
        for p in idx.first_token.get(ft, ()):
            features[p][0] = 1
    npfx = s1.get("npfx", "")
    if npfx:
        for p in idx.name_prefix.get(npfx, ()):
            features[p][1] = 1
    for token in sorted(s1.get("ntoks", ())):
        for p in idx.name_tokens.get(token, ()):
            features[p][2] += 1
    for number in sorted(set(s1.get("anums", ()))):
        for p in idx.addr_nums.get(number, ()):
            features[p][3] += 1
    atoks = s1.get("atoks", ())
    if len(atoks) >= 2:
        for token in sorted(atoks):
            for p in idx.addr_tokens.get(token, ()):
                features[p][4] += 1
    cs3 = s1.get("cskel3", "")
    if cs3 and len(cs3) >= 3:
        for p in idx.cskel3.get(cs3, ()):
            features[p][5] = 1
    cngrams = s1.get("cngrams", ())
    if len(cngrams) >= 2:
        for gram in sorted(cngrams):
            for p in idx.cngrams.get(gram, ()):
                features[p][6] += 1
    # Pool membership mirrors query(): the address-token and ngram strategies
    # require >=2 overlaps; all other present features independently block.
    pool = {p for p, f in features.items()
            if f[0] or f[1] or f[2] or f[3] or f[5] or f[4] >= 2 or f[6] >= 2}
    return features, pool


def diagnose_country(country):
    paths = {src: SPLITS / f"train_{src}_{country}.tsv" for src in ("S1", "S2", "S3", "GT")}
    gt = load_ground_truth(paths["GT"])
    matched = {sid: ids for sid, ids in gt.items() if ids}
    s1s = []
    for eid, name, addr, ctry in stream_records(paths["S1"]):
        if eid in matched:
            s1s.append(normalize_record(eid, name, addr, ctry))
            if len(s1s) == SAMPLE_SIZE:
                break
    wanted = set().union(*(matched.get(r["id"], set()) for r in s1s))
    print(f"\n{country.upper()}: indexing S2+S3 for {len(s1s)} S1s", flush=True)
    idx = CompactBlockingIndex()
    ids, pos_by_id, true_records = [], {}, {}
    seen = set()
    started = time.perf_counter()
    for src in ("S2", "S3"):
        for eid, name, addr, ctry in stream_records(paths[src]):
            if eid in seen:
                continue
            seen.add(eid)
            rec = normalize_record(eid, name, addr, ctry)
            pos = len(ids)
            ids.append(eid)
            pos_by_id[eid] = pos
            idx.add(pos, rec)
            if eid in wanted:
                true_records[eid] = rec
            if len(ids) % PROGRESS_EVERY == 0:
                print(f"  indexed {len(ids):,} candidates ({time.perf_counter()-started:.1f}s)", flush=True)
    print(f"  index ready: {len(ids):,} candidates ({time.perf_counter()-started:.1f}s)", flush=True)

    result = {name: {"true": 0, "found": 0, "sizes": [], "seconds": 0.0, "misses": []}
              for name in SCHEMES}
    uncapped_recovered = 0
    parity_failures = 0
    for n, s1 in enumerate(s1s, 1):
        feature_t0 = time.perf_counter()
        feat, pool = score_features(s1, idx)
        feature_seconds = time.perf_counter() - feature_t0
        # Verify precisely against the production union. Set order is never
        # consumed for ranking; sorting below always ends with the candidate ID.
        if pool != idx.query(s1, max_cands=len(ids) + 1):
            parity_failures += 1
        uncapped_recovered += sum(pos_by_id.get(mid) in pool for mid in matched[s1["id"]])
        for name, weights in SCHEMES.items():
            rank_t0 = time.perf_counter()
            scores = {p: sum(v * w for v, w in zip(feat[p], weights)) for p in pool}
            ranked = sorted(pool, key=lambda p: (-scores[p], ids[p]))
            result[name]["seconds"] += feature_seconds / len(SCHEMES) + time.perf_counter() - rank_t0
            result[name]["sizes"].append(min(CAP, len(ranked)))
            selected = set(ranked[:CAP])
            result[name]["true"] += len(matched[s1["id"]])
            result[name]["found"] += sum(pos_by_id.get(mid) in selected for mid in matched[s1["id"]])
            ranks = {p: i + 1 for i, p in enumerate(ranked)}
            for mid in matched[s1["id"]]:
                pos = pos_by_id.get(mid)
                if pos in pool and pos not in selected:
                    f = feat[pos]
                    sig = []
                    labels = ("first_token", "name_prefix", "name_tokens", "address_numbers", "address_tokens", "consonant_skeleton", "consonant_ngrams")
                    for label, value in zip(labels, f):
                        if value and (label not in ("address_tokens", "consonant_ngrams") or value >= 2):
                            sig.append(f"{label}={value}" if value > 1 else label)
                    result[name]["misses"].append((s1["id"], mid, scores[pos], ranks[pos], len(pool), sig,
                                                    s1, true_records.get(mid)))
        if n % 5 == 0 or n == len(s1s):
            print(f"  ranked {country.upper()} S1 {n}/{len(s1s)}", flush=True)
    return country, len(s1s), result, uncapped_recovered, parity_failures


def main():
    runs = [diagnose_country(c) for c in ("india", "us")]
    totals = {k: {"true": 0, "found": 0, "sizes": [], "seconds": 0.0, "misses": []} for k in SCHEMES}
    country_stats = {k: {} for k in SCHEMES}
    total_s1 = sum(run[1] for run in runs)
    print("\nWEIGHTED RANKING COMPARISON (CAP=200)")
    for country, n, data, unc, parity in runs:
        print(f"\n{country.upper()} uncapped recovered: {unc}; pool parity failures: {parity}")
        for key, d in data.items():
            country_stats[key][country] = d
            for field in ("true", "found", "seconds"):
                totals[key][field] += d[field]
            totals[key]["sizes"].extend(d["sizes"])
            totals[key]["misses"].extend(d["misses"])
            recall = 100 * d["found"] / d["true"] if d["true"] else 100
            print(f"  {key}: {d['found']}/{d['true']} ({recall:.2f}%), misses={d['true']-d['found']}, "
                  f"avg candidates/S1={sum(d['sizes'])/n:.1f}, ranking={d['seconds']:.3f}s ({d['seconds']/n:.4f}s/S1)")
    print("\nPER-SCHEME SUMMARY")
    for key, d in totals.items():
        print(f"\n{key} weights={SCHEMES[key]}")
        for c in ("india", "us"):
            x = country_stats[key][c]
            print(f"  {c.upper()}: true={x['true']} recovered={x['found']} recall={100*x['found']/x['true']:.2f}%")
        recall = 100 * d["found"] / d["true"]
        print(f"  Combined: true={d['true']} recovered={d['found']} recall={recall:.2f}% "
              f"missed={d['true']-d['found']} avg candidates/S1={sum(d['sizes'])/total_s1:.1f} "
              f"ranking={d['seconds']:.3f}s avg={d['seconds']/total_s1:.4f}s/S1")
    # Baseline values supplied from the existing diagnostic run.
    print("\nBASELINES: current=0/146 (0%); uncapped=146/146 (100%); simple ranking=117/146 (80.14%)")
    best = max(totals, key=lambda k: (totals[k]["found"], -totals[k]["seconds"], k))
    print(f"\nBEST WEIGHTED SCHEME: {best} {SCHEMES[best]}")
    misses = totals[best]["misses"]
    print("MISSED TRUE MATCHES: S1_ID\tTRUE_MATCH_ID\tSCORE\tRANK\tTOTAL_UNCAPPED_CANDIDATES\tBLOCKING_SIGNALS")
    for sid, mid, score, rank, size, sig, s1, rec in misses:
        print(f"{sid}\t{mid}\t{score}\t{rank}\t{size}\t{','.join(sig) or '(none)'}")
    print("\nMISS PATTERN DETAILS")
    for sid, mid, score, rank, size, sig, s1, rec in misses:
        if rec is None:
            pattern = "true match absent from indexed S2+S3 pool"
        else:
            name_overlap = len(s1.get('ntoks', set()) & rec.get('ntoks', set()))
            addr_overlap = len(s1.get('atoks', set()) & rec.get('atoks', set()))
            num_overlap = len(set(s1.get('anums', ())) & set(rec.get('anums', ())))
            pattern = f"name-token overlap={name_overlap}; address-token overlap={addr_overlap}; address-number overlap={num_overlap}"
        print(f"  {sid} -> {mid}: {pattern}; {','.join(sig) or 'no blocking signal'}; pool={size}")
    print("\nInterpretation: all reported misses are ranking/cap misses because each is present in the uncapped pool. "
          "Use pool size and shared-signal details above to distinguish broad-pool competition from weak overlap; "
          "a true match ranked above 200 indicates a score/tie ordering issue in this scheme.")
    print(f"Conclusion: best={best}; combined={totals[best]['found']}/{totals[best]['true']} ({100*totals[best]['found']/totals[best]['true']:.2f}%); "
          f"gain over 80.14%={100*totals[best]['found']/totals[best]['true']-80.14:+.2f} points; "
          f"runtime={totals[best]['seconds']/total_s1:.4f}s/S1; remaining misses={totals[best]['true']-totals[best]['found']}. "
          "Larger-sample testing is warranted only if the recall gain is material relative to this measured ranking cost.")


if __name__ == "__main__":
    main()
