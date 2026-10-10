#!/usr/bin/env python3
"""V2 weighted-ranking experiments; candidate pool is the existing blocker union."""
import sys
import time
import heapq
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

# Feature vector: first-token, prefix, shared name-token count, shared address-
# number count, shared address-token count, skeleton, shared consonant-trigram
# count. Existing address-token/ngram threshold rules still define pool only.
# Scheme weights were chosen to probe the requested failure modes.
SCHEMES = {
    "number_bonus": {"w": (7, 6, 2, 24, 5, 4, 1), "combo": 0, "consensus": 0, "caps": None},
    "address_combo": {"w": (7, 6, 2, 16, 7, 4, 1), "combo": 16, "consensus": 0, "caps": None},
    "independent_consensus": {"w": (6, 5, 2, 10, 5, 4, 1), "combo": 10, "consensus": 5, "caps": (1, 1, 3, 2, 4, 1, 4)},
    "normalized_combo": {"w": (7, 6, 2, 14, 6, 4, 1), "combo": 14, "consensus": 4, "caps": (1, 1, 3, 2, 4, 1, 4)},
    "address_heavy_combo": {"w": (5, 4, 1, 12, 12, 3, 1), "combo": 20, "consensus": 3, "caps": (1, 1, 2, 2, 4, 1, 3)},
}

LABELS = ("first_token", "name_prefix", "name_tokens", "address_numbers", "address_tokens", "consonant_skeleton", "consonant_ngrams")


def get_features(s1, idx):
    """Aggregate all current blocking evidence, returning exactly query()'s union."""
    f = defaultdict(lambda: [0] * 7)
    if s1.get("ft"):
        for p in idx.first_token.get(s1["ft"], ()):
            f[p][0] = 1
    if s1.get("npfx"):
        for p in idx.name_prefix.get(s1["npfx"], ()):
            f[p][1] = 1
    for tok in s1.get("ntoks", ()):
        for p in idx.name_tokens.get(tok, ()):
            f[p][2] += 1
    for num in set(s1.get("anums", ())):
        for p in idx.addr_nums.get(num, ()):
            f[p][3] += 1
    atoks = s1.get("atoks", ())
    if len(atoks) >= 2:
        for tok in atoks:
            for p in idx.addr_tokens.get(tok, ()):
                f[p][4] += 1
    cs = s1.get("cskel3", "")
    if cs and len(cs) >= 3:
        for p in idx.cskel3.get(cs, ()):
            f[p][5] = 1
    cng = s1.get("cngrams", ())
    if len(cng) >= 2:
        for gram in cng:
            for p in idx.cngrams.get(gram, ()):
                f[p][6] += 1
    pool = {p for p, x in f.items() if x[0] or x[1] or x[2] or x[3] or x[5] or x[4] >= 2 or x[6] >= 2}
    return f, pool


def contribution_parts(features, scheme):
    cfg = SCHEMES[scheme]
    vals = features if cfg["caps"] is None else tuple(min(x, cap) for x, cap in zip(features, cfg["caps"]))
    parts = [v * w for v, w in zip(vals, cfg["w"])]
    name_evidence = bool(features[0] or features[1] or features[2])
    addr_evidence = bool(features[3] or features[4] >= 2)
    if cfg["combo"] and name_evidence and addr_evidence:
        parts.append(cfg["combo"])
    if cfg["consensus"]:
        # Count independent evidence families (name, address, phonetic), not
        # individual correlated keys within each family.
        families = int(name_evidence) + int(addr_evidence) + int(bool(features[5] or features[6] >= 2))
        parts.append(cfg["consensus"] * families)
    return parts


def deterministic_top(features_by_pos, pool, ids, scheme):
    """Select exact top CAP by score then ID without sorting the full pool."""
    cfg = SCHEMES[scheme]
    weights = cfg["w"]
    caps = cfg["caps"]
    combo = cfg["combo"]
    consensus = cfg["consensus"]
    def key(p):
        f = features_by_pos[p]
        if caps is None:
            vals = f
        else:
            vals = tuple(min(x, cap) for x, cap in zip(f, caps))
        score = sum(v * w for v, w in zip(vals, weights))
        name_ev = bool(f[0] or f[1] or f[2])
        addr_ev = bool(f[3] or f[4] >= 2)
        if combo and name_ev and addr_ev:
            score += combo
        if consensus:
            score += consensus * (int(name_ev) + int(addr_ev) + int(bool(f[5] or f[6] >= 2)))
        # nsmallest sorts this ascending: larger score first, ID ascending.
        return -score, ids[p]
    best = heapq.nsmallest(CAP, pool, key=key)
    return best, key


def country_run(country):
    paths = {s: SPLITS / f"train_{s}_{country}.tsv" for s in ("S1", "S2", "S3", "GT")}
    gt = load_ground_truth(paths["GT"])
    truth = {sid: mids for sid, mids in gt.items() if mids}
    s1s = []
    for eid, name, addr, ctry in stream_records(paths["S1"]):
        if eid in truth:
            s1s.append(normalize_record(eid, name, addr, ctry))
            if len(s1s) == SAMPLE_SIZE:
                break
    needed = set().union(*(truth[s["id"]] for s in s1s))
    print(f"\n{country.upper()}: building same S2+S3 index", flush=True)
    idx = CompactBlockingIndex()
    ids, by_id, true_records, seen = [], {}, {}, set()
    t0 = time.perf_counter()
    for src in ("S2", "S3"):
        for eid, name, addr, ctry in stream_records(paths[src]):
            if eid in seen:
                continue
            seen.add(eid)
            rec = normalize_record(eid, name, addr, ctry)
            p = len(ids)
            ids.append(eid)
            by_id[eid] = p
            idx.add(p, rec)
            if eid in needed:
                true_records[eid] = rec
            if len(ids) % PROGRESS_EVERY == 0:
                print(f"  indexed {len(ids):,} ({time.perf_counter()-t0:.1f}s)", flush=True)
    print(f"  index ready: {len(ids):,} candidates ({time.perf_counter()-t0:.1f}s)", flush=True)

    out = {k: {"true": 0, "found": 0, "sizes": [], "secs": 0.0, "misses": [], "examples": []} for k in SCHEMES}
    uncapped_found, parity_fail = 0, 0
    for n, s1 in enumerate(s1s, 1):
        ft0 = time.perf_counter()
        feats, pool = get_features(s1, idx)
        feat_time = time.perf_counter() - ft0
        production_pool = idx.query(s1, max_cands=len(ids) + 1)
        if pool != production_pool:
            parity_fail += 1
        uncapped_found += sum(by_id.get(mid) in pool for mid in truth[s1["id"]])
        for scheme in SCHEMES:
            rt0 = time.perf_counter()
            top, keyfn = deterministic_top(feats, pool, ids, scheme)
            rank_time = time.perf_counter() - rt0
            d = out[scheme]
            d["secs"] += feat_time / len(SCHEMES) + rank_time
            d["sizes"].append(min(CAP, len(pool)))
            chosen = set(top)
            d["true"] += len(truth[s1["id"]])
            d["found"] += sum(by_id.get(mid) in chosen for mid in truth[s1["id"]])
            d["examples"].append((s1["id"], [(ids[q], -keyfn(q)[0], tuple(feats[q])) for q in top[:10]]))
            for mid in truth[s1["id"]]:
                p = by_id.get(mid)
                if p not in pool or p in chosen:
                    continue
                # Rank only the match for diagnostics; number ahead is exact
                # (including deterministic ID tie-break), without full sort.
                rankkey = keyfn(p)
                rank = 1 + sum(keyfn(q) < rankkey for q in pool)
                d["misses"].append((s1["id"], mid, -rankkey[0], rank, len(pool), feats[p], s1, true_records.get(mid)))
        if n % 5 == 0 or n == len(s1s):
            print(f"  ranked {country.upper()} {n}/{len(s1s)}", flush=True)
    return country, len(s1s), out, uncapped_found, parity_fail


def print_signal_detail(s1, rec, features, scheme):
    cfg = SCHEMES[scheme]
    vals = features if cfg["caps"] is None else tuple(min(x, cap) for x, cap in zip(features, cfg["caps"]))
    parts = [(label, value, value * weight) for label, value, weight in zip(LABELS, features, cfg["w"])]
    print("    matching signals / contribution: " + "; ".join(f"{label}={value} (score +{contrib})" for label, value, contrib in parts if value))
    if cfg["combo"] and (features[0] or features[1] or features[2]) and (features[3] or features[4] >= 2):
        print(f"    name+address combination bonus: +{cfg['combo']}")
    if cfg["consensus"]:
        fam = int(bool(features[0] or features[1] or features[2])) + int(bool(features[3] or features[4] >= 2)) + int(bool(features[5] or features[6] >= 2))
        print(f"    independent-family consensus: {fam} families x {cfg['consensus']} = +{fam * cfg['consensus']}")
    if rec:
        name_overlap = len(s1.get("ntoks", set()) & rec.get("ntoks", set()))
        atok_overlap = len(s1.get("atoks", set()) & rec.get("atoks", set()))
        anum_overlap = len(set(s1.get("anums", ())) & set(rec.get("anums", ())))
        print(f"    overlap counts: name tokens={name_overlap}, address tokens={atok_overlap}, address numbers={anum_overlap}")


def main():
    runs = [country_run(c) for c in ("india", "us")]
    total_s1 = sum(x[1] for x in runs)
    totals = {k: {"true": 0, "found": 0, "secs": 0.0, "sizes": [], "misses": []} for k in SCHEMES}
    by_country = {k: {} for k in SCHEMES}
    print("\nV2 RESULTS (all pools checked against production uncapped query)")
    for country, n, data, uncapped, parity in runs:
        print(f"\n{country.upper()}: uncapped truth recovered={uncapped}; parity failures={parity}")
        for scheme, d in data.items():
            by_country[scheme][country] = d
            t = totals[scheme]
            t["true"] += d["true"]; t["found"] += d["found"]; t["secs"] += d["secs"]
            t["sizes"].extend(d["sizes"]); t["misses"].extend(d["misses"])
            print(f"  {scheme}: {d['found']}/{d['true']} ({100*d['found']/d['true']:.2f}%), misses={d['true']-d['found']}, "
                  f"avg selected={sum(d['sizes'])/n:.1f}, ranking={d['secs']:.3f}s ({d['secs']/n:.4f}s/S1)")
    print("\nSCHEME SUMMARY AND BASELINES")
    print("Current cap 0/146 (0%); simple 117/146 (80.14%); Address Rescue 135/146 (92.47%); uncapped 146/146 (100%)")
    for name, d in totals.items():
        print(f"\n{name}: {SCHEMES[name]}")
        for country in ("india", "us"):
            x = by_country[name][country]
            print(f"  {country.upper()}: {x['found']}/{x['true']} ({100*x['found']/x['true']:.2f}%)")
        print(f"  Combined: {d['found']}/{d['true']} ({100*d['found']/d['true']:.2f}%), misses={d['true']-d['found']}, "
              f"average selected={sum(d['sizes'])/total_s1:.1f}, ranking={d['secs']:.3f}s, {d['secs']/total_s1:.4f}s/S1")
    best = max(totals, key=lambda k: (totals[k]["found"], -totals[k]["secs"], k))
    print(f"\nBEST V2: {best} config={SCHEMES[best]}")
    print("MISSES: S1_ID\tTRUE_MATCH_ID\tSCORE\tRANK\tUNCAPPED_CANDIDATES\tSIGNALS")
    for sid, mid, score, rank, count, f, s1, rec in totals[best]["misses"]:
        print(f"{sid}\t{mid}\t{score}\t{rank}\t{count}\t" + ",".join(f"{l}={v}" for l,v in zip(LABELS,f) if v))
        print_signal_detail(s1, rec, f, best)
    # Three representative missed S1s with the largest pool / lowest true-match rank.
    miss_by_s1 = {}
    for miss in totals[best]["misses"]:
        miss_by_s1.setdefault(miss[0], miss)
    representatives = sorted(miss_by_s1.values(), key=lambda m: (-m[4], -m[3]))[:3]
    print("\nTOP 10 CANDIDATES FOR THREE REPRESENTATIVE MISSED S1s")
    for miss in representatives:
        sid = miss[0]
        top_rows = None
        for country in ("india", "us"):
            for ex in by_country[best][country]["examples"]:
                if ex[0] == sid:
                    top_rows = ex[1]
                    break
            if top_rows is not None: break
        if top_rows is None:
            print(f"  {sid}: top-candidate context not retained")
            continue
        print(f"\n  {sid} true match={miss[1]} true rank={miss[3]} pool={miss[4]}")
        for i, (eid, score, f) in enumerate(top_rows, 1):
            print(f"    {i:2d}. {eid} score={score} signals=" + ",".join(f"{l}={v}" for l,v in zip(LABELS,f) if v))
    print("\nConclusion: " + f"best={best}; combined={totals[best]['found']}/{totals[best]['true']} "
          f"({100*totals[best]['found']/totals[best]['true']:.2f}%); gain over Address Rescue="
          f"{100*totals[best]['found']/totals[best]['true']-92.47:+.2f} points; misses={totals[best]['true']-totals[best]['found']}; "
          f"runtime={totals[best]['secs']/total_s1:.4f}s/S1. Pools are large and several misses have weak evidence; "
          "practicality depends on the measured runtime and whether the recall gain justifies it.")


if __name__ == "__main__":
    main()
