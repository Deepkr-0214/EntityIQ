#!/usr/bin/env python3
"""Targeted V3 scoring experiments on the fixed existing blocker candidate pool."""
import sys, time, heapq
from collections import defaultdict
from pathlib import Path

if sys.platform == "win32":
    try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception: pass
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from src.preprocessing.normalization import normalize_record
from src.pipeline.run_pipeline import CompactBlockingIndex, stream_records, load_ground_truth

SPLITS, SAMPLE_SIZE, CAP = ROOT / "splits", 20, 200
PROGRESS_EVERY = 500_000
# Vector: first token, prefix, name-token count, address-number count,
# address-token count, skeleton, consonant-trigram count.
SCHEMES = {
    "number_bonus_control": dict(w=(7,6,2,24,5,4,1)),
    "address_rescue_control": dict(w=(8,7,2,8,6,6,1)),
    # Address Rescue with a small bonus for independent name/address/phonetic families.
    "rescue_family_consensus": dict(w=(8,7,2,8,6,6,1), consensus=3),
    # Saturate repeated address / phonetic counts so high raw overlap cannot dominate.
    "rescue_saturated_counts": dict(w=(8,7,2,8,6,6,1), caps=(1,1,3,2,3,1,4)),
    # Small boost when both independent exact-name keys agree; less address-token weight.
    "name_anchor_combo": dict(w=(10,8,2,8,5,6,1), name_combo=6),
    # Test address dominance directly: capped address count + stronger name anchors.
    "balanced_saturated_combo": dict(w=(10,8,2,8,5,6,1), caps=(1,1,3,2,3,1,4), combo=5, consensus=2),
}
LABELS=("first_token","name_prefix","name_tokens","address_numbers","address_tokens","consonant_skeleton","consonant_ngrams")

def features_and_pool(s1, idx):
    f=defaultdict(lambda:[0]*7)
    if s1.get("ft"):
        for p in idx.first_token.get(s1["ft"],()): f[p][0]=1
    if s1.get("npfx"):
        for p in idx.name_prefix.get(s1["npfx"],()): f[p][1]=1
    for x in s1.get("ntoks",()):
        for p in idx.name_tokens.get(x,()): f[p][2]+=1
    for x in set(s1.get("anums",())):
        for p in idx.addr_nums.get(x,()): f[p][3]+=1
    if len(s1.get("atoks",()))>=2:
        for x in s1["atoks"]:
            for p in idx.addr_tokens.get(x,()): f[p][4]+=1
    cs=s1.get("cskel3","")
    if cs and len(cs)>=3:
        for p in idx.cskel3.get(cs,()): f[p][5]=1
    if len(s1.get("cngrams",()))>=2:
        for x in s1["cngrams"]:
            for p in idx.cngrams.get(x,()): f[p][6]+=1
    pool={p for p,x in f.items() if x[0] or x[1] or x[2] or x[3] or x[5] or x[4]>=2 or x[6]>=2}
    return f,pool

def make_key(f, ids, name):
    cfg=SCHEMES[name]; weights=cfg["w"]; caps=cfg.get("caps")
    def key(p):
        x=f[p]
        vals=x if caps is None else tuple(min(v,c) for v,c in zip(x,caps))
        score=sum(v*w for v,w in zip(vals,weights))
        name_ev=bool(x[0] or x[1] or x[2]); addr_ev=bool(x[3] or x[4]>=2)
        if cfg.get("name_combo") and x[0] and x[1]: score+=cfg["name_combo"]
        if cfg.get("combo") and name_ev and addr_ev: score+=cfg["combo"]
        if cfg.get("consensus"):
            families=int(name_ev)+int(addr_ev)+int(bool(x[5] or x[6]>=2))
            score+=families*cfg["consensus"]
        return -score,ids[p]
    return key

def run_country(country):
    paths={s:SPLITS/f"train_{s}_{country}.tsv" for s in ("S1","S2","S3","GT")}
    gt=load_ground_truth(paths["GT"]); truth={k:v for k,v in gt.items() if v}
    s1s=[]
    for eid,n,a,c in stream_records(paths["S1"]):
        if eid in truth:
            s1s.append(normalize_record(eid,n,a,c))
            if len(s1s)==SAMPLE_SIZE: break
    needed=set().union(*(truth[s["id"]] for s in s1s))
    idx=CompactBlockingIndex(); ids=[]; byid={}; trecs={}; seen=set(); t0=time.perf_counter()
    print(f"\n{country.upper()}: indexing S2+S3",flush=True)
    for src in ("S2","S3"):
        for eid,n,a,c in stream_records(paths[src]):
            if eid in seen: continue
            seen.add(eid); rec=normalize_record(eid,n,a,c); p=len(ids)
            ids.append(eid); byid[eid]=p; idx.add(p,rec)
            if eid in needed: trecs[eid]=rec
            if len(ids)%PROGRESS_EVERY==0: print(f"  indexed {len(ids):,} ({time.perf_counter()-t0:.1f}s)",flush=True)
    print(f"  index ready: {len(ids):,} ({time.perf_counter()-t0:.1f}s)",flush=True)
    out={k:dict(true=0,found=0,secs=0.,misses=[],top={}) for k in SCHEMES}; uncapped=0; parity=0
    for i,s1 in enumerate(s1s,1):
        ft=time.perf_counter(); f,pool=features_and_pool(s1,idx); feature_secs=time.perf_counter()-ft
        if pool!=idx.query(s1,max_cands=len(ids)+1): parity+=1
        uncapped+=sum(byid.get(mid) in pool for mid in truth[s1["id"]])
        for name in SCHEMES:
            t=time.perf_counter(); key=make_key(f,ids,name)
            top=heapq.nsmallest(CAP,pool,key=key)
            out[name]["secs"]+=feature_secs/len(SCHEMES)+time.perf_counter()-t
            chosen=set(top); d=out[name]; d["true"]+=len(truth[s1["id"]]); d["found"]+=sum(byid.get(mid) in chosen for mid in truth[s1["id"]])
            d["top"][s1["id"]]=[(ids[q],-key(q)[0],tuple(f[q])) for q in top[:10]]
            for mid in truth[s1["id"]]:
                p=byid.get(mid)
                if p not in pool or p in chosen: continue
                rk=key(p); rank=1+sum(key(q)<rk for q in pool)
                # Three highest-ranked competitors immediately ahead of this true match.
                comps=heapq.nsmallest(3,(q for q in pool if key(q)<rk),key=key)
                d["misses"].append((s1["id"],mid,-rk[0],rank,len(pool),tuple(f[p]),s1,trecs.get(mid),[(ids[q],-key(q)[0],tuple(f[q])) for q in comps]))
        if i%5==0 or i==len(s1s): print(f"  ranked {country.upper()} {i}/{len(s1s)}",flush=True)
    return country,len(s1s),out,uncapped,parity

def main():
    runs=[run_country(c) for c in ("india","us")]; total_s1=sum(r[1] for r in runs)
    totals={k:dict(true=0,found=0,secs=0.,misses=[],top={}) for k in SCHEMES}; bycountry={k:{} for k in SCHEMES}
    for c,n,data,unc,parity in runs:
        print(f"\n{c.upper()}: uncapped matches={unc}; pool parity failures={parity}")
        for k,d in data.items():
            x=totals[k]; bycountry[k][c]=d; x["true"]+=d["true"]; x["found"]+=d["found"]; x["secs"]+=d["secs"]; x["misses"].extend(d["misses"]); x["top"].update(d["top"])
            print(f"  {k}: {d['found']}/{d['true']} ({100*d['found']/d['true']:.2f}%), misses={d['true']-d['found']}, selected=200, time={d['secs']/n:.4f}s/S1")
    print("\nBASELINES: current=0/146; simple=117/146 (80.14%); Address Rescue=135/146 (92.47%); uncapped=146/146")
    for k,d in totals.items():
        print(f"{k}: {d['found']}/{d['true']} ({100*d['found']/d['true']:.2f}%), misses={d['true']-d['found']}, runtime={d['secs']/total_s1:.4f}s/S1")
    best=max(totals,key=lambda k:(totals[k]["found"],-totals[k]["secs"],k)); d=totals[best]
    print(f"\nBEST V3: {best} config={SCHEMES[best]}")
    print("13-MISS REVIEW (or misses for winning variant): each true candidate followed by its three nearest higher-ranked competitors")
    for sid,mid,score,rank,pool,f,s1,rec,comps in d["misses"]:
        print(f"\n{sid} -> {mid}: score={score}, rank={rank}, pool={pool}; true signals="+",".join(f"{l}={v}" for l,v in zip(LABELS,f) if v))
        for cid,sc,cf in comps:
            print(f"  beats it: {cid} score={sc}; signals="+",".join(f"{l}={v}" for l,v in zip(LABELS,cf) if v))
        if rec:
            print(f"  true raw overlaps: name={len(s1['ntoks']&rec['ntoks'])}, address={len(s1['atoks']&rec['atoks'])}, numbers={len(set(s1['anums'])&set(rec['anums']))}")
    # The Number bonus misses remain especially useful for direct failure audit.
    nd=totals["number_bonus_control"]
    if best!="number_bonus_control":
        print("\nNUMBER BONUS CONTROL: miss comparison")
        for sid,mid,score,rank,pool,f,s1,rec,comps in nd["misses"]:
            print(f"{sid}->{mid} score={score} rank={rank} pool={pool} true="+",".join(f"{l}={v}" for l,v in zip(LABELS,f) if v))
            for cid,sc,cf in comps: print(f"  higher {cid} score={sc} signals="+",".join(f"{l}={v}" for l,v in zip(LABELS,cf) if v))
    print("\nTOP 10 FOR THREE REPRESENTATIVE MISSED S1s")
    miss_by_sid={}
    for m in d["misses"]: miss_by_sid.setdefault(m[0],m)
    reps=sorted(miss_by_sid.values(),key=lambda m:(-m[4],-m[3]))[:3]
    for m in reps:
        print(f"\n{m[0]} true={m[1]} rank={m[3]} pool={m[4]}")
        for rank,(cid,score,f) in enumerate(d["top"][m[0]],1): print(f"  {rank}. {cid} score={score} signals="+",".join(f"{l}={v}" for l,v in zip(LABELS,f) if v))
    print(f"\nConclusion: best={best}; {d['found']}/{d['true']} ({100*d['found']/d['true']:.2f}%), Address Rescue delta={100*d['found']/d['true']-92.47:+.2f} points, misses={d['true']-d['found']}, runtime={d['secs']/total_s1:.4f}s/S1")

if __name__=="__main__": main()
