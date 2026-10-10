import time
import sys
from pathlib import Path

ROOT = Path("D:/Amazon/business_entity_resolution")
sys.path.insert(0, str(ROOT))

from src.pipeline.run_pipeline import stream_records
from src.preprocessing.normalization import normalize_record
from run_test_inference_fast import build_index_and_store, ArrayBlockingIndex
from experiment_matching_model import score_candidate_features, address_rescue_score
from optimize_matching_experiment import NumpyQueryIndex, query_keys

print("Loading index for France...")
idx, cids, store, idx_s = build_index_and_store("france")
print(f"Index built: {len(cids)} candidates in {idx_s:.2f}s")

# Load 5 S1s
s1s = []
for eid, name, addr, ctry in stream_records(ROOT / "splits" / "test_S1_france.tsv"):
    s1s.append(normalize_record(eid, name, addr, ctry))
    if len(s1s) == 5:
        break

print(f"\nLoaded {len(s1s)} S1 queries.")

# Test 1: score_candidate_features (from experiment_matching_model / run_test_inference_fast)
for i, s1 in enumerate(s1s):
    t0 = time.perf_counter()
    f, pool = score_candidate_features(s1, idx)
    t_feat = time.perf_counter() - t0
    
    t0 = time.perf_counter()
    scores = {p: address_rescue_score(f[p]) for p in pool}
    t_score = time.perf_counter() - t0
    
    print(f"S1 {i} ({s1['id']}): |pool|={len(pool)}, |f|={len(f)}, score_cand_feat={t_feat:.4f}s, scoring={t_score:.4f}s")
    # Let's inspect which keys generated huge postings
    ft = s1.get("ft")
    npfx = s1.get("npfx")
    ntoks = s1.get("ntoks", ())
    anums = set(s1.get("anums", ()))
    atoks = s1.get("atoks", ())
    cs3 = s1.get("cskel3", "")
    cngrams = s1.get("cngrams", ())
    
    print(f"   ft postings: {len(idx.first_token.get(ft, [])) if ft else 0}")
    print(f"   npfx postings: {len(idx.name_prefix.get(npfx, [])) if npfx else 0}")
    print(f"   ntoks postings: {[len(idx.name_tokens.get(t, [])) for t in ntoks]}")
    print(f"   anums postings: {[len(idx.addr_nums.get(n, [])) for n in anums]}")
    if len(atoks) >= 2:
        print(f"   atoks postings: {[len(idx.addr_tokens.get(t, [])) for t in atoks]}")
    if cs3 and len(cs3) >= 3:
        print(f"   cs3 postings: {len(idx.cskel3.get(cs3, []))}")
    if len(cngrams) >= 2:
        print(f"   cngrams postings: {[len(idx.cngrams.get(g, [])) for g in cngrams]}")

print("\nNow testing NumpyQueryIndex with these 5 queries...")
t0 = time.perf_counter()
nqi = NumpyQueryIndex(idx, cids, s1s)
print(f"NumpyQueryIndex built in {time.perf_counter()-t0:.4f}s")
for i, s1 in enumerate(s1s):
    t0 = time.perf_counter()
    pool, touched = nqi.generate(s1)
    t_gen = time.perf_counter() - t0
    
    t0 = time.perf_counter()
    top_pos, ranked = nqi.rank(pool)
    t_rank = time.perf_counter() - t0
    
    nqi.clear(touched)
    print(f"NQI S1 {i}: |pool|={len(pool)}, gen={t_gen:.4f}s, rank={t_rank:.4f}s")
