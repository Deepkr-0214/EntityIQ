import time
import sys
from pathlib import Path
import numpy as np

ROOT = Path("D:/Amazon/business_entity_resolution")
sys.path.insert(0, str(ROOT))

from run_test_inference_final import (
    build_country_index, NumbaCandidateGenerator,
    make_feature_vector_optimized, predict_proba, FROZEN_MODEL,
    FROZEN_THRESHOLD, CAP, rss_mb
)
from src.pipeline.run_pipeline import stream_records
from src.preprocessing.normalization import normalize_record
import pickle

print("=== MINI TEST: 20 France S1 records ===")
with open(FROZEN_MODEL, "rb") as f:
    saved = pickle.load(f)
    model = saved["model"]
    scaler = saved["scaler"]

idx, cids, store = build_country_index("france")
generator = NumbaCandidateGenerator(idx, cids)

# Warmup JIT
generator.query({"ft": "test", "npfx": "test", "ntoks": ["test"]})

# Load 20 S1s
s1s = []
for eid, name, addr, ctry in stream_records(ROOT / "splits" / "test_S1_france.tsv"):
    s1s.append(normalize_record(eid, name, addr, ctry))
    if len(s1s) == 20:
        break

t_cand = 0.0
t_feat = 0.0
t_lr   = 0.0

total_pairs = 0
matches_count = 0

t_start = time.perf_counter()

for s1 in s1s:
    sid = s1["id"]
    t0 = time.perf_counter()
    top_cands = generator.query(s1)
    t_cand += time.perf_counter() - t0

    if top_cands:
        cand_ids = [cids[pos] for pos, _, _, _ in top_cands]
        total_pairs += len(cand_ids)

        s1_nl = s1.get("nl", "")
        s1_na = s1.get("na", "")
        s1_al = s1.get("al", "")
        s1_aa = s1.get("aa", "")
        s1_anums = s1.get("anums", [])
        s1_ntoks = s1.get("ntoks", set())
        s1_atoks = s1.get("atoks", set())
        s1_cngrams = s1.get("cngrams", set())

        t0 = time.perf_counter()
        feature_vectors = []
        for pos, blocker, score, rank in top_cands:
            c_nl, c_na, c_al, c_aa, c_anums, c_ntoks, c_atoks, c_cngrams = store.get_cand_data(pos)
            cid = cids[pos]
            vec = make_feature_vector_optimized(
                s1_nl, s1_na, s1_al, s1_aa, s1_anums, s1_ntoks, s1_atoks, s1_cngrams,
                c_nl, c_na, c_al, c_aa, c_anums, c_ntoks, c_atoks, c_cngrams, cid,
                blocker, rank
            )
            feature_vectors.append(vec)
        t_feat += time.perf_counter() - t0

        t0 = time.perf_counter()
        X = np.asarray(feature_vectors, dtype=np.float32)
        probs = predict_proba(model, X, scaler=scaler)
        matched_indices = np.flatnonzero(probs >= FROZEN_THRESHOLD)
        matches_count += len(matched_indices)
        t_lr += time.perf_counter() - t0

t_total = time.perf_counter() - t_start
avg_s_per_s1 = t_total / len(s1s)
total_s1 = 1_732_544

print(f"\nResults for {len(s1s)} S1 queries:")
print(f"  - candidate-generation time : {t_cand:.4f}s ({t_cand/len(s1s)*1000:.2f} ms/S1)")
print(f"  - feature time              : {t_feat:.4f}s ({t_feat/len(s1s)*1000:.2f} ms/S1)")
print(f"  - prediction time           : {t_lr:.4f}s ({t_lr/len(s1s)*1000:.2f} ms/S1)")
print(f"  - average seconds/S1        : {avg_s_per_s1:.4f}s")
print(f"  - Total candidate pairs     : {total_pairs} (avg {total_pairs/len(s1s):.1f}/S1)")
print(f"  - Matches found at threshold: {matches_count}")
print(f"  - RAM                       : {rss_mb():.1f} MB")
print(f"  - estimated full-test runtime: {avg_s_per_s1 * total_s1 / 3600:.2f} hours (excluding one-time index build)")
