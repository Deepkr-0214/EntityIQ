import time
import sys
import gc
from pathlib import Path
import numpy as np
import numba as nb
import psutil

ROOT = Path("D:/Amazon/business_entity_resolution")
sys.path.insert(0, str(ROOT))

from src.pipeline.run_pipeline import stream_records
from src.preprocessing.normalization import normalize_record
from run_test_inference_fast import build_index_and_store
from experiment_matching_model import (
    score_candidate_features, address_rescue_score, make_feature_vector,
    CAP, FEATURES
)
from optimize_matching_experiment import NumpyQueryIndex, query_keys, WEIGHTS
from src.modeling.train import predict_proba
import pickle

def get_rss():
    return psutil.Process().memory_info().rss / 1024 / 1024

print(f"Initial RSS: {get_rss():.1f} MB")

# Load frozen model
with open(ROOT / "output" / "_frozen_lr_model.pkl", "rb") as f:
    saved = pickle.load(f)
    model = saved["model"]
    scaler = saved["scaler"]

print(f"Model loaded. Building France index...")
idx, cids, store, idx_s = build_index_and_store("france")
print(f"Index built: {len(cids)} candidates in {idx_s:.2f}s, RSS: {get_rss():.1f} MB")

# Load 20 S1s
s1s = []
for eid, name, addr, ctry in stream_records(ROOT / "splits" / "test_S1_france.tsv"):
    s1s.append(normalize_record(eid, name, addr, ctry))
    if len(s1s) == 20:
        break

print(f"Loaded {len(s1s)} S1 queries.")

# -------------------------------------------------------------
# 1. Baseline NumpyQueryIndex run
# -------------------------------------------------------------
nqi = NumpyQueryIndex(idx, cids, s1s)
nqi_results = {}
t0_baseline = time.perf_counter()
for s1 in s1s:
    pool, touched = nqi.generate(s1)
    top_pos, ranked = nqi.rank(pool)
    nqi_results[s1["id"]] = (
        top_pos,
        [tuple(int(v) for v in nqi.features[pos].tolist()) for pos in top_pos],
        [int(nqi.features[pos].astype(np.int32) @ WEIGHTS) for pos in top_pos]
    )
    nqi.clear(touched)
t_baseline_cand = time.perf_counter() - t0_baseline
print(f"Baseline NQI candidate gen time (20 S1s): {t_baseline_cand:.4f}s ({t_baseline_cand/20:.4f}s/S1)")

# -------------------------------------------------------------
# 2. Numba-accelerated candidate index
# -------------------------------------------------------------
@nb.njit(fastmath=True)
def _numba_score_and_filter(
    p0, p1, p2_flat, p2_offsets,
    p3_flat, p3_offsets,
    p4_flat, p4_offsets,
    p5, p6_flat, p6_offsets,
    features, touched, pool_buf, scores_buf
):
    num_touched = 0

    # 0: first_token
    for i in range(len(p0)):
        idx = p0[i]
        if features[idx, 0] == 0:
            if (features[idx, 0] | features[idx, 1] | features[idx, 2] |
                features[idx, 3] | features[idx, 4] | features[idx, 5] | features[idx, 6]) == 0:
                touched[num_touched] = idx
                num_touched += 1
            features[idx, 0] = 1

    # 1: name_prefix
    for i in range(len(p1)):
        idx = p1[i]
        if features[idx, 1] == 0:
            if (features[idx, 0] | features[idx, 1] | features[idx, 2] |
                features[idx, 3] | features[idx, 4] | features[idx, 5] | features[idx, 6]) == 0:
                touched[num_touched] = idx
                num_touched += 1
            features[idx, 1] = 1

    # 2: name_tokens
    for t in range(len(p2_offsets) - 1):
        for i in range(p2_offsets[t], p2_offsets[t + 1]):
            idx = p2_flat[i]
            if (features[idx, 0] | features[idx, 1] | features[idx, 2] |
                features[idx, 3] | features[idx, 4] | features[idx, 5] | features[idx, 6]) == 0:
                touched[num_touched] = idx
                num_touched += 1
            features[idx, 2] += 1

    # 3: addr_nums
    for t in range(len(p3_offsets) - 1):
        for i in range(p3_offsets[t], p3_offsets[t + 1]):
            idx = p3_flat[i]
            if (features[idx, 0] | features[idx, 1] | features[idx, 2] |
                features[idx, 3] | features[idx, 4] | features[idx, 5] | features[idx, 6]) == 0:
                touched[num_touched] = idx
                num_touched += 1
            features[idx, 3] += 1

    # 4: addr_tokens
    for t in range(len(p4_offsets) - 1):
        for i in range(p4_offsets[t], p4_offsets[t + 1]):
            idx = p4_flat[i]
            if (features[idx, 0] | features[idx, 1] | features[idx, 2] |
                features[idx, 3] | features[idx, 4] | features[idx, 5] | features[idx, 6]) == 0:
                touched[num_touched] = idx
                num_touched += 1
            features[idx, 4] += 1

    # 5: cskel3
    for i in range(len(p5)):
        idx = p5[i]
        if features[idx, 5] == 0:
            if (features[idx, 0] | features[idx, 1] | features[idx, 2] |
                features[idx, 3] | features[idx, 4] | features[idx, 5] | features[idx, 6]) == 0:
                touched[num_touched] = idx
                num_touched += 1
            features[idx, 5] = 1

    # 6: cngrams
    for t in range(len(p6_offsets) - 1):
        for i in range(p6_offsets[t], p6_offsets[t + 1]):
            idx = p6_flat[i]
            if (features[idx, 0] | features[idx, 1] | features[idx, 2] |
                features[idx, 3] | features[idx, 4] | features[idx, 5] | features[idx, 6]) == 0:
                touched[num_touched] = idx
                num_touched += 1
            features[idx, 6] += 1

    # Filter pool and score
    pool_size = 0
    for i in range(num_touched):
        idx = touched[i]
        f0 = features[idx, 0]
        f1 = features[idx, 1]
        f2 = features[idx, 2]
        f3 = features[idx, 3]
        f4 = features[idx, 4]
        f5 = features[idx, 5]
        f6 = features[idx, 6]

        if (f0 > 0) or (f1 > 0) or (f2 > 0) or (f3 > 0) or (f5 > 0) or (f4 >= 2) or (f6 >= 2):
            score = 8*f0 + 7*f1 + 2*f2 + 8*f3 + 6*f4 + 6*f5 + 1*f6
            pool_buf[pool_size] = idx
            scores_buf[pool_size] = score
            pool_size += 1

    return num_touched, pool_size

@nb.njit(fastmath=True)
def _numba_clear(features, touched, num_touched):
    for i in range(num_touched):
        idx = touched[i]
        features[idx, 0] = 0
        features[idx, 1] = 0
        features[idx, 2] = 0
        features[idx, 3] = 0
        features[idx, 4] = 0
        features[idx, 5] = 0
        features[idx, 6] = 0

_EMPTY_INT32 = np.empty(0, dtype=np.int32)
_EMPTY_OFFSETS = np.array([0], dtype=np.int32)

class NumbaCandidateGenerator:
    def __init__(self, idx, cids):
        self.idx = idx
        self.cids = cids
        self.n = len(cids)
        self.features = np.zeros((self.n, 7), dtype=np.uint16)
        self.touched = np.zeros(self.n, dtype=np.int32)
        self.pool_buf = np.zeros(self.n, dtype=np.int32)
        self.scores_buf = np.zeros(self.n, dtype=np.int32)

    def query(self, s1):
        ft = s1.get("ft")
        p0 = np.frombuffer(self.idx.first_token[ft], dtype=np.int32) if (ft and ft in self.idx.first_token) else _EMPTY_INT32

        npfx = s1.get("npfx")
        p1 = np.frombuffer(self.idx.name_prefix[npfx], dtype=np.int32) if (npfx and npfx in self.idx.name_prefix) else _EMPTY_INT32

        # 2: name_tokens
        ntoks = s1.get("ntoks", ())
        if ntoks:
            p2_list = [np.frombuffer(self.idx.name_tokens[t], dtype=np.int32) for t in ntoks if t in self.idx.name_tokens]
            if p2_list:
                p2_flat = np.concatenate(p2_list)
                lens = [len(p) for p in p2_list]
                p2_offsets = np.empty(len(p2_list) + 1, dtype=np.int32)
                p2_offsets[0] = 0
                np.cumsum(lens, out=p2_offsets[1:])
            else:
                p2_flat, p2_offsets = _EMPTY_INT32, _EMPTY_OFFSETS
        else:
            p2_flat, p2_offsets = _EMPTY_INT32, _EMPTY_OFFSETS

        # 3: addr_nums
        anums = s1.get("anums", ())
        if anums:
            anums_set = set(anums)
            p3_list = [np.frombuffer(self.idx.addr_nums[n], dtype=np.int32) for n in anums_set if n in self.idx.addr_nums]
            if p3_list:
                p3_flat = np.concatenate(p3_list)
                lens = [len(p) for p in p3_list]
                p3_offsets = np.empty(len(p3_list) + 1, dtype=np.int32)
                p3_offsets[0] = 0
                np.cumsum(lens, out=p3_offsets[1:])
            else:
                p3_flat, p3_offsets = _EMPTY_INT32, _EMPTY_OFFSETS
        else:
            p3_flat, p3_offsets = _EMPTY_INT32, _EMPTY_OFFSETS

        # 4: addr_tokens
        atoks = s1.get("atoks", ())
        if len(atoks) >= 2:
            p4_list = [np.frombuffer(self.idx.addr_tokens[t], dtype=np.int32) for t in atoks if t in self.idx.addr_tokens]
            if p4_list:
                p4_flat = np.concatenate(p4_list)
                lens = [len(p) for p in p4_list]
                p4_offsets = np.empty(len(p4_list) + 1, dtype=np.int32)
                p4_offsets[0] = 0
                np.cumsum(lens, out=p4_offsets[1:])
            else:
                p4_flat, p4_offsets = _EMPTY_INT32, _EMPTY_OFFSETS
        else:
            p4_flat, p4_offsets = _EMPTY_INT32, _EMPTY_OFFSETS

        # 5: cskel3
        cs3 = s1.get("cskel3", "")
        p5 = np.frombuffer(self.idx.cskel3[cs3], dtype=np.int32) if (cs3 and len(cs3) >= 3 and cs3 in self.idx.cskel3) else _EMPTY_INT32

        # 6: cngrams
        cngrams = s1.get("cngrams", ())
        if len(cngrams) >= 2:
            p6_list = [np.frombuffer(self.idx.cngrams[g], dtype=np.int32) for g in cngrams if g in self.idx.cngrams]
            if p6_list:
                p6_flat = np.concatenate(p6_list)
                lens = [len(p) for p in p6_list]
                p6_offsets = np.empty(len(p6_list) + 1, dtype=np.int32)
                p6_offsets[0] = 0
                np.cumsum(lens, out=p6_offsets[1:])
            else:
                p6_flat, p6_offsets = _EMPTY_INT32, _EMPTY_OFFSETS
        else:
            p6_flat, p6_offsets = _EMPTY_INT32, _EMPTY_OFFSETS

        num_touched, pool_size = _numba_score_and_filter(
            p0, p1, p2_flat, p2_offsets,
            p3_flat, p3_offsets,
            p4_flat, p4_offsets,
            p5, p6_flat, p6_offsets,
            self.features, self.touched, self.pool_buf, self.scores_buf
        )

        if pool_size == 0:
            _numba_clear(self.features, self.touched, num_touched)
            return []

        # Ranking
        pool = self.pool_buf[:pool_size]
        scores = self.scores_buf[:pool_size]
        take = min(CAP, pool_size)

        if pool_size <= CAP:
            chosen_local = np.arange(pool_size)
        else:
            threshold = np.partition(scores, pool_size - take)[pool_size - take]
            above = np.flatnonzero(scores > threshold)
            ties = np.flatnonzero(scores == threshold)
            remaining = take - above.size
            if remaining:
                import heapq
                tie_positions = pool[ties]
                tie_local = heapq.nsmallest(
                    remaining, range(ties.size),
                    key=lambda i: self.cids[int(tie_positions[i])],
                )
                chosen_local = np.concatenate((above, ties[np.asarray(tie_local, dtype=np.int32)]))
            else:
                chosen_local = above

        selected_positions = pool[chosen_local]
        selected_scores = scores[chosen_local]
        order = sorted(
            range(selected_positions.size),
            key=lambda i: (-int(selected_scores[i]), self.cids[int(selected_positions[i])]),
        )
        ranked = selected_positions[np.asarray(order, dtype=np.int32)]

        top_positions = ranked.tolist()
        result = []
        for rank_idx, pos in enumerate(top_positions, 1):
            blocker = tuple(int(v) for v in self.features[pos].tolist())
            score = int(scores[chosen_local[order[rank_idx - 1]]])
            result.append((pos, blocker, score, rank_idx))

        _numba_clear(self.features, self.touched, num_touched)
        return result

generator = NumbaCandidateGenerator(idx, cids)

# Warmup JIT
generator.query(s1s[0])

# Measure Numba candidate generation
t0_numba = time.perf_counter()
numba_results = {}
for s1 in s1s:
    res = generator.query(s1)
    numba_results[s1["id"]] = res
t_numba_cand = time.perf_counter() - t0_numba
print(f"Numba candidate gen time (20 S1s): {t_numba_cand:.4f}s ({t_numba_cand/20:.4f}s/S1)")
print(f"Cand gen speedup over NQI: {t_baseline_cand / t_numba_cand:.2f}x")

# Check Parity
cand_parity = True
for s1 in s1s:
    sid = s1["id"]
    nqi_top, nqi_blockers, nqi_scores = nqi_results[sid]
    numba_top = [r[0] for r in numba_results[sid]]
    numba_blockers = [r[1] for r in numba_results[sid]]
    numba_scores = [r[2] for r in numba_results[sid]]

    if nqi_top != numba_top:
        print(f"Top-200 mismatch on {sid}: len {len(nqi_top)} vs {len(numba_top)}")
        cand_parity = False
        break
    if nqi_blockers != numba_blockers:
        print(f"Blocker feature mismatch on {sid}")
        cand_parity = False
        break
    if nqi_scores != numba_scores:
        print(f"Score mismatch on {sid}")
        cand_parity = False
        break

print(f"\n==========================================")
print(f"CANDIDATE & BLOCKER PARITY: {'PASS' if cand_parity else 'FAIL'}")
print(f"==========================================")

# Now measure feature generation and LR prediction parity
t0_feat = time.perf_counter()
all_vecs = []
for s1 in s1s:
    sid = s1["id"]
    for pos, blocker, score, rank in numba_results[sid]:
        cand_rec = store.get_rec(pos, cids[pos], "france")
        vec = make_feature_vector(s1, cand_rec, blocker, rank)
        all_vecs.append(vec)
t_feat = time.perf_counter() - t0_feat

t0_lr = time.perf_counter()
X = np.asarray(all_vecs, dtype=np.float32)
probs = predict_proba(model, X, scaler=scaler)
t_lr = time.perf_counter() - t0_lr

matches = (probs >= 0.998722).sum()

print(f"Feature computation time (4000 pairs): {t_feat:.4f}s ({t_feat/20:.4f}s/S1)")
print(f"LR prediction time (4000 pairs): {t_lr:.4f}s ({t_lr/20:.4f}s/S1)")
print(f"Matches predicted at 0.998722: {matches}")
print(f"Total time per S1: {(t_numba_cand + t_feat + t_lr)/20:.4f}s")
print(f"Peak RSS: {get_rss():.1f} MB")
