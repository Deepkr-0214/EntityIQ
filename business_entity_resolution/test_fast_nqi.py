import time
import sys
from pathlib import Path
import numpy as np

ROOT = Path("D:/Amazon/business_entity_resolution")
sys.path.insert(0, str(ROOT))

from src.pipeline.run_pipeline import stream_records
from src.preprocessing.normalization import normalize_record
from run_test_inference_fast import build_index_and_store
from optimize_matching_experiment import NumpyQueryIndex, query_keys, WEIGHTS, CAP

print("Loading index for France...")
idx, cids, store, idx_s = build_index_and_store("france")
print(f"Index built: {len(cids)} candidates in {idx_s:.2f}s")

# Load 20 S1s
s1s = []
for eid, name, addr, ctry in stream_records(ROOT / "splits" / "test_S1_france.tsv"):
    s1s.append(normalize_record(eid, name, addr, ctry))
    if len(s1s) == 20:
        break

print(f"\nLoaded {len(s1s)} S1 queries.")

# Old NQI
nqi = NumpyQueryIndex(idx, cids, s1s)

# Optimized NQI v2
class FastNumpyQueryIndex:
    MAPS = (
        ("first_token", 0, True),   # binary
        ("name_prefix", 1, True),   # binary
        ("name_tokens", 2, False),  # count
        ("addr_nums", 3, False),    # count
        ("addr_tokens", 4, False),  # count
        ("cskel3", 5, True),        # binary
        ("cngrams", 6, False),      # count
    )

    def __init__(self, idx, ids, queries):
        n = len(ids)
        all_keys = {name: set() for name, _, _ in self.MAPS}
        for s1 in queries:
            for name, keys in query_keys(s1).items():
                all_keys[name].update(keys)
        started = time.perf_counter()
        self.postings = {}
        for name, _, _ in self.MAPS:
            source = getattr(idx, name)
            # array('i') has buffer protocol, so np.frombuffer is instantaneous O(1)
            self.postings[name] = {
                key: np.frombuffer(source[key], dtype=np.int32)
                for key in all_keys[name] if key in source
            }
        self.features = np.zeros((n, 7), dtype=np.uint16)
        self.n = n
        self.ids = ids
        self.cache_seconds = time.perf_counter() - started

    def generate_and_rank(self, s1):
        keys = query_keys(s1)
        postings_list = []
        for name, col, is_binary in self.MAPS:
            postmap = self.postings[name]
            for key in keys[name]:
                p = postmap.get(key)
                if p is not None and p.size > 0:
                    postings_list.append(p)
                    if is_binary:
                        self.features[p, col] = 1
                    else:
                        np.add.at(self.features[:, col], p, 1)

        if not postings_list:
            return [], []

        all_p = np.concatenate(postings_list)
        touched = np.unique(all_p)
        f = self.features[touched]
        keep = (f[:, 0] > 0) | (f[:, 1] > 0) | (f[:, 2] > 0) | (f[:, 3] > 0) | (f[:, 5] > 0) | (f[:, 4] >= 2) | (f[:, 6] >= 2)
        pool = touched[keep]
        
        if pool.size == 0:
            self.features[touched] = 0
            return [], []

        # Ranking
        scores = self.features[pool].astype(np.int32) @ WEIGHTS
        take = min(CAP, pool.size)
        if pool.size <= CAP:
            chosen_local = np.arange(pool.size)
        else:
            threshold = np.partition(scores, pool.size - take)[pool.size - take]
            above = np.flatnonzero(scores > threshold)
            ties = np.flatnonzero(scores == threshold)
            remaining = take - above.size
            if remaining:
                import heapq
                tie_positions = pool[ties]
                tie_local = heapq.nsmallest(
                    remaining, range(ties.size),
                    key=lambda i: self.ids[int(tie_positions[i])],
                )
                chosen_local = np.concatenate((above, ties[np.asarray(tie_local, dtype=np.int32)]))
            else:
                chosen_local = above

        selected_positions = pool[chosen_local]
        selected_scores = scores[chosen_local]
        order = sorted(
            range(selected_positions.size),
            key=lambda i: (-int(selected_scores[i]), self.ids[int(selected_positions[i])]),
        )
        ranked = selected_positions[np.asarray(order, dtype=np.int32)]
        
        # Extract blocker features and scores for selected top
        top_positions = ranked.tolist()
        results = []
        for rank_idx, pos in enumerate(top_positions, 1):
            blocker = tuple(int(v) for v in self.features[pos].tolist())
            score = int(scores[chosen_local[order[rank_idx - 1]]])
            results.append((self.ids[pos], blocker, score, rank_idx, pos))

        # Clear features
        self.features[touched] = 0
        return top_positions, results

fast_nqi = FastNumpyQueryIndex(idx, cids, s1s)

# Compare parity and timing on 20 S1s
old_total_s = 0.0
fast_total_s = 0.0
parity_pass = True

for i, s1 in enumerate(s1s):
    # Old
    t0 = time.perf_counter()
    pool, touched = nqi.generate(s1)
    old_top, old_ranked = nqi.rank(pool)
    nqi.clear(touched)
    old_total_s += time.perf_counter() - t0
    
    # Fast
    t0 = time.perf_counter()
    fast_top, fast_results = fast_nqi.generate_and_rank(s1)
    fast_total_s += time.perf_counter() - t0
    
    if old_top != fast_top:
        print(f"MISMATCH on S1 {i} ({s1['id']}):")
        print(f"  Old top len={len(old_top)}, Fast top len={len(fast_top)}")
        print(f"  Old first 5: {old_top[:5]}")
        print(f"  Fast first 5: {fast_top[:5]}")
        parity_pass = False
        break

print(f"\nParity test: {'PASS' if parity_pass else 'FAIL'}")
print(f"Old NQI time (20 S1s): {old_total_s:.4f}s ({old_total_s/20:.4f}s/S1)")
print(f"Fast NQI time (20 S1s): {fast_total_s:.4f}s ({fast_total_s/20:.4f}s/S1)")
print(f"Speedup: {old_total_s / fast_total_s:.2f}x")
