import time
import numpy as np
import numba as nb

# Numba kernel for candidate scoring and pool filtering
@nb.njit(fastmath=True)
def score_and_filter_candidates(
    p0, p1, p2_flat, p2_offsets,
    p3_flat, p3_offsets,
    p4_flat, p4_offsets,
    p5, p6_flat, p6_offsets,
    features, touched,
    pool_buf, scores_buf
):
    # features has shape (N, 7)
    # touched tracks which indices were modified
    num_touched = 0
    
    # 0: first_token (binary)
    for i in range(len(p0)):
        idx = p0[i]
        if features[idx, 0] == 0:
            if (features[idx, 0] | features[idx, 1] | features[idx, 2] |
                features[idx, 3] | features[idx, 4] | features[idx, 5] | features[idx, 6]) == 0:
                touched[num_touched] = idx
                num_touched += 1
            features[idx, 0] = 1

    # 1: name_prefix (binary)
    for i in range(len(p1)):
        idx = p1[i]
        if features[idx, 1] == 0:
            if (features[idx, 0] | features[idx, 1] | features[idx, 2] |
                features[idx, 3] | features[idx, 4] | features[idx, 5] | features[idx, 6]) == 0:
                touched[num_touched] = idx
                num_touched += 1
            features[idx, 1] = 1

    # 2: name_tokens (count)
    for t in range(len(p2_offsets) - 1):
        start = p2_offsets[t]
        end = p2_offsets[t + 1]
        for i in range(start, end):
            idx = p2_flat[i]
            if (features[idx, 0] | features[idx, 1] | features[idx, 2] |
                features[idx, 3] | features[idx, 4] | features[idx, 5] | features[idx, 6]) == 0:
                touched[num_touched] = idx
                num_touched += 1
            features[idx, 2] += 1

    # 3: addr_nums (count)
    for t in range(len(p3_offsets) - 1):
        start = p3_offsets[t]
        end = p3_offsets[t + 1]
        for i in range(start, end):
            idx = p3_flat[i]
            if (features[idx, 0] | features[idx, 1] | features[idx, 2] |
                features[idx, 3] | features[idx, 4] | features[idx, 5] | features[idx, 6]) == 0:
                touched[num_touched] = idx
                num_touched += 1
            features[idx, 3] += 1

    # 4: addr_tokens (count)
    for t in range(len(p4_offsets) - 1):
        start = p4_offsets[t]
        end = p4_offsets[t + 1]
        for i in range(start, end):
            idx = p4_flat[i]
            if (features[idx, 0] | features[idx, 1] | features[idx, 2] |
                features[idx, 3] | features[idx, 4] | features[idx, 5] | features[idx, 6]) == 0:
                touched[num_touched] = idx
                num_touched += 1
            features[idx, 4] += 1

    # 5: cskel3 (binary)
    for i in range(len(p5)):
        idx = p5[i]
        if features[idx, 5] == 0:
            if (features[idx, 0] | features[idx, 1] | features[idx, 2] |
                features[idx, 3] | features[idx, 4] | features[idx, 5] | features[idx, 6]) == 0:
                touched[num_touched] = idx
                num_touched += 1
            features[idx, 5] = 1

    # 6: cngrams (count)
    for t in range(len(p6_offsets) - 1):
        start = p6_offsets[t]
        end = p6_offsets[t + 1]
        for i in range(start, end):
            idx = p6_flat[i]
            if (features[idx, 0] | features[idx, 1] | features[idx, 2] |
                features[idx, 3] | features[idx, 4] | features[idx, 5] | features[idx, 6]) == 0:
                touched[num_touched] = idx
                num_touched += 1
            features[idx, 6] += 1

    # Pool filter and scoring
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

    # Reset touched in features
    for i in range(num_touched):
        idx = touched[i]
        features[idx, 0] = 0
        features[idx, 1] = 0
        features[idx, 2] = 0
        features[idx, 3] = 0
        features[idx, 4] = 0
        features[idx, 5] = 0
        features[idx, 6] = 0

    return num_touched, pool_size

# Benchmark
N = 1434993
features = np.zeros((N, 7), dtype=np.uint16)
touched = np.zeros(N, dtype=np.int32)
pool_buf = np.zeros(N, dtype=np.int32)
scores_buf = np.zeros(N, dtype=np.int32)

# Mock postings simulating 400,000 total touched entries
p0 = np.random.randint(0, N, size=5000, dtype=np.int32)
p1 = np.random.randint(0, N, size=5000, dtype=np.int32)
p2_flat = np.random.randint(0, N, size=50000, dtype=np.int32)
p2_offsets = np.array([0, 20000, 50000], dtype=np.int32)
p3_flat = np.empty(0, dtype=np.int32)
p3_offsets = np.array([0], dtype=np.int32)
p4_flat = np.random.randint(0, N, size=200000, dtype=np.int32)
p4_offsets = np.array([0, 100000, 200000], dtype=np.int32)
p5 = np.random.randint(0, N, size=2000, dtype=np.int32)
p6_flat = np.random.randint(0, N, size=100000, dtype=np.int32)
p6_offsets = np.array([0, 50000, 100000], dtype=np.int32)

# Warmup compile
score_and_filter_candidates(
    p0, p1, p2_flat, p2_offsets,
    p3_flat, p3_offsets,
    p4_flat, p4_offsets,
    p5, p6_flat, p6_offsets,
    features, touched, pool_buf, scores_buf
)

# Measure
times = []
for _ in range(50):
    t0 = time.perf_counter()
    num_t, p_size = score_and_filter_candidates(
        p0, p1, p2_flat, p2_offsets,
        p3_flat, p3_offsets,
        p4_flat, p4_offsets,
        p5, p6_flat, p6_offsets,
        features, touched, pool_buf, scores_buf
    )
    times.append(time.perf_counter() - t0)

print(f"Numba execution time: {np.mean(times)*1000:.3f} ms per query! (median={np.median(times)*1000:.3f} ms)")
print(f"Num touched: {num_t}, Pool size: {p_size}")
