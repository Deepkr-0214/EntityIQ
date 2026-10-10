#!/usr/bin/env python3
"""
Business Entity Resolution - Full End-to-End Pipeline
=====================================================
Memory-efficient, country-partitioned, high-recall blocking and rank-aware matching.
Designed to run stably within memory constraints and optimize macro F0.5.
"""

import gc
import os
import sys
import time
import pickle
import re
from collections import defaultdict
from pathlib import Path

# Configure UTF-8 output streams safely for Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

import numpy as np
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Paths
SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent           # business_entity_resolution/
RESOURCE     = PROJECT_ROOT.parent / "student_resource"
TRAIN_DIR    = RESOURCE / "dataset" / "train"
TEST_DIR     = RESOURCE / "dataset" / "test"
OUTPUT_DIR   = PROJECT_ROOT / "output"
MODELS_DIR   = PROJECT_ROOT / "models"
REPORTS_DIR  = PROJECT_ROOT / "reports"
SPLIT_DIR    = PROJECT_ROOT / "splits"            # country-split files

for d in (OUTPUT_DIR, MODELS_DIR, REPORTS_DIR, SPLIT_DIR):
    d.mkdir(parents=True, exist_ok=True)

SEED = 42
TRAIN_SAMPLE_PER_CTRY = 50_000   # S1 entities per country for training
VAL_SAMPLE_PER_CTRY   = 15_000   # S1 entities per country for validation

sys.path.insert(0, str(PROJECT_ROOT))
from src.preprocessing.normalization import normalize_record
from src.features.similarity_features import compute_pairwise_features, FEATURE_COLUMNS
from src.evaluation.metrics import macro_f05


# ==============================================================================
#  Phase 0 - Split Source Files by Country
# ==============================================================================

def _country_split_path(split: str, source: str, country: str) -> Path:
    safe = country.strip().lower().replace(' ', '_')
    return SPLIT_DIR / f"{split}_{source}_{safe}.tsv"


def split_file_by_country(filepath: Path, split: str, source: str):
    """Read filepath line-by-line and stream into per-country subfiles."""
    countries_seen = set()
    handles = {}
    try:
        with open(filepath, 'r', encoding='utf-8', errors='replace') as fin:
            header = fin.readline()
            for line in fin:
                parts = line.split('\t')
                ctry = parts[-1].strip().lower() if len(parts) >= 4 else 'unknown'
                if ctry not in handles:
                    p = _country_split_path(split, source, ctry)
                    handles[ctry] = open(p, 'w', encoding='utf-8')
                    handles[ctry].write(header)
                    countries_seen.add(ctry)
                handles[ctry].write(line)
    finally:
        for h in handles.values():
            h.close()
    return countries_seen


def split_gt_by_country(gt_path: Path, s1_country_map: dict):
    """Split ground-truth into per-country files using pre-built S1->country map."""
    handles = {}
    try:
        with open(gt_path, 'r', encoding='utf-8', errors='replace') as fin:
            header = fin.readline()
            for line in fin:
                s1_id = line.split('\t', 1)[0].strip()
                ctry = s1_country_map.get(s1_id, 'unknown')
                if ctry not in handles:
                    p = SPLIT_DIR / f"train_GT_{ctry}.tsv"
                    handles[ctry] = open(p, 'w', encoding='utf-8')
                    handles[ctry].write(header)
                handles[ctry].write(line)
    finally:
        for h in handles.values():
            h.close()


def build_s1_country_map(s1_path: Path) -> dict:
    """Stream S1 to build {entity_id -> country_lower}."""
    m = {}
    with open(s1_path, 'r', encoding='utf-8', errors='replace') as f:
        f.readline()
        for line in f:
            parts = line.split('\t')
            if len(parts) >= 4:
                m[parts[0].strip()] = parts[-1].strip().lower()
    return m


def phase0_split_all():
    """Split train and test files by country."""
    print("=" * 70)
    print("PHASE 0: Splitting source files by country on disk")
    print("=" * 70)

    train_countries = set()
    for fname, src in [("train_source1.tsv", "S1"),
                       ("train_source2.tsv", "S2"),
                       ("train_source3.tsv", "S3")]:
        p = TRAIN_DIR / fname
        print(f"  Splitting {fname}...", end=" ", flush=True)
        t0 = time.time()
        cs = split_file_by_country(p, "train", src)
        train_countries |= cs
        print(f"done ({time.time()-t0:.1f}s) -> {cs}")

    print("  Building S1->country map...", end=" ", flush=True)
    s1_ctry = build_s1_country_map(TRAIN_DIR / "train_source1.tsv")
    print(f"{len(s1_ctry):,} entities")
    print("  Splitting ground truth...", end=" ", flush=True)
    split_gt_by_country(TRAIN_DIR / "train_ground_truth.tsv", s1_ctry)
    print("done")
    del s1_ctry; gc.collect()

    test_countries = set()
    for fname, src in [("test_source1.tsv", "S1"),
                       ("test_source2.tsv", "S2"),
                       ("test_source3.tsv", "S3")]:
        p = TEST_DIR / fname
        print(f"  Splitting {fname}...", end=" ", flush=True)
        t0 = time.time()
        cs = split_file_by_country(p, "test", src)
        test_countries |= cs
        print(f"done ({time.time()-t0:.1f}s) -> {cs}")

    print(f"\n  Train countries: {sorted(train_countries)}")
    print(f"  Test  countries: {sorted(test_countries)}")
    return sorted(train_countries), sorted(test_countries)


# ==============================================================================
#  Stream & Load Helpers
# ==============================================================================

def stream_records(path: Path):
    """Yield (entity_id, name, address, country) tuples."""
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        f.readline()  # skip header
        for line in f:
            parts = line.rstrip('\n').split('\t')
            eid  = parts[0].strip() if len(parts) > 0 else ''
            name = parts[1].strip() if len(parts) > 1 else ''
            addr = parts[2].strip() if len(parts) > 2 else ''
            ctry = parts[3].strip() if len(parts) > 3 else ''
            yield eid, name, addr, ctry


def load_records_dict(path: Path, subset_ids: set = None) -> dict:
    """Load a TSV into {entity_id -> normalize_record(...)} dict."""
    recs = {}
    for eid, name, addr, ctry in stream_records(path):
        if subset_ids is not None and eid not in subset_ids:
            continue
        recs[eid] = normalize_record(eid, name, addr, ctry)
    return recs


def load_ground_truth(path: Path) -> dict:
    """Load GT file -> {s1_id: set_of_matched_ids}."""
    gt = {}
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        f.readline()
        for line in f:
            parts = line.rstrip('\n').split('\t')
            s1_id = parts[0].strip()
            matched_str = parts[1].strip() if len(parts) > 1 else ''
            if matched_str:
                gt[s1_id] = set(m.strip() for m in matched_str.split(',') if m.strip())
            else:
                gt[s1_id] = set()
    return gt


# ==============================================================================
#  Compact Inverted Index & Blocking
# ==============================================================================

class CompactBlockingIndex:
    """
    Ultra-compact inverted index storing integer pointers into candidate records list.
    Includes consonant skeleton blocking for cross-script (Indic/Latin) matching.
    """
    def __init__(self):
        self.first_token = defaultdict(list)
        self.name_prefix = defaultdict(list)
        self.name_tokens = defaultdict(list)
        self.addr_nums   = defaultdict(list)
        self.addr_tokens = defaultdict(list)
        self.cskel3      = defaultdict(list)   # 3-char consonant skeleton prefix
        self.cngrams     = defaultdict(list)   # consonant trigrams

    def add(self, idx: int, rec: dict):
        if rec['ft']:
            self.first_token[rec['ft']].append(idx)
        if rec['npfx']:
            self.name_prefix[rec['npfx']].append(idx)
        for t in rec.get('ntoks', set()):
            self.name_tokens[t].append(idx)
        for n in rec.get('anums', []):
            self.addr_nums[n].append(idx)
        for t in rec.get('atoks', set()):
            self.addr_tokens[t].append(idx)
        # Consonant skeleton blocking for cross-script matching
        cs3 = rec.get('cskel3', '')
        if cs3 and len(cs3) >= 3:
            self.cskel3[cs3].append(idx)
        for cng in rec.get('cngrams', set()):
            self.cngrams[cng].append(idx)

    def query(self, rec: dict, max_cands: int = 200) -> set:
        """Query candidate record indices for a given S1 record."""
        cand_indices = set()

        # Strategy 1: First token
        ft = rec.get('ft', '')
        if ft and ft in self.first_token:
            cand_indices.update(self.first_token[ft])

        # Strategy 2: 4-char name prefix
        npfx = rec.get('npfx', '')
        if npfx and npfx in self.name_prefix:
            cand_indices.update(self.name_prefix[npfx])

        # Strategy 3: Any shared significant name token
        for t in rec.get('ntoks', set()):
            if t in self.name_tokens:
                cand_indices.update(self.name_tokens[t])

        # Strategy 4: Numeric tokens >= 4 digits (PIN / ZIP codes)
        for n in rec.get('anums', []):
            if n in self.addr_nums:
                cand_indices.update(self.addr_nums[n])

        # Strategy 5: Shared significant address tokens (>= 2 matches)
        atoks = rec.get('atoks', set())
        if len(atoks) >= 2:
            counts = defaultdict(int)
            for t in atoks:
                if t in self.addr_tokens:
                    for i in self.addr_tokens[t]:
                        counts[i] += 1
            cand_indices.update(i for i, c in counts.items() if c >= 2)

        # Strategy 6: Consonant skeleton prefix (3 chars) - phonetic blocking
        cs3 = rec.get('cskel3', '')
        if cs3 and len(cs3) >= 3 and cs3 in self.cskel3:
            cand_indices.update(self.cskel3[cs3])

        # Strategy 7: Consonant n-gram overlap (>= 2 shared trigrams)
        cngrams = rec.get('cngrams', set())
        if len(cngrams) >= 2:
            counts = defaultdict(int)
            for cng in cngrams:
                if cng in self.cngrams:
                    for i in self.cngrams[cng]:
                        counts[i] += 1
            cand_indices.update(i for i, c in counts.items() if c >= 2)

        if len(cand_indices) > max_cands:
            cand_indices = set(list(cand_indices)[:max_cands])
        return cand_indices


def build_candidate_store_and_index(s2_path: Path, s3_path: Path):
    """
    Load S2 and S3 records into a compact list and build inverted index.
    Returns:
      cand_records: list of normalized record dicts
      cand_id_map: dict mapping cand_id -> int index in cand_records
      idx: CompactBlockingIndex
    """
    cand_records = []
    cand_id_map = {}
    idx = CompactBlockingIndex()

    for path in (s2_path, s3_path):
        if not path.exists():
            continue
        for eid, name, addr, ctry in stream_records(path):
            if eid in cand_id_map:
                continue
            rec = normalize_record(eid, name, addr, ctry)
            pos = len(cand_records)
            cand_records.append(rec)
            cand_id_map[eid] = pos
            idx.add(pos, rec)

    return cand_records, cand_id_map, idx


def blocking_recall(candidates: dict, gt: dict):
    """Fraction of true matches surviving candidate generation."""
    total = found = 0
    for s1_id, true_set in gt.items():
        if not true_set:
            continue
        cands = candidates.get(s1_id, set())
        for m in true_set:
            total += 1
            if m in cands:
                found += 1
    return (found / total if total else 1.0), found, total


# ==============================================================================
#  Feature Engineering
# ==============================================================================

def add_rank_group_features(rows: list) -> list:
    """Compute relative rank, group aggregates, and score gaps."""
    if not rows:
        return rows

    groups = defaultdict(list)
    for r in rows:
        groups[r['s1_id']].append(r)

    for s1_id, grp in groups.items():
        n = len(grp)
        wratios = sorted([r.get('name_wratio', 0.0) for r in grp], reverse=True)
        addr_sets = sorted([r.get('addr_token_set', 0.0) for r in grp], reverse=True)
        mx_wr = wratios[0] if wratios else 0.0
        mn_wr = float(np.mean(wratios)) if wratios else 0.0
        mx_at = addr_sets[0] if addr_sets else 0.0
        mn_at = float(np.mean(addr_sets)) if addr_sets else 0.0

        grp.sort(key=lambda r: r.get('name_wratio', 0.0), reverse=True)
        for rank, r in enumerate(grp, 1):
            r['cand_rank_name_wratio']    = rank
            r['cand_rank_addr_token_set'] = 0
            r['group_max_name_wratio']    = mx_wr
            r['group_mean_name_wratio']   = mn_wr
            r['group_max_addr_token_set'] = mx_at
            r['group_mean_addr_token_set']= mn_at
            r['group_n_candidates']       = n
            if rank < n:
                gap = r.get('name_wratio', 0.0) - grp[rank].get('name_wratio', 0.0)
            else:
                gap = r.get('name_wratio', 0.0)
            r['score_gap_to_next'] = gap

        grp.sort(key=lambda r: r.get('addr_token_set', 0.0), reverse=True)
        for rank, r in enumerate(grp, 1):
            r['cand_rank_addr_token_set'] = rank

    return rows


RANK_FEATURES = [
    'cand_rank_name_wratio', 'cand_rank_addr_token_set',
    'group_max_name_wratio', 'group_mean_name_wratio',
    'group_max_addr_token_set', 'group_mean_addr_token_set',
    'group_n_candidates', 'score_gap_to_next',
]
ALL_FEATURES = FEATURE_COLUMNS + RANK_FEATURES


def featurize_s1_batch(s1_batch: list, cand_records: list, cand_id_map: dict,
                       idx: CompactBlockingIndex, gt_dict: dict = None,
                       max_cands: int = 200):
    """
    Featurize a batch of S1 records against the candidate index.
    Returns:
      rows: list of feature dicts
      candidates_map: dict {s1_id: set of candidate string IDs}
    """
    rows = []
    candidates_map = {}

    for s1_rec in s1_batch:
        s1_id = s1_rec['id']
        cand_int_indices = idx.query(s1_rec, max_cands=max_cands)
        cand_str_ids = {cand_records[i]['id'] for i in cand_int_indices}

        true_matches = gt_dict.get(s1_id, set()) if gt_dict else set()
        if gt_dict is not None:
            for tm in true_matches:
                if tm in cand_id_map:
                    cand_int_indices.add(cand_id_map[tm])
                    cand_str_ids.add(tm)

        candidates_map[s1_id] = cand_str_ids

        for ci in cand_int_indices:
            crec = cand_records[ci]
            cid = crec['id']
            feats = compute_pairwise_features(s1_rec, crec)
            feats['s1_id']   = s1_id
            feats['cand_id'] = cid
            if gt_dict is not None:
                feats['label'] = 1 if cid in true_matches else 0
            rows.append(feats)

    rows = add_rank_group_features(rows)
    return rows, candidates_map


# ==============================================================================
#  Model Helpers
# ==============================================================================

def rows_to_Xy(rows, feature_cols):
    """Convert list-of-dicts to (X, y, s1_ids, cand_ids)."""
    X = np.array([[r.get(c, 0.0) for c in feature_cols] for r in rows], dtype=np.float32)
    y = np.array([r.get('label', -1) for r in rows], dtype=np.int8)
    s1_ids   = [r['s1_id']   for r in rows]
    cand_ids = [r['cand_id'] for r in rows]
    return X, y, s1_ids, cand_ids


def train_lr(X, y, feat_names):
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    model = LogisticRegression(max_iter=1000, random_state=SEED,
                               class_weight='balanced', C=1.0)
    model.fit(Xs, y)
    imp = sorted(zip(feat_names, model.coef_[0]), key=lambda x: abs(x[1]), reverse=True)
    print("  Logistic Regression Top Features:")
    for n, c in imp[:8]:
        print(f"    {n:32s} {c:+.4f}")
    return model, scaler


def train_lgbm(X, y, Xv=None, yv=None, feat_names=None):
    params = dict(
        objective='binary', metric='binary_logloss', verbosity=-1,
        seed=SEED, n_estimators=600, learning_rate=0.05,
        num_leaves=63, max_depth=7, min_child_samples=50,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0, is_unbalance=True,
    )
    cbs = [lgb.log_evaluation(100)]
    if Xv is not None:
        cbs.append(lgb.early_stopping(50))
    m = lgb.LGBMClassifier(**params)
    m.fit(X, y, eval_set=[(Xv, yv)] if Xv is not None else None, callbacks=cbs)
    if feat_names:
        imp = sorted(zip(feat_names, m.feature_importances_), key=lambda x: x[1], reverse=True)
        print("\n  LightGBM Top Feature Importances:")
        for n, v in imp[:10]:
            print(f"    {n:32s} {v:8d}")
    return m


def predict_probs(model, X, scaler=None):
    if scaler is not None:
        X = scaler.transform(X)
    return model.predict_proba(X)[:, 1]


# ==============================================================================
#  Rank-Aware Threshold Optimization
# ==============================================================================

def apply_rank_threshold(s1_ids, cand_ids, probs, tau1, gap_thr, all_s1_set):
    """
    Rank-aware thresholding:
      1. S1 with no candidates or top candidate prob < tau1 -> singleton (empty set)
      2. Top candidate >= tau1 is accepted as match 1.
      3. Additional matches accepted only if prob >= tau1 AND previous_prob - prob <= gap_thr.
    """
    groups = defaultdict(list)
    for s, c, p in zip(s1_ids, cand_ids, probs):
        groups[s].append((c, float(p)))

    preds = {}
    for s1 in all_s1_set:
        grp = groups.get(s1, [])
        if not grp:
            preds[s1] = set()
            continue

        grp.sort(key=lambda x: x[1], reverse=True)
        matched = set()
        if grp[0][1] >= tau1:
            matched.add(grp[0][0])
            for i in range(1, len(grp)):
                cid, prob = grp[i]
                if prob < tau1:
                    break
                if (grp[i-1][1] - prob) > gap_thr:
                    break
                matched.add(cid)
        preds[s1] = matched

    return preds


def sweep_thresholds(s1_ids, cand_ids, probs, gt, all_s1_set):
    """Grid search tau1 and gap thresholds directly for macro F0.5."""
    best_f, best_p = -1.0, {}
    results = []
    for tau1 in np.arange(0.30, 0.91, 0.05):
        for gap in np.arange(0.05, 0.41, 0.05):
            preds = apply_rank_threshold(s1_ids, cand_ids, probs, tau1, gap, all_s1_set)
            m = macro_f05(preds, gt)
            results.append((tau1, gap, m))
            if m['f05_macro'] > best_f:
                best_f = m['f05_macro']
                best_p = {'tau1': float(tau1), 'gap': float(gap)}

    results.sort(key=lambda x: x[2]['f05_macro'], reverse=True)
    print(f"\n  {'tau1':>6} {'gap':>6} {'F0.5':>8} {'Prec':>8} {'Rec':>8} "
          f"{'Sing':>8} {'1-M':>8} {'N-M':>8}")
    print("  " + "-" * 66)
    for tau1, gap, m in results[:10]:
        print(f"  {tau1:6.2f} {gap:6.2f} {m['f05_macro']:8.4f} "
              f"{m['precision_macro']:8.4f} {m['recall_macro']:8.4f} "
              f"{m['singleton_f05']:8.4f} {m['single_match_f05']:8.4f} "
              f"{m['multi_match_f05']:8.4f}")
    return best_p, best_f, results[0][2]


# ==============================================================================
#  Master Execution Pipeline
# ==============================================================================

def run():
    wall_t0 = time.time()
    print("=" * 70)
    print("  BUSINESS ENTITY RESOLUTION - FULL ML PIPELINE")
    print("=" * 70)

    # Phase 0: Country Splitting
    split_done = (_country_split_path("train", "S1", "us").exists() and
                  _country_split_path("test", "S1", "us").exists())
    if split_done:
        train_countries = sorted({
            p.stem.split('_')[-1] for p in SPLIT_DIR.glob("train_S1_*.tsv")
        })
        test_countries = sorted({
            p.stem.split('_')[-1] for p in SPLIT_DIR.glob("test_S1_*.tsv")
        })
        print(f"  Existing country splits found: train={train_countries}, test={test_countries}")
    else:
        train_countries, test_countries = phase0_split_all()

    # Load Full Ground Truth
    print("\n--- Loading Ground Truth ---")
    full_gt = {}
    for ctry in train_countries:
        gt_p = SPLIT_DIR / f"train_GT_{ctry}.tsv"
        if gt_p.exists():
            g = load_ground_truth(gt_p)
            full_gt.update(g)
            print(f"  {ctry.upper()}: {len(g):,} entities")

    # Sample S1 entities for Training and Validation
    print("\n--- Sampling Training & Validation Entities ---")
    np.random.seed(SEED)
    train_s1_sample = {}
    val_s1_sample = {}

    for ctry in train_countries:
        s1_path = _country_split_path("train", "S1", ctry)
        ctry_s1_ids = []
        with open(s1_path, 'r', encoding='utf-8', errors='replace') as f:
            f.readline()
            for line in f:
                eid = line.split('\t', 1)[0].strip()
                if eid:
                    ctry_s1_ids.append(eid)

        np.random.shuffle(ctry_s1_ids)
        t_ids = set(ctry_s1_ids[:TRAIN_SAMPLE_PER_CTRY])
        v_ids = set(ctry_s1_ids[TRAIN_SAMPLE_PER_CTRY:TRAIN_SAMPLE_PER_CTRY + VAL_SAMPLE_PER_CTRY])

        train_s1_sample[ctry] = t_ids
        val_s1_sample[ctry]   = v_ids
        print(f"  {ctry.upper()}: {len(t_ids):,} train S1, {len(v_ids):,} val S1 (from {len(ctry_s1_ids):,} total)")

    # Feature Extraction for Training & Validation
    print("\n--- Feature Extraction (Training & Validation) ---")
    train_rows, val_rows = [], []
    blocking_stats = {}

    for ctry in train_countries:
        print(f"\n  Processing country: {ctry.upper()}")
        s2_path = _country_split_path("train", "S2", ctry)
        s3_path = _country_split_path("train", "S3", ctry)
        s1_path = _country_split_path("train", "S1", ctry)

        print(f"    Building candidate store & index from S2 + S3...", end=" ", flush=True)
        t0 = time.time()
        cand_records, cand_id_map, idx = build_candidate_store_and_index(s2_path, s3_path)
        print(f"{len(cand_records):,} records ({time.time()-t0:.1f}s)")

        # Load train S1 subset
        t_needed = train_s1_sample[ctry]
        v_needed = val_s1_sample[ctry]
        s1_train_recs = []
        s1_val_recs   = []

        for eid, name, addr, c in stream_records(s1_path):
            if eid in t_needed:
                s1_train_recs.append(normalize_record(eid, name, addr, c))
            elif eid in v_needed:
                s1_val_recs.append(normalize_record(eid, name, addr, c))

        print(f"    Featurizing train S1 ({len(s1_train_recs):,} entities)...", end=" ", flush=True)
        t0 = time.time()
        sub_gt_train = {k: full_gt[k] for k in t_needed if k in full_gt}
        t_rows, t_cands = featurize_s1_batch(s1_train_recs, cand_records, cand_id_map, idx, gt_dict=sub_gt_train)
        train_rows.extend(t_rows)
        rec_t, f_t, tot_t = blocking_recall(t_cands, sub_gt_train)
        print(f"{len(t_rows):,} pairs ({time.time()-t0:.1f}s), recall: {rec_t:.4f}")

        print(f"    Featurizing val S1 ({len(s1_val_recs):,} entities)...", end=" ", flush=True)
        t0 = time.time()
        sub_gt_val = {k: full_gt[k] for k in v_needed if k in full_gt}
        v_rows, v_cands = featurize_s1_batch(s1_val_recs, cand_records, cand_id_map, idx, gt_dict=sub_gt_val)
        val_rows.extend(v_rows)
        rec_v, f_v, tot_v = blocking_recall(v_cands, sub_gt_val)
        print(f"{len(v_rows):,} pairs ({time.time()-t0:.1f}s), recall: {rec_v:.4f}")

        blocking_stats[ctry] = {'train_recall': rec_t, 'val_recall': rec_v}
        del cand_records, cand_id_map, idx, s1_train_recs, s1_val_recs
        gc.collect()

    pos_t = sum(1 for r in train_rows if r.get('label') == 1)
    pos_v = sum(1 for r in val_rows if r.get('label') == 1)
    print(f"\n  Accumulated Train Pairs: {len(train_rows):,} (pos={pos_t:,}, neg={len(train_rows)-pos_t:,})")
    print(f"  Accumulated Val   Pairs: {len(val_rows):,} (pos={pos_v:,}, neg={len(val_rows)-pos_v:,})")

    # Model Training
    print("\n--- Model Training ---")
    feature_cols = [c for c in ALL_FEATURES if c in train_rows[0]]
    print(f"  Using {len(feature_cols)} features: {feature_cols}")

    Xt, yt, _, _ = rows_to_Xy(train_rows, feature_cols)
    Xv, yv, val_s1s, val_cids = rows_to_Xy(val_rows, feature_cols)

    print("\n  [1/2] Training Baseline Logistic Regression...")
    lr_model, lr_scaler = train_lr(Xt, yt, feature_cols)

    print("\n  [2/2] Training LightGBM GBDT...")
    lgbm = train_lgbm(Xt, yt, Xv, yv, feature_cols)

    # Save models
    with open(MODELS_DIR / "lgbm.pkl", 'wb') as f:
        pickle.dump(lgbm, f)
    with open(MODELS_DIR / "lr.pkl", 'wb') as f:
        pickle.dump({'model': lr_model, 'scaler': lr_scaler}, f)
    print("  Models saved to models/")

    del Xt, yt, train_rows
    gc.collect()

    # Validation & Threshold Sweep
    print("\n--- Validation & Threshold Sweep ---")
    probs_v = predict_probs(lgbm, Xv)
    all_val_s1_ids = set()
    for ids in val_s1_sample.values():
        all_val_s1_ids |= ids
    val_gt = {k: full_gt[k] for k in all_val_s1_ids if k in full_gt}

    best_params, best_f05, best_metrics = sweep_thresholds(
        val_s1s, val_cids, probs_v, val_gt, all_val_s1_ids)

    print(f"\n  Optimal Thresholds: tau1={best_params['tau1']:.2f}, gap={best_params['gap']:.2f}")
    print(f"  Validation Macro F0.5 = {best_f05:.4f}")
    print(f"    - Macro Precision   = {best_metrics['precision_macro']:.4f}")
    print(f"    - Macro Recall      = {best_metrics['recall_macro']:.4f}")
    print(f"    - Singleton F0.5    = {best_metrics['singleton_f05']:.4f} ({best_metrics['singleton_count']:,} entities)")
    print(f"    - Single-Match F0.5 = {best_metrics['single_match_f05']:.4f} ({best_metrics['single_match_count']:,} entities)")
    print(f"    - Multi-Match F0.5  = {best_metrics['multi_match_f05']:.4f} ({best_metrics['multi_match_count']:,} entities)")

    # LR Baseline Score
    lr_probs = predict_probs(lr_model, Xv, lr_scaler)
    lr_preds = apply_rank_threshold(val_s1s, val_cids, lr_probs,
                                    best_params['tau1'], best_params['gap'], all_val_s1_ids)
    lr_m = macro_f05(lr_preds, val_gt)
    print(f"\n  Baseline Logistic Regression F0.5 = {lr_m['f05_macro']:.4f}")

    del Xv, yv, val_rows, probs_v, lr_probs
    gc.collect()

    with open(MODELS_DIR / "threshold.pkl", 'wb') as f:
        pickle.dump(best_params, f)

    # Save validation report
    val_report_path = REPORTS_DIR / "validation_results.md"
    with open(val_report_path, 'w', encoding='utf-8') as f:
        f.write("# Validation & Evaluation Report\n\n")
        f.write(f"- **Validation Entities**: {len(all_val_s1_ids):,}\n")
        f.write(f"- **Optimal Parameters**: `tau1 = {best_params['tau1']:.2f}`, `gap = {best_params['gap']:.2f}`\n\n")
        f.write("## Model Performance Comparison\n\n")
        f.write("| Model | Macro F0.5 | Macro Precision | Macro Recall |\n")
        f.write("| :--- | :---: | :---: | :---: |\n")
        f.write(f"| Logistic Regression Baseline | {lr_m['f05_macro']:.4f} | {lr_m['precision_macro']:.4f} | {lr_m['recall_macro']:.4f} |\n")
        f.write(f"| **LightGBM (Proposed)** | **{best_f05:.4f}** | **{best_metrics['precision_macro']:.4f}** | **{best_metrics['recall_macro']:.4f}** |\n\n")
        f.write("## Breakdown by Entity Type\n\n")
        f.write("| Category | Count | F0.5 Score |\n")
        f.write("| :--- | :---: | :---: |\n")
        f.write(f"| Genuine Singletons | {best_metrics['singleton_count']:,} | {best_metrics['singleton_f05']:.4f} |\n")
        f.write(f"| Single-Match Entities | {best_metrics['single_match_count']:,} | {best_metrics['single_match_f05']:.4f} |\n")
        f.write(f"| Multi-Match Entities | {best_metrics['multi_match_count']:,} | {best_metrics['multi_match_f05']:.4f} |\n")
    print(f"  Validation report written to {val_report_path}")

    # Test Inference (Streaming Country by Country)
    print("\n" + "=" * 70)
    print("PHASE 13-15: Test Inference & File Generation")
    print("=" * 70)

    matching_path  = OUTPUT_DIR / "matching_results.tsv"
    candidate_path = OUTPUT_DIR / "candidate_pairs.tsv"

    # Pre-read all test S1 IDs to guarantee 100% coverage
    test_s1_raw_path = TEST_DIR / "test_source1.tsv"
    all_test_s1_ids = set()
    with open(test_s1_raw_path, 'r', encoding='utf-8', errors='replace') as f:
        f.readline()
        for line in f:
            eid = line.split('\t', 1)[0].strip()
            if eid:
                all_test_s1_ids.add(eid)
    print(f"  Total Test S1 Entities to Resolve: {len(all_test_s1_ids):,}")

    f_match = open(matching_path, 'w', encoding='utf-8')
    f_cand  = open(candidate_path, 'w', encoding='utf-8')
    f_match.write("source1_entity_id\tmatched_entity_ids\n")
    f_cand.write("source1_entity_id\tcandidate_entity_ids\n")

    written_s1_ids = set()
    tot_matches_written = 0
    tot_candidates_written = 0
    BATCH_SIZE = 40_000

    for ctry in test_countries:
        print(f"\n  Inference on test country: {ctry.upper()}")
        s2_path = _country_split_path("test", "S2", ctry)
        s3_path = _country_split_path("test", "S3", ctry)
        s1_path = _country_split_path("test", "S1", ctry)

        if not s1_path.exists():
            continue

        print(f"    Building candidate index from S2 + S3...", end=" ", flush=True)
        t0 = time.time()
        cand_records, cand_id_map, idx = build_candidate_store_and_index(s2_path, s3_path)
        print(f"{len(cand_records):,} candidates ({time.time()-t0:.1f}s)")

        # Stream S1 in batches
        s1_batch = []
        n_ctry_s1 = 0
        t_start_ctry = time.time()

        for eid, name, addr, c in stream_records(s1_path):
            s1_batch.append(normalize_record(eid, name, addr, c))
            if len(s1_batch) >= BATCH_SIZE:
                # Process batch
                rows, cands_map = featurize_s1_batch(
                    s1_batch, cand_records, cand_id_map, idx, gt_dict=None, max_cands=200)
                
                batch_s1_set = {r['id'] for r in s1_batch}
                if rows:
                    X_b = np.array([[r.get(c, 0.0) for c in feature_cols] for r in rows], dtype=np.float32)
                    probs_b = predict_probs(lgbm, X_b)
                    s1_ids_b   = [r['s1_id']   for r in rows]
                    cand_ids_b = [r['cand_id'] for r in rows]
                    preds_b = apply_rank_threshold(
                        s1_ids_b, cand_ids_b, probs_b,
                        best_params['tau1'], best_params['gap'], batch_s1_set)
                else:
                    preds_b = {s1_id: set() for s1_id in batch_s1_set}

                for s1_id in batch_s1_set:
                    matched = sorted(preds_b.get(s1_id, set()))
                    cands   = sorted(cands_map.get(s1_id, set()))
                    f_match.write(f"{s1_id}\t{','.join(matched)}\n")
                    f_cand.write(f"{s1_id}\t{','.join(cands)}\n")
                    written_s1_ids.add(s1_id)
                    tot_matches_written += len(matched)
                    tot_candidates_written += len(cands)

                n_ctry_s1 += len(s1_batch)
                print(f"      Processed {n_ctry_s1:,} S1 entities...", flush=True)
                s1_batch = []
                del rows, cands_map; gc.collect()

        # Final batch for country
        if s1_batch:
            rows, cands_map = featurize_s1_batch(
                s1_batch, cand_records, cand_id_map, idx, gt_dict=None, max_cands=200)
            batch_s1_set = {r['id'] for r in s1_batch}
            if rows:
                X_b = np.array([[r.get(c, 0.0) for c in feature_cols] for r in rows], dtype=np.float32)
                probs_b = predict_probs(lgbm, X_b)
                s1_ids_b   = [r['s1_id']   for r in rows]
                cand_ids_b = [r['cand_id'] for r in rows]
                preds_b = apply_rank_threshold(
                    s1_ids_b, cand_ids_b, probs_b,
                    best_params['tau1'], best_params['gap'], batch_s1_set)
            else:
                preds_b = {s1_id: set() for s1_id in batch_s1_set}

            for s1_id in batch_s1_set:
                matched = sorted(preds_b.get(s1_id, set()))
                cands   = sorted(cands_map.get(s1_id, set()))
                f_match.write(f"{s1_id}\t{','.join(matched)}\n")
                f_cand.write(f"{s1_id}\t{','.join(cands)}\n")
                written_s1_ids.add(s1_id)
                tot_matches_written += len(matched)
                tot_candidates_written += len(cands)

            n_ctry_s1 += len(s1_batch)
            s1_batch = []
            del rows, cands_map; gc.collect()

        print(f"    Completed {ctry.upper()}: {n_ctry_s1:,} entities in {time.time()-t_start_ctry:.1f}s")
        del cand_records, cand_id_map, idx
        gc.collect()

    # Fill any remaining S1 entities as genuine singletons
    missing_s1 = all_test_s1_ids - written_s1_ids
    if missing_s1:
        print(f"  Writing {len(missing_s1):,} unmatched test entities as singletons...")
        for s1_id in missing_s1:
            f_match.write(f"{s1_id}\t\n")
            f_cand.write(f"{s1_id}\t\n")
            written_s1_ids.add(s1_id)

    f_match.close()
    f_cand.close()

    print(f"\n  Output files generated:")
    print(f"    - {matching_path} ({len(written_s1_ids):,} S1 rows, {tot_matches_written:,} matches)")
    print(f"    - {candidate_path} ({len(written_s1_ids):,} S1 rows, {tot_candidates_written:,} candidates)")

    # Run Submission Validator
    print("\n" + "=" * 70)
    print("PHASE 16: Running Official Submission Validator")
    print("=" * 70)
    validator_script = RESOURCE / "utils" / "validate_submission.py"
    if validator_script.exists():
        cmd = f'python "{validator_script}" --matching "{matching_path}" --candidate "{candidate_path}" --test-dir "{TEST_DIR}"'
        print(f"  Executing: {cmd}\n")
        os.system(cmd)
    else:
        print(f"  Validator script not found at {validator_script}")

    elapsed = time.time() - wall_t0
    print("\n" + "=" * 70)
    print(f"  PIPELINE COMPLETE - Elapsed: {elapsed/60:.2f} minutes")
    print("=" * 70)


if __name__ == "__main__":
    run()
