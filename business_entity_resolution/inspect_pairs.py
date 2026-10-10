"""Inspect 20 real matched pairs from train_GT_india.tsv."""
import sys, os
from pathlib import Path
from unidecode import unidecode
import re

PROJECT_ROOT = Path('d:/Amazon/business_entity_resolution')
SPLIT_DIR = PROJECT_ROOT / 'splits'

gt_path = SPLIT_DIR / "train_GT_india.tsv"
s1_path = SPLIT_DIR / "train_S1_india.tsv"
s2_path = SPLIT_DIR / "train_S2_india.tsv"
s3_path = SPLIT_DIR / "train_S3_india.tsv"

# Load 30 GT pairs
pairs = []
with open(gt_path, 'r', encoding='utf-8', errors='replace') as f:
    f.readline()
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) >= 2 and parts[1]:
            s1_id = parts[0]
            cand_ids = [c.strip() for c in parts[1].split(',') if c.strip()]
            for cid in cand_ids:
                pairs.append((s1_id, cid))
                if len(pairs) >= 30:
                    break
        if len(pairs) >= 30:
            break

s1_needed = {s1 for s1, _ in pairs}
cand_needed = {cid for _, cid in pairs}

s1_data = {}
with open(s1_path, 'r', encoding='utf-8', errors='replace') as f:
    f.readline()
    for line in f:
        parts = line.strip().split('\t')
        if parts and parts[0] in s1_needed:
            s1_data[parts[0]] = (parts[1] if len(parts)>1 else '', parts[2] if len(parts)>2 else '')

cand_data = {}
for p in (s2_path, s3_path):
    if not p.exists(): continue
    with open(p, 'r', encoding='utf-8', errors='replace') as f:
        f.readline()
        for line in f:
            parts = line.strip().split('\t')
            if parts and parts[0] in cand_needed:
                cand_data[parts[0]] = (parts[1] if len(parts)>1 else '', parts[2] if len(parts)>2 else '')

with open('scratch_pairs.txt', 'w', encoding='utf-8') as out:
    for s1_id, cid in pairs:
        s1_n, s1_a = s1_data.get(s1_id, ('', ''))
        c_n, c_a = cand_data.get(cid, ('', ''))
        out.write(f"S1: {s1_id} | {s1_n} | {s1_a}\n")
        out.write(f"CD: {cid} | {c_n} | {c_a}\n")
        out.write(f"TL: {unidecode(c_n)} | {unidecode(c_a)}\n")
        out.write("-" * 60 + "\n")
print(f"Wrote {len(pairs)} pairs to scratch_pairs.txt")
