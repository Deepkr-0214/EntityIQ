"""Debug script to analyze why blocking recall is low."""
import sys
import os

# Force UTF-8 on Windows
if sys.platform == 'win32':
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path('d:/Amazon/business_entity_resolution')
sys.path.insert(0, str(PROJECT_ROOT))
from src.preprocessing.normalization import normalize_record

TRAIN_DIR = Path('d:/Amazon/student_resource/dataset/train')

# Load ground truth
gt = {}
with open(TRAIN_DIR / 'train_ground_truth.tsv', 'r', encoding='utf-8', errors='replace') as f:
    f.readline()
    for line in f:
        parts = line.rstrip('\n').split('\t')
        s1_id = parts[0].strip()
        matched = parts[1].strip() if len(parts) > 1 else ''
        if matched:
            gt[s1_id] = set(m.strip() for m in matched.split(',') if m.strip())
        else:
            gt[s1_id] = set()

# Collect all S2/S3 IDs that appear in ground truth
all_match_ids = set()
for ms in gt.values():
    all_match_ids |= ms

print(f"Total GT entries: {len(gt)}, with matches: {sum(1 for v in gt.values() if v)}")
print(f"Total match IDs referenced: {len(all_match_ids)}")

# Load S1 records
s1_recs = {}
with open(TRAIN_DIR / 'train_source1.tsv', 'r', encoding='utf-8', errors='replace') as f:
    f.readline()
    for line in f:
        parts = line.rstrip('\n').split('\t')
        eid = parts[0].strip()
        name = parts[1].strip() if len(parts) > 1 else ''
        addr = parts[2].strip() if len(parts) > 2 else ''
        ctry = parts[3].strip() if len(parts) > 3 else ''
        s1_recs[eid] = {'name': name, 'addr': addr, 'ctry': ctry}

# Load matching S2/S3 records  
s23_recs = {}
for fn in ['train_source2.tsv', 'train_source3.tsv']:
    with open(TRAIN_DIR / fn, 'r', encoding='utf-8', errors='replace') as f:
        f.readline()
        for line in f:
            parts = line.rstrip('\n').split('\t')
            eid = parts[0].strip()
            if eid in all_match_ids:
                name = parts[1].strip() if len(parts) > 1 else ''
                addr = parts[2].strip() if len(parts) > 2 else ''
                ctry = parts[3].strip() if len(parts) > 3 else ''
                s23_recs[eid] = {'name': name, 'addr': addr, 'ctry': ctry}

print(f"Loaded {len(s23_recs)} matched S2/S3 records")

# safe print helper
def sp(text):
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode('ascii', errors='replace').decode('ascii'))

# Analyze matched pairs from India
india_s1 = [s for s in gt if s in s1_recs and s1_recs[s]['ctry'].lower() == 'india' and gt[s]]
print(f"\nIndia S1 entities with matches: {len(india_s1)}")

count = 0
no_overlap_count = 0
ft_miss = 0
pfx_miss = 0
tok_miss = 0
total_pairs = 0

# Track why blocking fails
reasons = defaultdict(int)

for s1_id in india_s1[:500]:
    s1 = s1_recs[s1_id]
    s1_norm = normalize_record(s1_id, s1['name'], s1['addr'], s1['ctry'])
    for mid in list(gt[s1_id]):
        if mid not in s23_recs:
            continue
        m = s23_recs[mid]
        m_norm = normalize_record(mid, m['name'], m['addr'], m['ctry'])
        total_pairs += 1
        
        ft_match = s1_norm['ft'] == m_norm['ft']
        npfx_match = s1_norm['npfx'] == m_norm['npfx']
        tok_overlap = s1_norm['ntoks'] & m_norm['ntoks']
        addr_overlap = s1_norm['atoks'] & m_norm['atoks']
        num_overlap = set(s1_norm['anums']) & set(m_norm['anums'])
        
        any_overlap = ft_match or npfx_match or bool(tok_overlap) or bool(num_overlap) or len(addr_overlap) >= 2
        
        if not ft_match:
            ft_miss += 1
        if not npfx_match:
            pfx_miss += 1
        if not tok_overlap:
            tok_miss += 1
        
        if not any_overlap:
            no_overlap_count += 1
            if count < 30:
                sp(f"--- {s1_id} <-> {mid} (NO BLOCKING OVERLAP) ---")
                sp(f"  S1 name: {s1['name'][:80]}")
                sp(f"  Mx name: {m['name'][:80]}")
                sp(f"  S1 na='{s1_norm['na'][:60]}'")
                sp(f"  Mx na='{m_norm['na'][:60]}'")
                sp(f"  S1 ft='{s1_norm['ft']}', pfx='{s1_norm['npfx']}'")
                sp(f"  Mx ft='{m_norm['ft']}', pfx='{m_norm['npfx']}'")
                sp(f"  S1 ntoks={s1_norm['ntoks']}")
                sp(f"  Mx ntoks={m_norm['ntoks']}")
                sp(f"  S1 atoks={list(s1_norm['atoks'])[:8]}")
                sp(f"  Mx atoks={list(m_norm['atoks'])[:8]}")
                sp(f"  S1 anums={s1_norm['anums']}, Mx anums={m_norm['anums']}")
                sp("")
                count += 1

# Also check US
us_s1 = [s for s in gt if s in s1_recs and s1_recs[s]['ctry'].lower() == 'us' and gt[s]]
us_no_overlap = 0
us_total = 0
for s1_id in us_s1[:500]:
    s1 = s1_recs[s1_id]
    s1_norm = normalize_record(s1_id, s1['name'], s1['addr'], s1['ctry'])
    for mid in list(gt[s1_id]):
        if mid not in s23_recs:
            continue
        m = s23_recs[mid]
        m_norm = normalize_record(mid, m['name'], m['addr'], m['ctry'])
        us_total += 1
        
        ft_match = s1_norm['ft'] == m_norm['ft']
        npfx_match = s1_norm['npfx'] == m_norm['npfx']
        tok_overlap = s1_norm['ntoks'] & m_norm['ntoks']
        addr_overlap = s1_norm['atoks'] & m_norm['atoks']
        num_overlap = set(s1_norm['anums']) & set(m_norm['anums'])
        
        any_overlap = ft_match or npfx_match or bool(tok_overlap) or bool(num_overlap) or len(addr_overlap) >= 2
        if not any_overlap:
            us_no_overlap += 1

sp(f"\n=== Summary (first 500 India S1 with matches) ===")
sp(f"Total pairs checked: {total_pairs}")
sp(f"Pairs with NO blocking overlap: {no_overlap_count} ({no_overlap_count/max(1,total_pairs)*100:.1f}%)")
sp(f"First-token misses: {ft_miss}/{total_pairs} ({ft_miss/max(1,total_pairs)*100:.1f}%)")
sp(f"Prefix misses: {pfx_miss}/{total_pairs} ({pfx_miss/max(1,total_pairs)*100:.1f}%)")
sp(f"Token overlap misses (no shared name token): {tok_miss}/{total_pairs} ({tok_miss/max(1,total_pairs)*100:.1f}%)")
sp(f"\n=== US Summary (first 500) ===")
sp(f"Total pairs: {us_total}")
sp(f"Pairs with NO blocking overlap: {us_no_overlap} ({us_no_overlap/max(1,us_total)*100:.1f}%)")
