import csv
import re
import sys
import time
import unicodedata
from collections import defaultdict
from rapidfuzz import fuzz

LEGAL_RE = re.compile(
    r'\b(?:private\s+limited|pvt\.?\s*ltd\.?|ltd\.?|limited|llc\.?|llp\.?|inc\.?|incorporated|'
    r'corp\.?|corporation|co\.?|company|enterprises?|industries|group|holdings?|'
    r'partnership|associates?|foundation|trust|society|ventures?|solutions?|services?|'
    r'technolog(?:y|ies)|systems?|international|global|udyog|udhyog|sansthan|'
    r'sarl\.?|sas|eurl|sci\.?|societe|cie\.?|pc|pllc|pa|lp)\b',
    re.IGNORECASE
)
ABBREV = {
    'street': 'st', 'avenue': 'ave', 'boulevard': 'blvd', 'drive': 'dr',
    'road': 'rd', 'lane': 'ln', 'court': 'ct', 'place': 'pl', 'circle': 'cir',
    'highway': 'hwy', 'parkway': 'pkwy', 'terrace': 'ter', 'square': 'sq',
    'route': 'rte', 'north': 'n', 'south': 's', 'east': 'e', 'west': 'w',
    'suite': '', 'apartment': '', 'apt': '', 'unit': '', 'bldg': '', 'building': '',
    'floor': '', 'fl': '', 'room': '', 'no': '', 'number': '', 'box': '', 'pobox': '',
    'rue': 'r', 'chemin': 'ch', 'allee': 'all'
}
ABBREV_RE = re.compile(r'\b(' + '|'.join(k for k in ABBREV if k) + r')\b', re.IGNORECASE)
NUM_RE = re.compile(r'\b\d+\b')

def norm_str(x):
    if not x: return ''
    x = unicodedata.normalize('NFKD', x).encode('ascii', 'ignore').decode().lower()
    return re.sub(r'[^a-z0-9]+', ' ', x)

def clean_name(name):
    n = norm_str(name)
    n = LEGAL_RE.sub(' ', n)
    return ' '.join(n.split())

def clean_addr_sorted(addr):
    a = norm_str(addr)
    a = ABBREV_RE.sub(lambda m: ABBREV[m.group(0).lower()], a)
    toks = [t for t in a.split() if len(t) > 1 or t in {'n', 's', 'e', 'w'}]
    toks.sort()
    return ' '.join(toks)

def extract_main_number(addr):
    nums = NUM_RE.findall(addr)
    return nums[0] if nums else ''

print("Testing France pipeline...", flush=True)
t0 = time.time()

idx_name_addr = defaultdict(list)
idx_raw = defaultdict(list)
idx_name_num = defaultdict(list)
# We store raw address in a compact dict for candidates: {eid: clean_addr_str}
eid_addr = {}

s2_path = 'splits/test_S2_france.tsv'
s3_path = 'splits/test_S3_france.tsv'

for path in (s2_path, s3_path):
    print("Reading", path, flush=True)
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        r = csv.DictReader(f, delimiter='\t')
        for row in r:
            eid = row['entity_id']
            bname = row.get('business_name', '')
            baddr = row.get('business_address', '')
            
            cn = clean_name(bname)
            ca = clean_addr_sorted(baddr)
            
            if cn and ca:
                idx_name_addr[(cn, ca)].append(eid)
                
            rn = ' '.join(norm_str(bname).split())
            ra = ' '.join(norm_str(baddr).split())
            if rn and ra:
                idx_raw[(rn, ra)].append(eid)
                
            num = extract_main_number(baddr)
            if cn and num and len(cn) >= 3:
                idx_name_num[(cn, num)].append(eid)
                
            eid_addr[eid] = ca

t_idx = time.time() - t0
print(f"Index built in {t_idx:.1f}s", flush=True)
print(f"  idx_name_addr: {len(idx_name_addr):,} keys")
print(f"  idx_raw:       {len(idx_raw):,} keys")
print(f"  idx_name_num:  {len(idx_name_num):,} keys")

# Now process France S1
s1_path = 'splits/test_S1_france.tsv'
total_s1 = 0
matched_s1 = 0
match_details = Counter = defaultdict(int)

t_proc = time.time()
with open(s1_path, 'r', encoding='utf-8', errors='replace') as f:
    r = csv.DictReader(f, delimiter='\t')
    for row in r:
        total_s1 += 1
        sid = row['entity_id']
        bname = row.get('business_name', '')
        baddr = row.get('business_address', '')
        
        cn = clean_name(bname)
        ca = clean_addr_sorted(baddr)
        rn = ' '.join(norm_str(bname).split())
        ra = ' '.join(norm_str(baddr).split())
        num = extract_main_number(baddr)
        
        matches = []
        
        # Step 1: Cleaned name + sorted address tokens
        if cn and ca and (cn, ca) in idx_name_addr:
            cands = idx_name_addr[(cn, ca)]
            if len(cands) <= 8:
                matches = cands
                match_details['clean_both'] += 1
                
        # Step 2: Raw exact name + address
        if not matches and rn and ra and (rn, ra) in idx_raw:
            cands = idx_raw[(rn, ra)]
            if len(cands) <= 8:
                matches = cands
                match_details['raw_both'] += 1
                
        # Step 3: Clean name + same street number + address token overlap
        if not matches and cn and num and len(cn) >= 3 and (cn, num) in idx_name_num:
            cands = idx_name_num[(cn, num)]
            if 0 < len(cands) <= 5:
                # Check address overlap
                s1_toks = set(ca.split())
                good = []
                for cid in cands:
                    cand_toks = set(eid_addr[cid].split())
                    inter = len(s1_toks & cand_toks)
                    union = len(s1_toks | cand_toks)
                    jacc = inter / union if union > 0 else 0
                    if jacc >= 0.4 or inter >= 3:
                        good.append(cid)
                if good:
                    matches = good
                    match_details['name_num_overlap'] += 1
                    
        if matches:
            matched_s1 += 1

el_proc = time.time() - t_proc
print(f"Processed {total_s1:,} S1 rows in {el_proc:.1f}s ({total_s1/el_proc:.0f} rows/s)", flush=True)
print(f"Matches found: {matched_s1:,} / {total_s1:,} ({matched_s1/total_s1:.2%})", flush=True)
print("Match details:")
for k, v in match_details.items():
    print(f"  {k}: {v:,}", flush=True)
