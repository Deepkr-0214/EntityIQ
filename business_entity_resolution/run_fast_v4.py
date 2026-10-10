import csv
import re
import time
import unicodedata
from collections import defaultdict

BASE = r"D:\Amazon\student_resource\dataset\test"
OUT = r"output\matching_results_v4.tsv"
CAND = r"output\candidate_pairs_v4.tsv"

LEGAL_RE = re.compile(
    r'\b(?:private\s+limited|pvt\.?\s*ltd\.?|ltd\.?|limited|llc\.?|llp\.?|inc\.?|incorporated|'
    r'corp\.?|corporation|co\.?|company|enterprises?|industries|group|holdings?|'
    r'partnership|associates?|foundation|trust|society|ventures?|solutions?|services?|'
    r'technolog(?:y|ies)|systems?|international|global|udyog|udhyog|sansthan|'
    r'sarl\.?|sas|eurl|sci\.?|societe|cie\.?|pc|pllc|pa|lp)\b',
    re.IGNORECASE
)

ABBREV = {
    'street':'st','avenue':'ave','boulevard':'blvd','drive':'dr',
    'road':'rd','lane':'ln','court':'ct','place':'pl','circle':'cir',
    'highway':'hwy','parkway':'pkwy','terrace':'ter','square':'sq',
    'route':'rte','north':'n','south':'s','east':'e','west':'w',
    'suite':'','apartment':'','apt':'','unit':'','bldg':'','building':'',
    'floor':'','fl':'','room':'','no':'','number':'','box':'','pobox':'',
    'rue':'r','chemin':'ch','allee':'all'
}

ABBREV_RE = re.compile(
    r'\b(' + '|'.join(ABBREV.keys()) + r')\b',
    re.IGNORECASE
)
NUM_RE = re.compile(r'\b\d+\b')

def norm_str(x):
    if not x:
        return ""
    x = unicodedata.normalize("NFKD", x).encode("ascii","ignore").decode().lower()
    return re.sub(r"[^a-z0-9]+", " ", x)

def clean_name(x):
    n = norm_str(x)
    n = LEGAL_RE.sub(" ", n)
    return " ".join(n.split())

def clean_addr(x):
    a = norm_str(x)
    a = ABBREV_RE.sub(lambda m: ABBREV[m.group(0).lower()], a)
    toks = [t for t in a.split() if len(t) > 1 or t in {"n","s","e","w"}]
    toks.sort()
    return " ".join(toks)

def main_num(x):
    nums = NUM_RE.findall(x or "")
    return nums[0] if nums else ""

def process_country(country):
    print("\n" + "="*60, flush=True)
    print("COUNTRY:", country.upper(), flush=True)
    print("="*60, flush=True)

    s2 = f"{BASE}\\test_source2.tsv"
    s3 = f"{BASE}\\test_source3.tsv"
    s1 = f"{BASE}\\test_source1.tsv"

    idx_name_addr = defaultdict(list)
    idx_raw = defaultdict(list)
    idx_name_num = defaultdict(list)
    eid_addr = {}

    t0 = time.time()
    rows = 0

    print("BUILDING INDEX...", flush=True)

    for path in (s2, s3):
        print("Reading:", path, flush=True)

        with open(path, "r", encoding="utf-8", errors="replace", newline="") as f:
            r = csv.DictReader(f, delimiter="\t")

            for row in r:
                if row.get("country","").strip().lower() != country:
                    continue

                eid = row["entity_id"]
                name = row.get("business_name","")
                addr = row.get("business_address","")

                cn = clean_name(name)
                ca = clean_addr(addr)

                if cn and ca:
                    idx_name_addr[(cn,ca)].append(eid)

                rn = " ".join(norm_str(name).split())
                ra = " ".join(norm_str(addr).split())

                if rn and ra:
                    idx_raw[(rn,ra)].append(eid)

                num = main_num(addr)

                if cn and num and len(cn) >= 3:
                    idx_name_num[(cn,num)].append(eid)

                eid_addr[eid] = ca
                rows += 1

                if rows % 500000 == 0:
                    print("indexed:", rows, flush=True)

    print("INDEX COMPLETE", flush=True)
    print("Rows indexed:", rows, flush=True)
    print("clean name+addr keys:", len(idx_name_addr), flush=True)
    print("raw keys:", len(idx_raw), flush=True)
    print("name+number keys:", len(idx_name_num), flush=True)
    print("Index time:", round(time.time()-t0,1), "sec", flush=True)

    total = 0
    matched = 0
    detail = defaultdict(int)

    results = {}
    candidates = {}

    t1 = time.time()

    print("PROCESSING S1...", flush=True)

    with open(s1, "r", encoding="utf-8", errors="replace", newline="") as f:
        r = csv.DictReader(f, delimiter="\t")

        for row in r:
            if row.get("country","").strip().lower() != country:
                continue

            total += 1
            sid = row["entity_id"]
            name = row.get("business_name","")
            addr = row.get("business_address","")

            cn = clean_name(name)
            ca = clean_addr(addr)

            rn = " ".join(norm_str(name).split())
            ra = " ".join(norm_str(addr).split())

            num = main_num(addr)

            matches = []

            # 1. Clean name + sorted address
            if cn and ca:
                ids = idx_name_addr.get((cn,ca), [])
                if 0 < len(ids) <= 8:
                    matches = list(dict.fromkeys(ids))
                    detail["clean_both"] += 1

            # 2. Raw normalized exact
            if not matches and rn and ra:
                ids = idx_raw.get((rn,ra), [])
                if 0 < len(ids) <= 8:
                    matches = list(dict.fromkeys(ids))
                    detail["raw_both"] += 1

            # 3. Clean name + main address number + overlap
            if not matches and cn and num and len(cn) >= 3:
                ids = idx_name_num.get((cn,num), [])

                if 0 < len(ids) <= 5:
                    st = set(ca.split())
                    good = []

                    for cid in ids:
                        ct = set(eid_addr.get(cid,"").split())

                        inter = len(st & ct)
                        union = len(st | ct)

                        jac = inter / union if union else 0

                        if jac >= 0.55 or inter >= 3:
                            good.append(cid)

                    if good:
                        matches = list(dict.fromkeys(good))
                        detail["name_num_overlap"] += 1

            results[sid] = matches
            candidates[sid] = matches

            if matches:
                matched += 1

            if total % 100000 == 0:
                print(
                    country.upper(),
                    "S1:", total,
                    "matches:", matched,
                    flush=True
                )

    elapsed = time.time() - t1

    print("COUNTRY COMPLETE:", country.upper(), flush=True)
    print("S1:", total, flush=True)
    print("Matches:", matched, flush=True)
    print("Match rate:", round(100*matched/max(1,total),2), "%", flush=True)
    print("Processing time:", round(elapsed,1), "sec", flush=True)

    for k,v in detail.items():
        print(" ",k,":",v, flush=True)

    return results, candidates, total, matched


def main():
    all_results = {}
    all_candidates = {}

    total_s1 = 0
    total_matches = 0

    # Process each country separately.
    for country in ("france","india","us"):
        res, cand, n, m = process_country(country)

        all_results.update(res)
        all_candidates.update(cand)

        total_s1 += n
        total_matches += m

    print("\nWRITING OUTPUT...", flush=True)

    with open(OUT, "w", encoding="utf-8", newline="") as f1, \
         open(CAND, "w", encoding="utf-8", newline="") as f2:

        w1 = csv.writer(f1, delimiter="\t", lineterminator="\n")
        w2 = csv.writer(f2, delimiter="\t", lineterminator="\n")

        w1.writerow(["source1_entity_id","matched_entity_ids"])
        w2.writerow(["source1_entity_id","candidate_entity_ids"])

        # IMPORTANT:
        # Read S1 again to preserve exact test-source ordering.
        with open(
            f"{BASE}\\test_source1.tsv",
            "r",
            encoding="utf-8",
            errors="replace",
            newline=""
        ) as f:

            r = csv.DictReader(f, delimiter="\t")

            written = 0

            for row in r:
                sid = row["entity_id"]
                ids = list(dict.fromkeys(all_results.get(sid, [])))

                value = ",".join(ids)

                w1.writerow([sid,value])
                w2.writerow([sid,value])

                written += 1

                if written % 250000 == 0:
                    print("written:", written, flush=True)

    print("\nDONE", flush=True)
    print("Total S1:", total_s1, flush=True)
    print("Total matches:", total_matches, flush=True)
    print("Output:", OUT, flush=True)
    print("Candidates:", CAND, flush=True)


if __name__ == "__main__":
    main()

