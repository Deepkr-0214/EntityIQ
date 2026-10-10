import csv
import re
import unicodedata
from collections import defaultdict

BASE = r"D:\Amazon\student_resource\dataset\test"
S1 = BASE + r"\test_source1.tsv"
S2 = BASE + r"\test_source2.tsv"
S3 = BASE + r"\test_source3.tsv"

OUT = r"output\matching_results.tsv"
CAND = r"output\candidate_pairs.tsv"

def norm(x):
    if not x:
        return ""
    x = unicodedata.normalize("NFKD", x).encode("ascii", "ignore").decode()
    x = x.lower()
    x = re.sub(r"[^a-z0-9]+", " ", x)
    return " ".join(x.split())

def make_key(name, addr, country):
    return (
        country.strip().lower(),
        norm(name),
        norm(addr)
    )

idx = defaultdict(list)

print("BUILDING INDEX", flush=True)

for path in (S2, S3):
    print("Reading", path, flush=True)

    with open(path, "r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")

        n = 0

        for row in reader:
            n += 1

            eid = row["entity_id"]

            k = make_key(
                row.get("business_name", ""),
                row.get("business_address", ""),
                row.get("country", "")
            )

            if k[1] and k[2]:
                idx[k].append(eid)

            if n % 500000 == 0:
                print(
                    path.split("\\")[-1],
                    n,
                    "rows processed",
                    flush=True
                )

print("INDEX COMPLETE")
print("Unique keys:", len(idx), flush=True)

total = 0
matched = 0
pairs = 0

print("PROCESSING S1", flush=True)

with open(S1, "r", encoding="utf-8", errors="replace", newline="") as f1, \
     open(OUT, "w", encoding="utf-8", newline="") as fo, \
     open(CAND, "w", encoding="utf-8", newline="") as fc:

    reader = csv.DictReader(f1, delimiter="\t")

    wo = csv.writer(fo, delimiter="\t", lineterminator="\n")
    wc = csv.writer(fc, delimiter="\t", lineterminator="\n")

    wo.writerow(["source1_entity_id", "matched_entity_ids"])
    wc.writerow(["source1_entity_id", "candidate_entity_ids"])

    for row in reader:
        total += 1

        s1id = row["entity_id"]

        k = make_key(
            row.get("business_name", ""),
            row.get("business_address", ""),
            row.get("country", "")
        )

        ids = idx.get(k, [])

        # Only use an unambiguous exact match.
        if len(ids) == 1:
            value = ids[0]
            matched += 1
            pairs += 1
        else:
            value = ""

        wo.writerow([s1id, value])
        wc.writerow([s1id, value])

        if total % 100000 == 0:
            print(
                "S1 processed:",
                total,
                "| matches:",
                matched,
                flush=True
            )

print("DONE", flush=True)
print("S1 rows:", total, flush=True)
print("Matches:", matched, flush=True)
print("Candidate pairs:", pairs, flush=True)
print("Output:", OUT, flush=True)
print("Candidates:", CAND, flush=True)
