# Business Entity Resolution Challenge — Implementation Plan

## 0. Objective

Build an end-to-end ML pipeline that, for every Source 1 (S1) business entity, identifies all matching records in Source 2 (S2) and Source 3 (S3). An S1 entity may have zero, one, or multiple matches. Evaluated on **F0.5** (precision-heavy, macro-averaged per S1 entity). Only challenge-provided data may be used — no external lookups, APIs, or geocoding.

**Core insight driving every decision:** F0.5 weights precision 2x over recall, and singletons score fully in the average. Blocking sets a hard recall ceiling that cannot be recovered later. So the two levers that matter most are:
1. Blocking recall (must be ~99%+ before anything else matters)
2. Matching precision / confident "no-match" decisions

---

## 1. Dataset

```
dataset/
├── train/
│   ├── train_source1.tsv   (reference/deduplicated)
│   ├── train_source2.tsv
│   ├── train_source3.tsv
│   └── train_ground_truth.tsv
└── test/
    ├── test_source1.tsv
    ├── test_source2.tsv
    └── test_source3.tsv
```

- All files are **tab-separated** — always `pd.read_csv(path, sep="\t")`.
- Columns: `entity_id` (prefix `S1-`/`S2-`/`S3-`), `business_name`, `business_address`, `country`.
- Ground truth: `source1_entity_id`, `matched_entity_ids` (comma-separated S2/S3 IDs, empty if none).
- Training countries: US, India. **Test adds France** — country must be treated as an open-set string field, never hardcoded or one-hot encoded to {US, India}.

---

## 2. Pipeline Architecture

```
Raw Data → EDA → Ground Truth Analysis → Normalization →
Blocking/Candidate Generation → Training Pair Construction →
Feature Engineering → Baseline Model → Validation (F0.5) →
Stronger Model → Threshold Optimization → Singleton Handling →
Test Inference → candidate_pairs.tsv + matching_results.tsv →
Submission Validation → Documentation → Final ZIP
```

Each phase must be implemented and validated independently — do not skip ahead to model training.

---

## 3. Phase-by-Phase Plan

### Phase 1 — Dataset Discovery & Profiling
- Inspect file paths, sizes, row counts, columns, dtypes, missing values, duplicate IDs/names/addresses, country distribution, string lengths.
- **Extra:** check per-source noise asymmetry (e.g., does S2 skew toward name noise, S3 toward address/landmark noise?). Check whether `country` is reliable enough to hard-filter on later.
- Output: `reports/dataset_profile.md`. Do not normalize or model yet.

### Phase 2 — Ground Truth Analysis
- Compute: total S1 entities, singleton count, single-match count, multi-match count, max matches per entity, S2-only / S3-only / both counts.
- Validate: every ground-truth ID exists in the right source file, no malformed or duplicate IDs.
- **Extra:** as soon as ground truth stats exist, run a first-pass blocking strategy and measure the **recall ceiling** immediately — don't wait until Phase 4 is "finished."

### Phase 3 — Data Cleaning & Normalization
- Never overwrite raw fields — always add `*_normalized` columns alongside originals.
- Build **two normalization tiers**:
  - *Light*: lowercase, whitespace/punctuation normalization (for exact-match blocking keys).
  - *Aggressive*: legal-suffix stripping, stopword removal, phonetic folding (for similarity features).
- **Mine the legal-suffix/abbreviation dictionary from the data itself** (frequency-count trailing tokens of business names) rather than hand-writing a generic Ltd/Pvt/Corp list — this surfaces dataset-specific and India-specific patterns.
- Same approach for address abbreviations (Rd/St/Nagar/Marg, etc.).

### Phase 4 — Candidate Generation / Blocking
- Don't do full S1×S2/S3 cross product for large data.
- **Use multiple independent blocking strategies and union the candidates**, not just one:
  - Sorted-neighborhood on normalized name prefix/first token
  - Character n-gram (trigram) inverted index + top-k cosine/Jaccard retrieval (catches typos, reordering)
  - Token-set blocking (any shared significant token)
  - Country as a filter — hard filter only if Phase 1 confirms country is reliable; otherwise soft filter
  - Address token overlap as a secondary/fallback key
- Track **recall** (fraction of true matches surviving blocking — target ≥99%) and **reduction ratio** together. This is the most common place competitions are won or lost.
- The final candidate set saved here must be the *exact* one fed to the matching model (this becomes `candidate_pairs.tsv`).

### Phase 5 — Training Pair Construction
- Construct `S1 + candidate → label (1/0)` from ground truth and blocked candidates.
- **Hard negatives are critical** — random negatives make F0.5 look artificially good in validation and fail on real ambiguous candidates. Prioritize negatives that are:
  - Same country + shared first name token, different entity (franchise/branch trap)
  - High Jaccard/TF-IDF similarity but confirmed non-match
  - Address near-duplicates (same street, different business)
- Set the negative:positive ratio to reflect the actual blocked candidate distribution, not an arbitrary 1:1, so predicted probabilities calibrate correctly for thresholding.

### Phase 6 — Feature Engineering
**Baseline (explicitly suggested by the problem statement):**
- `name_jaccard`, `name_levenshtein`, `name_tfidf_cosine`
- `address_jaccard`, `address_levenshtein`, `address_tfidf_cosine`
- `country_match`, `name_length_difference`, `address_length_difference`

**Additional features for a competitive edge:**
- Token-sort / token-set ratio (directly handles word-order transposition)
- Monge-Elkan / best-alignment similarity for partial/DBA name matches
- Numeric/PIN-code exact-match flags extracted from address
- Acronym-expansion match (name A's initials vs. name B)
- **Candidate rank features**: relative similarity rank among all candidates for that S1 entity (e.g., is this the top-1 candidate?) — teaches the model competitive/relative matching
- **Group-level aggregates**: max/mean similarity across all candidates per S1 entity — helps distinguish genuinely ambiguous entities from clean ones

Evaluate feature usefulness on validation data — don't blindly implement everything.

### Phase 7 — Baseline Model
- Logistic Regression (or similarly simple classifier) first.
- Purpose: initial F0.5, feature behavior, threshold behavior, debugging reference — not peak performance.

### Phase 8 — Stronger Models
- Random Forest / XGBoost / LightGBM after the baseline is established.
- Select based on validation F0.5, precision, recall, inference speed, memory, reproducibility.
- **License/size constraint: model must be MIT/Apache-2.0 licensed and ≤8B parameters.** Verify this explicitly for any pretrained component before use — classical ML (LogReg/GBM) is comfortably within bounds and usually sufficient; a bigger model won't compensate for weak features.

### Phase 9 — Validation Strategy
- No test ground truth exists — hold out a validation split from training.
- **Split by S1 entity, not by pair**, so all candidates for one S1 entity stay in the same split (avoids leakage).
- Validation should mirror the real task: full blocking → features → model → threshold → decision, evaluated per entity.

### Phase 10 — Evaluation Metric
```
F0.5 = (1.25 × Precision × Recall) / (0.25 × Precision + Recall)
```
- Macro-averaged per S1 entity (singletons included — correct empty prediction = 1.0, any false match = 0.0).
- Precision-heavy: false merges hurt more than misses. Track Precision, Recall, and F0.5 throughout, not just accuracy.

### Phase 11 — Threshold Optimization
- Do not default to 0.5. Sweep thresholds on the validation set, optimizing directly for F0.5.
- **Use rank-aware / gap-based thresholding**, not a single global cutoff:
  - Require top-1 candidate probability ≥ τ1 to accept any match at all.
  - Require a probability gap to the 2nd-best candidate before accepting a second match.
  - This directly encodes "don't add a second match unless it's clearly also real," protecting precision on multi-match entities.
- Break out validation F0.5 separately for **singleton / single-match / multi-match** entities — a common silent failure is good overall F0.5 masking poor singleton precision.
- Avoid overfitting the validation set when tuning thresholds.

### Phase 12 — Singleton / No-Match Handling
- A "no sufficiently reliable candidate" case must map to an empty match list — never force a match.
- Consider a lightweight auxiliary "has any match at all?" decision using group-level aggregate features, to catch borderline low-similarity entities the pairwise model might get wrong.
- Correct singleton identification is worth a full 1.0 per entity — don't neglect it.

### Phase 13 — Test Inference
- Run every test S1 entity through: normalization → blocking → features → model → threshold/decision logic.
- Every S1 test entity must be processed — none may be dropped, including France entities (open-set country handling must generalize here).

### Phase 14 — Generate `candidate_pairs.tsv`
- Columns: `source1_entity_id`, `candidate_entity_ids`.
- Must be the **exact final candidate set** immediately before model inference (last blocking/filtering stage, not an early pass).
- Every match in `matching_results.tsv` must appear here (validator checks this).

### Phase 15 — Generate `matching_results.tsv`
- Columns: `source1_entity_id`, `matched_entity_ids`.
- Exactly one row per S1 test entity, no duplicate S1 IDs, matched IDs only S2/S3 and must exist in test data, no duplicate IDs within a list, empty list for genuine singletons.

### Phase 16 — Submission Validation
```bash
python3 utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```
- Must print `PASS` before submission. Run this continuously during development, not just at the end — bake it into the pipeline script.

### Phase 17 — Documentation
Fill in `Documentation_template.md` covering: methodology, preprocessing, normalization, blocking strategy, training pair construction, feature engineering, model architecture, validation methodology, F0.5 results, threshold selection, singleton handling, final inference pipeline. Be precise about blocking reduction ratio and recall ceiling numbers — reviewers check this explicitly. Do not claim experiments that weren't actually run.

### Phase 18 — Reproducible Code Structure
```
business_entity_resolution/
├── src/
│   ├── config.py
│   ├── data/            (loader.py, validation.py)
│   ├── preprocessing/   (normalization.py, cleaning.py)
│   ├── blocking/        (candidate_generator.py)
│   ├── features/        (similarity_features.py)
│   ├── modeling/        (train.py, predict.py, threshold.py)
│   ├── evaluation/      (metrics.py)
│   └── pipeline/        (run_pipeline.py)
├── reports/
│   ├── dataset_profile.md
│   └── validation_results.md
├── output/
│   ├── matching_results.tsv
│   └── candidate_pairs.tsv
├── README.md
├── requirements.txt
└── Documentation_template.md
```
- Pin dependencies in `requirements.txt`; document exact run commands in `README.md`. Anyone should be able to regenerate both output files from raw data with no manual code edits.

### Final Submission Package
```
team_name_submission.zip
├── output/
│   ├── matching_results.tsv
│   └── candidate_pairs.tsv
├── code/
│   └── business_entity_resolution/ (src/, README.md, requirements.txt)
└── Documentation_template.md
```

---

## 4. Strict Restrictions (do not violate)
- No Google/internet lookups for businesses
- No external business/government registration databases
- No geocoding or commercial entity-resolution APIs
- No external data augmentation
- Final model: MIT/Apache-2.0 license, ≤8B parameters

---

## 5. Development Order (do not skip steps)

1. Dataset discovery
2. Dataset profiling / EDA
3. Ground-truth analysis (+ early recall-ceiling check)
4. Normalization
5. Blocking / candidate generation
6. Training pair generation (with hard negatives)
7. Similarity feature engineering
8. Baseline model
9. Validation + F0.5 (split by S1 entity)
10. Stronger models
11. Threshold optimization (rank/gap-aware)
12. Singleton handling
13. Final test inference
14. Generate `candidate_pairs.tsv`
15. Generate `matching_results.tsv`
16. Run official validator
17. Documentation
18. Package final ZIP

---

## 6. Priority Order If Time-Constrained

1. **Blocking recall ceiling** (~99%+) — nothing else matters until this is solid
2. Hard-negative-aware training pairs
3. Full feature set including token-sort and rank/group features
4. Threshold tuning with the gap-based multi-match rule
5. Singleton-specific validation and fix-up
6. Model upgrade (XGBoost/LightGBM) — smallest marginal win, do last

---

## 7. First Task

Start only with **Step 1 — Dataset Discovery + Profiling**: row counts, columns, missing values, duplicates, countries, string lengths, ground-truth distribution, singleton distribution, train/test differences, actual noise patterns. Stop and report findings before implementing normalization, blocking, or modeling.
