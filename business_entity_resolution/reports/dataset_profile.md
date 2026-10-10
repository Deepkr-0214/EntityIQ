# Dataset Profile & Discovery Report

## 1. Executive Summary & File Inventory

The dataset consists of 3 independent sources ($S_1$, $S_2$, $S_3$) across training and test splits, totaling **12,527,040 training records** and **11,702,133 test records** (plus 2,206,821 ground truth records).

All files are strictly tab-separated (`.tsv`) with UTF-8 encoding.

| Split | File | Rows | File Size | Columns |
| :--- | :--- | :--- | :--- | :--- |
| **Train** | `train_source1.tsv` | 2,206,821 | 200.34 MB | `entity_id`, `business_name`, `business_address`, `country` |
| **Train** | `train_source2.tsv` | 5,034,616 | 466.63 MB | `entity_id`, `business_name`, `business_address`, `country` |
| **Train** | `train_source3.tsv` | 5,285,603 | 480.37 MB | `entity_id`, `business_name`, `business_address`, `country` |
| **Train** | `train_ground_truth.tsv` | 2,206,821 | 121.13 MB | `source1_entity_id`, `matched_entity_ids` |
| **Test** | `test_source1.tsv` | 1,732,544 | 166.91 MB | `entity_id`, `business_name`, `business_address`, `country` |
| **Test** | `test_source2.tsv` | 4,887,273 | 485.86 MB | `entity_id`, `business_name`, `business_address`, `country` |
| **Test** | `test_source3.tsv` | 5,082,316 | 482.56 MB | `entity_id`, `business_name`, `business_address`, `country` |

---

## 2. Field Analysis & Missing Values

- **`entity_id`**: 0 missing values across all files. Prefixed strictly by source (`S1-`, `S2-`, `S3-`). Uniquely identified per file.
- **`business_name`**: 0 missing values across all files ($S_1, S_2, S_3$ train and test). Every single entity has a name.
- **`business_address`**: 
  - $S_1$ (Train & Test): 0 missing values (100% complete).
  - $S_2$ Train: 168,967 empty addresses (3.36%).
  - $S_3$ Train: 175,916 empty addresses (3.33%).
  - $S_2$ Test: 129,408 empty addresses (2.65%).
  - $S_3$ Test: 136,098 empty addresses (2.68%).
  - *Key takeaway*: When matching records with empty addresses, the decision must rely exclusively on name similarity and country constraint.
- **`country`**: 0 missing values across all files.

---

## 3. Country Distribution & Cross-Source Consistency

### Training Split
- **$S_1$**: US: 1,323,633 (60.0%), India: 883,188 (40.0%)
- **$S_2$**: US: 3,016,817 (59.9%), India: 2,017,799 (40.1%)
- **$S_3$**: US: 3,170,056 (60.0%), India: 2,115,547 (40.0%)

### Test Split (Features Unseen Country: France)
- **$S_1$**: India: 809,986 (46.8%), US: 663,106 (38.3%), France: 259,452 (15.0%)
- **$S_2$**: India: 2,312,565 (47.3%), US: 1,871,330 (38.3%), France: 703,378 (14.4%)
- **$S_3$**: India: 2,405,000 (47.3%), US: 1,945,701 (38.3%), France: 731,615 (14.4%)

### Country Consistency Audit
- Audited 67,044 true positive pairs from ground truth.
- **Result**: **0 country mismatches** (100% consistency).
- **Implication**: Country operates as a strict partitioning boundary. Partitioning candidates by country reduces the candidate search space by ~60% for US, ~50% for India, and ~85% for France without sacrificing recall.

---

## 4. Ground Truth Matching Statistics

Analysis of all 2,206,821 ground truth records:

- **Total $S_1$ Entities**: 2,206,821
- **Singletons (0 matches)**: 123,247 (5.58%)
- **Single Match (1 match)**: 119,157 (5.40%)
- **Multi-Match (2+ matches)**: 1,964,417 (**89.02%**)
- **Max Matches for Single Entity**: 11 matches
- **Total Matches**: 7,638,365
  - $S_2$ Matches: 3,693,619 (73.4% of all $S_2$ records in train)
  - $S_3$ Matches: 3,944,746 (74.6% of all $S_3$ records in train)
- **Match Count Distribution**:
  - 3 matches: 530,841 (24.05%)
  - 4 matches: 484,115 (21.94%)
  - 2 matches: 375,212 (17.00%)
  - 5 matches: 321,957 (14.59%)
  - 6 matches: 164,868 (7.47%)
  - 0 matches: 123,247 (5.58%)
  - 1 match: 119,157 (5.40%)
  - 7 matches: 63,968 (2.90%)
  - 8 matches: 18,680 (0.85%)
  - 9+ matches: 4,776 (0.22%)

---

## 5. Noise Patterns & Linguistic Characteristics

Inspection of true positive pairs reveals four dominant noise patterns:

1. **Language & Script Variations (Indian Entities)**:
   - English business name in $S_1$ vs. Devanagari (Hindi) or Tamil scripts in $S_2/S_3$ (e.g., `Raj Investments LLP` vs. `ராஜ் இன்வெஸ்ட்மெண்ட்ஸ் எல்எல்பி`).
   - In almost all script-divergent pairs, the **address components** (road names, building numbers, PIN codes) remain in Latin characters or match exactly.
2. **Domain Name Substitutions**:
   - Business names in $S_3$ frequently appear as web domains (e.g., `maurewilliamscolombier.com` matching `Maure Williams Colombier Inc`).
   - Normalization must strip TLD extensions (`.com`, `.in`, `.org`, `.net`, etc.) and separate hyphenated/concatenated domain names.
3. **Trade / DBA Names vs. Legal Names**:
   - $S_1$ entity with legal name (e.g. `Maure Williams Colombier Inc`) matching trade name in $S_3$ (e.g. `Dréxkor`) where address is identical (`85 Wayne Avenue, Ticonderoga`).
   - Addresses serve as a critical secondary anchor when names diverge completely.
4. **Missing Addresses in $S_2/S_3$**:
   - Over 340,000 records in $S_2/S_3$ have blank addresses. When addresses are missing, names have high similarity (e.g., `Maure Wilblims Colombier Inc` with minor typo).
