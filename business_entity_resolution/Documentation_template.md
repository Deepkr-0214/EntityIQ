# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** EntityResolvers  
**Team Members:** Deep Kumar Sinha  
**Submission Date:** 2026-09-27  

---

## 1. Executive Summary
We developed an end-to-end entity resolution pipeline for linking multi-source business records across France, India, and the United States. Our approach combines a high-recall multi-strategy inverted index blocking with Address Rescue candidate ranking (CAP=200) and a class-weighted Logistic Regression classifier over 40 engineered similarity, phonetics, and rank features. The frozen system achieves a validated F0.5 score of 0.7388 (Precision: 89.71%, Recall: 65.19%) and processes all 1.73M test queries in constant, safe memory (<2.5 GB RAM).

---

## 2. Methodology

### 2.1 Problem Analysis
EDA revealed several domain-specific challenges:
- High noise in address formatting, abbreviations (Rd/Road, St/Street, Apt/Suite), and PIN codes.
- Transliteration issues between Indic vernacular scripts and Latin characters.
- Severe class imbalance where true matches represent less than 1.5% of blocked candidate pairs.
- Unseen/open-set distributions in France requiring robust zero-shot generalization without overfitting country-specific tokens.

### 2.2 Solution Strategy
**Approach Type:** Multi-Strategy Inverted Index Blocking + Address Rescue Ranking + Class-Weighted Logistic Regression + Optimal Thresholding.  
**Core Innovation:** Address Rescue ranking that heavily weights consonant skeletons, numerical address tokens, and token overlap to ensure 89.83% blocking recall at a strict CAP of 200 candidates per query, paired with batch LR inference.

---

## 3. Candidate Generation (Blocking)
Comparison space reduction was achieved through a compact inverted index across 7 complementary blocking strategies:
- **First Token:** Exact match on first significant name token.
- **Name Prefix:** 4-character prefix of aggressive normalized name.
- **Name Tokens:** Inverted index over non-stopword name tokens.
- **Address Numbers:** Numerical house numbers, plot numbers, and PIN codes (length >= 4).
- **Address Tokens:** Inverted index over significant address tokens.
- **Consonant Skeleton:** 3-character phonetic consonant skeleton (vowel-stripped, deduplicated) to bridge Indic-Latin transliteration gaps.
- **Consonant Trigrams:** Trigram overlaps of consonant skeletons.

Candidates in the pool were scored using the Address Rescue weighting scheme: `W = [8, 7, 2, 8, 6, 6, 1]`. The top 200 candidates with stable ID tie-breaking were selected, achieving an uncapped recall of 95.93% and a capped recall of 89.83% (265/295 true matches retained).

---

## 4. Matching Model

**Features Used (40 Total):**
- **Pairwise Similarity (27 features):**
  - Name: Levenshtein distance, Jaccard token similarity, partial ratio, token sort ratio, token set ratio, Monge-Elkan similarity, length difference/ratio.
  - Address: Levenshtein distance, token Jaccard, numeric token overlap, address partial ratio, token sort/set ratio.
- **Blocker & Ranking Features (13 features):**
  - Blocker strategy indicators (first token, prefix, name tokens, address numbers, address tokens, consonant skeleton, consonant n-grams).
  - Address Rescue score and rank.
  - Source flag (`candidate_is_s3`).
  - Specific token and n-gram Jaccards (`name_token_jaccard`, `address_token_jaccard`, `consonant_ngram_jaccard`).

**Model Type:** `sklearn.linear_model.LogisticRegression(class_weight='balanced', C=1.0, max_iter=1000, random_state=42)`  
**Threshold Selection Method:** Global calibration thresholding optimized on held-out calibration entities for macro F0.5 (`threshold = 0.998722`).

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro):** **0.7388**
  - Precision: **89.71%** (TP=194, FP=24)
  - Recall: **65.19%** (FN=101)
  - Exact Match Sets: **24 / 86**
  - Capped Blocking Recall: **89.83%** (265 / 295)
- **Common False Positives (Wrong Merges):**
  - Franchise chains / retail branches sharing identical corporate business names located on the same major arterial road or commercial plaza.
- **Common False Negatives (Missed Matches):**
  - Extreme legal name acronym abbreviations paired with drastic address relocations or missing address components in noisy records.

---

## 6. Conclusion
The frozen pipeline demonstrates that combining multi-strategy consonant and numeric blocking with Address Rescue candidate ranking achieves near-90% candidate recall within a strict 200-candidate cap. Pairwise feature engineering combined with class-weighted Logistic Regression and strict thresholding (0.998722) delivers high precision (89.71%) and a strong macro F0.5 score of 0.7388, while maintaining robust constant-memory inference across all 1.73M test entities.

---

## Appendix

### A. Code Artefacts
Complete runnable code is structured in the submission archive under:
`code/business_entity_resolution/src/`
- `src/preprocessing/normalization.py`: Two-tier light and aggressive text normalizer, Unicode transliteration, and consonant skeleton generator.
- `src/pipeline/run_pipeline.py`: Country splitting and index construction.
- `src/features/similarity_features.py`: Pairwise similarity and token feature calculations.
- `src/modeling/train.py`: Model fitting, scaling, and probability prediction.
- `run_test_inference_fast.py`: Production-speed test inference engine generating `matching_results.tsv` and `candidate_pairs.tsv`.
