# ML Evaluation & Experiment Report — EntityIQ Phase 3

**Evaluation Timestamp:** 2026-10-09 18:36:39  
**Evaluation Scope:** Stratified Representative Validation Split (600 validation entities)  
**Objective:** Macro-averaged $F_{0.5}$ optimization (2x precision weighting over recall).

---

## 1. Candidate Generation & Blocking Performance

| Metric | Measured Value | Target / Benchmark |
| :--- | :--- | :--- |
| **Candidate Blocking Recall** | **94.50%** | $\ge 98.0\%$ |
| **Average Candidates / S1** | **92.3** | $\le 200$ |
| **Median Candidates / S1** | **103.0** | $\le 100$ |
| **Reduction Ratio** | **>99.7318%** | $\ge 99.9\%$ |

---

## 2. Model Performance Comparison

| Model Configuration | Best Threshold / Params | Macro $F_{0.5}$ | Precision | Recall | Singleton $F_{0.5}$ |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression (Baseline)** | $\tau = 0.85$ | **0.8710** | 0.9715 | 0.8129 | 1.0000 |
| **LightGBM (Global Threshold)** | $\tau = 0.70$ | **0.8782** | 0.9926 | 0.8112 | 0.9737 |
| **LightGBM (Rank-Aware Threshold)** | $\tau_1 = 0.25, \text{gap} = 0.05$ | **0.8798** | 0.9935 | 0.8067 | 0.9737 |

---

## 3. Feature Importance (Top Features)

```
Top LightGBM Features by Split Gain / Count:
  1. name_jaccard
  2. name_token_sort
  3. name_monge_elkan
  4. addr_numeric_match
  5. addr_jaccard
  6. name_exact_agg
  7. name_wratio
  8. addr_token_sort
```

---

## 4. Serialized Artifacts
- Model 1: `models/logistic_regression.pkl`
- Model 2: `models/lightgbm.pkl`
- Config & Metrics: `models/model_metadata.json`
