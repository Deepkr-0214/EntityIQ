"""
Integration test for end-to-end matching on synthetic data.
"""
import pytest
import numpy as np
from sklearn.linear_model import LogisticRegression
from src.preprocessing.normalization import normalize_record
from src.features.similarity_features import compute_pairwise_features, FEATURE_COLUMNS
from src.evaluation.metrics import macro_f05


def test_synthetic_end_to_end_matching():
    # Synthetic Source 1 Records
    s1_records = [
        {"id": "S1-1", "name": "Apple Inc", "address": "1 Infinite Loop, Cupertino CA", "country": "US"},
        {"id": "S1-2", "name": "Microsoft Corporation", "address": "One Microsoft Way, Redmond WA", "country": "US"},
        {"id": "S1-3", "name": "Unique Singleton Store", "address": "999 Remote Rd, Nowhere", "country": "US"},
    ]
    # Synthetic Source 2 Records
    s2_records = [
        {"id": "S2-1", "name": "Apple Computer Co.", "address": "Infinite Loop, Cupertino", "country": "US"},
        {"id": "S2-2", "name": "Microsoft Corp", "address": "1 Microsoft Way, Redmond", "country": "US"},
        {"id": "S2-99", "name": "Random Other Business", "address": "123 Elm St, Dallas", "country": "US"},
    ]

    s1_norm = [normalize_record(r["id"], r["name"], r["address"], r["country"]) for r in s1_records]
    s2_norm = [normalize_record(r["id"], r["name"], r["address"], r["country"]) for r in s2_records]

    # Ground truth
    gt = {
        "S1-1": {"S2-1"},
        "S1-2": {"S2-2"},
        "S1-3": set(),  # singleton
    }

    # Extract pairs and features
    X, y = [], []
    for s1 in s1_norm:
        for s2 in s2_norm:
            feats = compute_pairwise_features(s1, s2)
            row = [feats[col] for col in FEATURE_COLUMNS]
            X.append(row)
            is_match = 1 if s2["id"] in gt[s1["id"]] else 0
            y.append(is_match)

    X = np.array(X)
    y = np.array(y)

    # Fit a simple LogisticRegression baseline
    clf = LogisticRegression(random_state=42)
    clf.fit(X, y)

    # Predict probabilities
    probs = clf.predict_proba(X)[:, 1]
    
    # Threshold at 0.5
    predictions = {}
    pair_idx = 0
    for s1 in s1_norm:
        matched = set()
        for s2 in s2_norm:
            if probs[pair_idx] >= 0.5:
                matched.add(s2["id"])
            pair_idx += 1
        predictions[s1["id"]] = matched

    # Evaluate Macro F0.5
    eval_results = macro_f05(predictions, gt)
    assert eval_results["f05_macro"] == 1.0
    assert eval_results["singleton_f05"] == 1.0
    assert predictions["S1-1"] == {"S2-1"}
    assert predictions["S1-2"] == {"S2-2"}
    assert predictions["S1-3"] == set()
