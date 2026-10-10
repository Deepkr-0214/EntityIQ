"""
Tests for model loading, inference scoring, and rank-aware thresholding.
"""
import pytest
import os
import pickle
import numpy as np
from src.config import MODELS_DIR
from src.features.similarity_features import FEATURE_COLUMNS
from src.modeling.threshold import apply_threshold, apply_rank_aware_threshold


def test_saved_model_artifacts_exist():
    lr_path = os.path.join(MODELS_DIR, "logistic_regression.pkl")
    lgb_path = os.path.join(MODELS_DIR, "lightgbm.pkl")
    meta_path = os.path.join(MODELS_DIR, "model_metadata.json")

    assert os.path.exists(lr_path), "Logistic Regression artifact missing"
    assert os.path.exists(lgb_path), "LightGBM artifact missing"
    assert os.path.exists(meta_path), "Model metadata JSON missing"


def test_model_inference_scoring():
    lgb_path = os.path.join(MODELS_DIR, "lightgbm.pkl")
    with open(lgb_path, "rb") as f:
        data = pickle.load(f)

    model = data["model"]
    features = data["features"]

    # Synthetic sample pair features
    dummy_features = np.ones((2, len(features)), dtype=np.float32)
    probs = model.predict_proba(dummy_features)[:, 1]

    assert len(probs) == 2
    assert all(0.0 <= p <= 1.0 for p in probs)


def test_rank_aware_threshold_behavior():
    pair_scores = [
        ("S1-1", "S2-1", 0.95),
        ("S1-1", "S3-1", 0.92),
        ("S1-1", "S2-99", 0.40),  # big drop
        ("S1-2", "S2-5", 0.15),   # below tau1 -> singleton
    ]

    preds = apply_rank_aware_threshold(pair_scores, tau1=0.25, gap_threshold=0.10)
    assert preds["S1-1"] == {"S2-1", "S3-1"}
    assert preds["S1-2"] == set()
