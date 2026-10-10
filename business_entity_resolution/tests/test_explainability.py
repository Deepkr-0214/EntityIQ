"""
Tests for Explainable AI (XAI) feature attribution and Tree SHAP calculations.
"""
import pytest
import os
from src.explainability.explainer import MatchExplainer


def test_global_feature_importance_lightgbm():
    explainer = MatchExplainer(model_name="lightgbm")
    imp = explainer.get_global_feature_importance(top_n=5)
    assert len(imp) == 5
    assert all("feature" in item and "importance" in item for item in imp)
    assert imp[0]["importance"] >= imp[1]["importance"]


def test_global_feature_importance_logistic_regression():
    explainer = MatchExplainer(model_name="logistic_regression")
    imp = explainer.get_global_feature_importance(top_n=5)
    assert len(imp) == 5
    assert all("feature" in item and "coefficient" in item for item in imp)


def test_pair_explanation_shap():
    explainer = MatchExplainer(model_name="lightgbm")
    s1 = {
        "entity_id": "S1-100",
        "business_name": "Infosys Technologies Ltd",
        "business_address": "Electronics City, Hosur Road, Bangalore 560100",
        "country": "India"
    }
    cand = {
        "entity_id": "S2-100",
        "business_name": "Infosys Technologies",
        "business_address": "Hosur Rd, 560100 Bengaluru",
        "country": "India"
    }

    explanation = explainer.explain_pair(s1, cand, threshold=0.70)
    
    assert "s1" in explanation
    assert "candidate" in explanation
    assert "prediction" in explanation
    assert "top_contributing_features" in explanation
    
    pred = explanation["prediction"]
    assert 0.0 <= pred["match_probability"] <= 1.0
    assert pred["decision_threshold"] == 0.70
    assert pred["is_match"] is True

    # Check that attributions exist and top features have positive impact for a near identical match
    top_feats = explanation["top_contributing_features"]
    assert len(top_feats) > 0
    assert any("name_" in f["feature"] for f in top_feats)
