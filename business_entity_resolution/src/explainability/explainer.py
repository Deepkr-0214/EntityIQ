"""
Explainable AI (XAI) Module for EntityIQ.

Provides:
1. Exact Tree SHAP contributions directly from LightGBM tree structure.
2. Linear coefficient attributions for Logistic Regression baseline.
3. Global feature importance summaries.
4. Record-level evidence explanations distinguishing raw scores, thresholds, and feature forces.
"""
import os
import pickle
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional

from src.config import MODELS_DIR
from src.preprocessing.normalization import normalize_record
from src.features.similarity_features import compute_pairwise_features, FEATURE_COLUMNS


class MatchExplainer:
    def __init__(self, model_name: str = "lightgbm"):
        self.model_name = model_name
        self.model = None
        self.scaler = None
        self.features = FEATURE_COLUMNS
        self._load_model()

    def _load_model(self):
        model_file = f"{self.model_name}.pkl"
        path = os.path.join(MODELS_DIR, model_file)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")

        with open(path, "rb") as f:
            data = pickle.load(f)
        self.model = data["model"]
        self.scaler = data.get("scaler")
        self.features = data.get("features", FEATURE_COLUMNS)

    def get_global_feature_importance(self, top_n: int = 10) -> List[Dict[str, Any]]:
        """Return ranked global feature importances."""
        if hasattr(self.model, "feature_importances_"):
            importances = self.model.feature_importances_
            total = sum(importances) if sum(importances) > 0 else 1.0
            ranked = sorted(
                [{"feature": f, "importance": int(imp), "relative_weight": round(imp / total, 4)} 
                 for f, imp in zip(self.features, importances)],
                key=lambda x: x["importance"],
                reverse=True
            )
            return ranked[:top_n]
        elif hasattr(self.model, "coef_"):
            coefs = self.model.coef_[0]
            ranked = sorted(
                [{"feature": f, "coefficient": round(float(c), 4), "abs_weight": round(abs(float(c)), 4)} 
                 for f, c in zip(self.features, coefs)],
                key=lambda x: x["abs_weight"],
                reverse=True
            )
            return ranked[:top_n]
        return []

    def explain_pair(self, s1_record: Dict[str, str], candidate_record: Dict[str, str], threshold: float = 0.70) -> Dict[str, Any]:
        """
        Produce a comprehensive, evidence-based local explanation for a single record pair.
        """
        # 1. Normalization
        s1_norm = normalize_record(
            s1_record.get("entity_id", "S1-query"),
            s1_record.get("business_name", ""),
            s1_record.get("business_address", ""),
            s1_record.get("country", "")
        )
        cand_norm = normalize_record(
            candidate_record.get("entity_id", "C-query"),
            candidate_record.get("business_name", ""),
            candidate_record.get("business_address", ""),
            candidate_record.get("country", "")
        )

        # 2. Extract features
        feats = compute_pairwise_features(s1_norm, cand_norm)
        feature_vector = np.array([[feats[col] for col in self.features]], dtype=np.float32)

        # 3. Model scoring
        if self.scaler is not None:
            feature_vector_scaled = self.scaler.transform(feature_vector)
            prob = float(self.model.predict_proba(feature_vector_scaled)[:, 1][0])
        else:
            prob = float(self.model.predict_proba(feature_vector)[:, 1][0])

        # 4. Feature Attributions / SHAP values
        attributions = []
        if hasattr(self.model, "booster_"):
            # Native LightGBM Tree SHAP calculation
            shap_values = self.model.booster_.predict(feature_vector, pred_contrib=True)[0]
            # shap_values has length len(features) + 1 (last is base value)
            base_value = float(shap_values[-1])
            for i, f_name in enumerate(self.features):
                attributions.append({
                    "feature": f_name,
                    "value": round(float(feature_vector[0][i]), 4),
                    "attribution": round(float(shap_values[i]), 4),
                    "impact": "positive" if shap_values[i] > 0 else "negative"
                })
        elif hasattr(self.model, "coef_"):
            coefs = self.model.coef_[0]
            for i, f_name in enumerate(self.features):
                attr = float(coefs[i] * feature_vector_scaled[0][i])
                attributions.append({
                    "feature": f_name,
                    "value": round(float(feature_vector[0][i]), 4),
                    "attribution": round(attr, 4),
                    "impact": "positive" if attr > 0 else "negative"
                })

        # Sort attributions by absolute impact
        attributions.sort(key=lambda x: abs(x["attribution"]), reverse=True)

        is_match = prob >= threshold

        return {
            "s1": {
                "id": s1_norm["id"],
                "raw_name": s1_record.get("business_name"),
                "normalized_name": s1_norm["na"],
                "raw_address": s1_record.get("business_address"),
                "normalized_address": s1_norm["aa"],
                "country": s1_norm["country"]
            },
            "candidate": {
                "id": cand_norm["id"],
                "raw_name": candidate_record.get("business_name"),
                "normalized_name": cand_norm["na"],
                "raw_address": candidate_record.get("business_address"),
                "normalized_address": cand_norm["aa"],
                "country": cand_norm["country"]
            },
            "prediction": {
                "match_probability": round(prob, 4),
                "decision_threshold": threshold,
                "is_match": is_match,
                "confidence_band": "High Match" if prob >= 0.85 else ("Moderate Match" if prob >= threshold else "Non-Match"),
            },
            "top_contributing_features": attributions[:8],
            "all_features": {k: round(v, 4) if isinstance(v, float) else v for k, v in feats.items()}
        }
