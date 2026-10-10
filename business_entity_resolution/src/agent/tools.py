"""
Typed and Validated Tools for EntityIQ Agentic AI System.

Exposes safe, bounded data processing, matching, knowledge retrieval, and evaluation tools.
"""
import os
import json
import pickle
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional

from src.config import BASE_DIR, MODELS_DIR, REPORTS_DIR
from src.preprocessing.normalization import normalize_record
from src.features.similarity_features import compute_pairwise_features, FEATURE_COLUMNS
from src.modeling.threshold import apply_rank_aware_threshold
from src.rag.qa_engine import RAGAssistant

# ── Approved Tool Registry ────────────────────────────────────────────────────

def inspect_dataset_schema(file_path: str) -> Dict[str, Any]:
    """Inspect columns, dtypes, row counts, and nulls for a TSV/CSV file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset path does not exist: {file_path}")

    # Read top 500 rows for lightweight schema inspection
    sep = "\t" if file_path.endswith(".tsv") else ","
    df = pd.read_csv(file_path, sep=sep, nrows=500, dtype=str, keep_default_na=False)
    
    # Fast row counting
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        total_rows = sum(1 for _ in f) - 1

    return {
        "file_name": os.path.basename(file_path),
        "total_rows": max(0, total_rows),
        "columns": list(df.columns),
        "sample_preview": df.head(3).to_dict(orient="records"),
    }


def validate_input_dataset(file_path: str) -> Dict[str, Any]:
    """Validate dataset structure against entity resolution formatting rules."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset path does not exist: {file_path}")

    sep = "\t" if file_path.endswith(".tsv") else ","
    df = pd.read_csv(file_path, sep=sep, nrows=1000, dtype=str, keep_default_na=False)

    required_cols = {"entity_id", "business_name", "business_address", "country"}
    present_cols = set(df.columns)
    missing_cols = list(required_cols - present_cols)

    id_prefix_valid = True
    sample_ids = df["entity_id"].tolist() if "entity_id" in df.columns else []
    for eid in sample_ids[:50]:
        if not any(eid.startswith(prefix) for prefix in ["S1-", "S2-", "S3-"]):
            id_prefix_valid = False
            break

    is_valid = (len(missing_cols) == 0) and id_prefix_valid

    return {
        "file_name": os.path.basename(file_path),
        "is_valid": is_valid,
        "missing_columns": missing_cols,
        "id_prefix_compliant": id_prefix_valid,
        "checked_sample_size": len(df),
    }


def run_data_quality_report(file_path: str) -> Dict[str, Any]:
    """Generate summary data quality metrics (empty addresses, countries)."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset path does not exist: {file_path}")

    sep = "\t" if file_path.endswith(".tsv") else ","
    df = pd.read_csv(file_path, sep=sep, nrows=2000, dtype=str, keep_default_na=False)

    empty_names = int(sum(df["business_name"].str.strip() == "")) if "business_name" in df.columns else 0
    empty_addrs = int(sum(df["business_address"].str.strip() == "")) if "business_address" in df.columns else 0
    country_counts = df["country"].value_counts().to_dict() if "country" in df.columns else {}

    return {
        "sample_size": len(df),
        "empty_business_names": empty_names,
        "empty_addresses": empty_addrs,
        "empty_address_percentage": round(empty_addrs / len(df) * 100, 2) if len(df) > 0 else 0,
        "country_distribution": country_counts,
    }


def run_entity_matching(s1_records: List[Dict[str, str]], 
                        candidate_records: List[Dict[str, str]], 
                        tau1: float = 0.25, 
                        gap: float = 0.05) -> Dict[str, Any]:
    """
    Run bounded entity matching for a list of S1 records against candidates.
    Enforces maximum batch size limit (<= 1,000 S1 records) for memory safety.
    """
    if len(s1_records) > 1000:
        raise ValueError("Batch size exceeds safety limit of 1,000 S1 entities. Process in batches.")

    model_path = os.path.join(MODELS_DIR, "lightgbm.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError("Trained LightGBM model artifact not found in models/lightgbm.pkl")

    with open(model_path, "rb") as f:
        model_data = pickle.load(f)
    model = model_data["model"]

    s1_norm = [normalize_record(r.get("entity_id", f"S1-{i}"), r.get("business_name", ""), r.get("business_address", ""), r.get("country", "")) 
               for i, r in enumerate(s1_records)]
    cand_norm = [normalize_record(r.get("entity_id", f"C-{i}"), r.get("business_name", ""), r.get("business_address", ""), r.get("country", "")) 
                 for i, r in enumerate(candidate_records)]

    pair_scores = []
    for s1 in s1_norm:
        for cand in cand_norm:
            # Skip if country differs
            if s1["country"] and cand["country"] and s1["country"] != cand["country"]:
                continue
            feats = compute_pairwise_features(s1, cand)
            feature_vector = np.array([[feats[col] for col in FEATURE_COLUMNS]], dtype=np.float32)
            prob = float(model.predict_proba(feature_vector)[:, 1][0])
            pair_scores.append((s1["id"], cand["id"], prob))

    matches = apply_rank_aware_threshold(pair_scores, tau1=tau1, gap_threshold=gap)
    
    # Format matches into JSON-serializable output
    results = {}
    for s1 in s1_norm:
        s1_id = s1["id"]
        matched_cands = sorted(list(matches.get(s1_id, set())))
        results[s1_id] = matched_cands

    return {
        "processed_s1_count": len(s1_norm),
        "total_candidates": len(cand_norm),
        "total_pairs_scored": len(pair_scores),
        "matching_results": results,
    }


def explain_match_prediction(s1_record: Dict[str, str], candidate_record: Dict[str, str]) -> Dict[str, Any]:
    """Provide feature-level evidence and probability score for a record pair."""
    s1_norm = normalize_record(s1_record.get("entity_id", "S1-query"), s1_record.get("business_name", ""), s1_record.get("business_address", ""), s1_record.get("country", ""))
    cand_norm = normalize_record(candidate_record.get("entity_id", "C-query"), candidate_record.get("business_name", ""), candidate_record.get("business_address", ""), candidate_record.get("country", ""))

    feats = compute_pairwise_features(s1_norm, cand_norm)
    
    model_path = os.path.join(MODELS_DIR, "lightgbm.pkl")
    prob = 0.0
    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            model = pickle.load(f)["model"]
        feature_vector = np.array([[feats[col] for col in FEATURE_COLUMNS]], dtype=np.float32)
        prob = float(model.predict_proba(feature_vector)[:, 1][0])

    return {
        "s1_entity": {"id": s1_norm["id"], "name": s1_record.get("business_name"), "address": s1_record.get("business_address"), "country": s1_norm["country"]},
        "candidate_entity": {"id": cand_norm["id"], "name": candidate_record.get("business_name"), "address": candidate_record.get("business_address"), "country": cand_norm["country"]},
        "match_probability": round(prob, 4),
        "is_match_predicted": prob >= 0.70,
        "key_similarity_features": {
            "name_jaccard": round(feats["name_jaccard"], 4),
            "name_token_sort": round(feats["name_token_sort"], 4),
            "name_monge_elkan": round(feats["name_monge_elkan"], 4),
            "addr_jaccard": round(feats["addr_jaccard"], 4),
            "addr_numeric_match": feats["addr_numeric_match"],
            "country_match": feats["country_match"],
        }
    }


def retrieve_project_knowledge(query: str, top_k: int = 3) -> Dict[str, Any]:
    """Query the verified project documentation and evaluation reports via RAG."""
    assistant = RAGAssistant(base_dir=BASE_DIR, auto_build=True)
    return assistant.answer_question(query=query, top_k=top_k)


def summarize_evaluation_report() -> Dict[str, Any]:
    """Retrieve verified evaluation metrics and experiment comparisons."""
    meta_path = os.path.join(MODELS_DIR, "model_metadata.json")
    if not os.path.exists(meta_path):
        return {"error": "No evaluation report metadata found. Run Phase 3 training first."}

    with open(meta_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    return {
        "evaluation_timestamp": metadata.get("timestamp"),
        "sample_size": metadata.get("sample_size"),
        "blocking_recall": f"{metadata.get('blocking_recall', 0):.2%}",
        "avg_candidates_per_entity": metadata.get("avg_candidates_per_entity"),
        "model_metrics": metadata.get("results"),
    }
