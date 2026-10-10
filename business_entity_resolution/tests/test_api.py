"""
Tests for FastAPI REST API endpoints.
"""
import pytest
from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)


def test_api_health():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "HEALTHY"
    assert data["model_loaded"] is True


def test_api_match():
    payload = {
        "s1_records": [
            {"entity_id": "S1-1", "business_name": "Tata Steel Ltd", "business_address": "Jamshedpur Road", "country": "India"}
        ],
        "candidate_records": [
            {"entity_id": "S2-1", "business_name": "Tata Steel", "business_address": "Jamshedpur", "country": "India"},
            {"entity_id": "S2-99", "business_name": "Random Shop", "business_address": "Kolkata", "country": "India"}
        ],
        "tau1": 0.25,
        "gap": 0.05
    }
    response = client.post("/api/v1/match", json=payload)
    assert response.status_code == 200
    res = response.json()
    assert res["status"] == "SUCCESS"
    assert "S1-1" in res["data"]["matching_results"]


def test_api_explain():
    payload = {
        "s1_record": {"entity_id": "S1-1", "business_name": "Wipro Limited", "business_address": "Sarjapur Road, Bangalore", "country": "India"},
        "candidate_record": {"entity_id": "S2-1", "business_name": "Wipro Ltd", "business_address": "Sarjapur, Bengaluru", "country": "India"},
        "threshold": 0.70
    }
    response = client.post("/api/v1/explain", json=payload)
    assert response.status_code == 200
    res = response.json()
    assert res["status"] == "SUCCESS"
    assert "match_probability" in res["data"]["prediction"]


def test_api_rag_query():
    payload = {"query": "What is the macro F0.5 score?", "top_k": 2}
    response = client.post("/api/v1/rag/query", json=payload)
    assert response.status_code == 200
    res = response.json()
    assert res["status"] == "SUCCESS"
    assert len(res["data"]["sources"]) > 0


def test_api_agent_run():
    payload = {
        "intent": "evaluation_metrics",
        "parameters": {}
    }
    response = client.post("/api/v1/agent/run", json=payload)
    assert response.status_code == 200
    assert response.json()["status"] == "SUCCESS"


def test_api_reports_summary():
    response = client.get("/api/v1/reports/summary")
    assert response.status_code == 200
    assert response.json()["status"] == "SUCCESS"
