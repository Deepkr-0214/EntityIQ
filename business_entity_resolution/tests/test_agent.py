"""
Tests for Agentic AI system, tool routing, safety boundaries, and audit logging.
"""
import pytest
import os
from src.agent.orchestrator import EntityIQAgent
from src.config import TRAIN_S1


def test_agent_tool_rejection_unauthorized():
    agent = EntityIQAgent()
    with pytest.raises(ValueError, match="Unauthorized tool request"):
        agent.execute_tool("delete_database_records")


def test_agent_inspect_schema_tool():
    agent = EntityIQAgent()
    res = agent.process_task("inspect_schema", file_path=TRAIN_S1)
    assert res["status"] == "SUCCESS"
    data = res["data"]
    assert "entity_id" in data["columns"]
    assert data["total_rows"] > 0


def test_agent_validate_dataset_tool():
    agent = EntityIQAgent()
    res = agent.process_task("validate_data", file_path=TRAIN_S1)
    assert res["status"] == "SUCCESS"
    assert res["data"]["is_valid"] is True


def test_agent_run_matching_tool():
    agent = EntityIQAgent()
    s1_sample = [
        {"entity_id": "S1-10", "business_name": "Tata Motors Ltd", "business_address": "Mumbai Road", "country": "India"}
    ]
    cand_sample = [
        {"entity_id": "S2-10", "business_name": "Tata Motors", "business_address": "Mumbai", "country": "India"},
        {"entity_id": "S2-99", "business_name": "Random Shop", "business_address": "Delhi", "country": "India"},
    ]

    res = agent.process_task("match_records", s1_records=s1_sample, candidate_records=cand_sample)
    assert res["status"] == "SUCCESS"
    assert "S1-10" in res["data"]["matching_results"]
    assert "S2-10" in res["data"]["matching_results"]["S1-10"]


def test_agent_explain_pair_tool():
    agent = EntityIQAgent()
    s1 = {"entity_id": "S1-1", "business_name": "Acme Corp", "business_address": "123 Elm St", "country": "US"}
    cand = {"entity_id": "S2-1", "business_name": "Acme Corporation", "business_address": "123 Elm Street", "country": "US"}

    res = agent.process_task("explain_pair", s1_record=s1, candidate_record=cand)
    assert res["status"] == "SUCCESS"
    assert "match_probability" in res["data"]
    assert res["data"]["match_probability"] > 0.5


def test_agent_audit_trail():
    agent = EntityIQAgent()
    agent.process_task("evaluation_metrics")
    trail = agent.get_audit_trail()
    assert len(trail) == 1
    assert trail[0]["tool"] == "summarize_evaluation_report"
    assert trail[0]["status"] == "SUCCESS"
