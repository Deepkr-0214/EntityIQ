"""
FastAPI REST Service for EntityIQ Business Entity Resolution Platform.
"""
import os
import time
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.agent.orchestrator import EntityIQAgent
from src.agent.tools import (
    inspect_dataset_schema,
    validate_input_dataset,
    run_data_quality_report,
    run_entity_matching,
    explain_match_prediction,
    retrieve_project_knowledge,
    summarize_evaluation_report,
)
from src.explainability.explainer import MatchExplainer

app = FastAPI(
    title="EntityIQ REST API",
    description="Agentic AI Business Entity Resolution & Explainability API",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

agent = EntityIQAgent()
explainer = MatchExplainer(model_name="lightgbm")

# ── Pydantic Request Models ───────────────────────────────────────────────────

class EntityRecord(BaseModel):
    entity_id: Optional[str] = "S1-001"
    business_name: str
    business_address: Optional[str] = ""
    country: Optional[str] = ""

class MatchRequest(BaseModel):
    s1_records: List[EntityRecord] = Field(..., max_length=1000)
    candidate_records: List[EntityRecord] = Field(..., max_length=5000)
    tau1: float = Field(0.25, ge=0.0, le=1.0)
    gap: float = Field(0.05, ge=0.0, le=1.0)

class ExplainRequest(BaseModel):
    s1_record: EntityRecord
    candidate_record: EntityRecord
    threshold: float = Field(0.70, ge=0.0, le=1.0)

class RAGQueryRequest(BaseModel):
    query: str
    top_k: int = Field(3, ge=1, le=10)

class AgentTaskRequest(BaseModel):
    intent: str
    parameters: Dict[str, Any] = {}

# ── API Endpoints ─────────────────────────────────────────────────────────────

@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {
        "status": "HEALTHY",
        "service": "EntityIQ API",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_loaded": explainer.model is not None,
    }


@app.post("/api/v1/match")
def match_entities(req: MatchRequest):
    """Run bounded entity matching with rank-aware thresholding."""
    try:
        s1_dicts = [r.model_dump() for r in req.s1_records]
        cand_dicts = [r.model_dump() for r in req.candidate_records]
        result = run_entity_matching(s1_dicts, cand_dicts, tau1=req.tau1, gap=req.gap)
        return {"status": "SUCCESS", "data": result}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/v1/explain")
def explain_pair(req: ExplainRequest):
    """Provide feature-level Tree SHAP attribution for a record pair."""
    try:
        explanation = explainer.explain_pair(
            req.s1_record.model_dump(),
            req.candidate_record.model_dump(),
            threshold=req.threshold
        )
        return {"status": "SUCCESS", "data": explanation}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/v1/rag/query")
def rag_query(req: RAGQueryRequest):
    """Ask questions grounded on verified project documentation."""
    try:
        res = retrieve_project_knowledge(req.query, top_k=req.top_k)
        return {"status": "SUCCESS", "data": res}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/v1/agent/run")
def run_agent_task(req: AgentTaskRequest):
    """Coordinate approved tool execution through the Agentic AI orchestrator."""
    try:
        res = agent.process_task(req.intent, **req.parameters)
        return res
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/v1/reports/summary")
def get_reports_summary():
    """Retrieve verified evaluation metrics and benchmarks."""
    summary = summarize_evaluation_report()
    return {"status": "SUCCESS", "data": summary}
