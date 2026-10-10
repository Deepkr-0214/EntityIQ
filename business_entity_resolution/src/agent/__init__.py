"""
EntityIQ Agentic AI Package.
"""
from src.agent.orchestrator import EntityIQAgent, TOOL_REGISTRY
from src.agent.tools import (
    inspect_dataset_schema,
    validate_input_dataset,
    run_data_quality_report,
    run_entity_matching,
    explain_match_prediction,
    retrieve_project_knowledge,
    summarize_evaluation_report,
)

__all__ = [
    "EntityIQAgent",
    "TOOL_REGISTRY",
    "inspect_dataset_schema",
    "validate_input_dataset",
    "run_data_quality_report",
    "run_entity_matching",
    "explain_match_prediction",
    "retrieve_project_knowledge",
    "summarize_evaluation_report",
]
