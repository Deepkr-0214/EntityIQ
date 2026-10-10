"""
Agentic AI Orchestrator for EntityIQ.

Coordinates approved tool execution with safety policies, input validation, and execution logging.
"""
import time
from typing import Dict, Any, List, Optional
from src.agent.tools import (
    inspect_dataset_schema,
    validate_input_dataset,
    run_data_quality_report,
    run_entity_matching,
    explain_match_prediction,
    retrieve_project_knowledge,
    summarize_evaluation_report,
)

TOOL_REGISTRY = {
    "inspect_dataset_schema": inspect_dataset_schema,
    "validate_input_dataset": validate_input_dataset,
    "run_data_quality_report": run_data_quality_report,
    "run_entity_matching": run_entity_matching,
    "explain_match_prediction": explain_match_prediction,
    "retrieve_project_knowledge": retrieve_project_knowledge,
    "summarize_evaluation_report": summarize_evaluation_report,
}


class EntityIQAgent:
    def __init__(self):
        self.execution_log: List[Dict[str, Any]] = []

    def execute_tool(self, tool_name: str, **kwargs) -> Dict[str, Any]:
        """Safely execute an approved tool with input validation and execution tracking."""
        if tool_name not in TOOL_REGISTRY:
            raise ValueError(f"Unauthorized tool request: '{tool_name}'. Permitted tools: {list(TOOL_REGISTRY.keys())}")

        tool_func = TOOL_REGISTRY[tool_name]
        start_time = time.time()
        
        try:
            result = tool_func(**kwargs)
            duration = round(time.time() - start_time, 4)
            
            log_entry = {
                "tool": tool_name,
                "status": "SUCCESS",
                "duration_sec": duration,
                "args_keys": list(kwargs.keys()),
            }
            self.execution_log.append(log_entry)

            return {
                "tool": tool_name,
                "status": "SUCCESS",
                "duration_seconds": duration,
                "data": result,
            }

        except Exception as e:
            duration = round(time.time() - start_time, 4)
            log_entry = {
                "tool": tool_name,
                "status": "ERROR",
                "error": str(e),
                "duration_sec": duration,
            }
            self.execution_log.append(log_entry)

            return {
                "tool": tool_name,
                "status": "ERROR",
                "error_message": str(e),
                "duration_seconds": duration,
            }

    def process_task(self, intent: str, **kwargs) -> Dict[str, Any]:
        """
        Map a user intent to the appropriate tool and execute with safety verification.
        """
        # Intent routing map
        routing = {
            "inspect_schema": "inspect_dataset_schema",
            "validate_data": "validate_input_dataset",
            "quality_report": "run_data_quality_report",
            "match_records": "run_entity_matching",
            "explain_pair": "explain_match_prediction",
            "rag_query": "retrieve_project_knowledge",
            "evaluation_metrics": "summarize_evaluation_report",
        }

        tool_name = routing.get(intent, intent)
        return self.execute_tool(tool_name, **kwargs)

    def get_audit_trail(self) -> List[Dict[str, Any]]:
        """Return the immutable audit trail of all executed tools."""
        return list(self.execution_log)
