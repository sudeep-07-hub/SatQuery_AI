import pytest
import asyncio
from unittest.mock import patch, MagicMock

from job_manager import job_registry, execute_agentic_pipeline
from mc5_evidence.evidence_graph import EvidenceGraph

def test_multitool_orchestration():
    job_id = job_registry.create_job()
    
    # A query that implies multiple intents: describing scene (VQA/captioning) AND change detection.
    # We provide 2 images to satisfy the change detection temporal requirement.
    query = "Describe the scene in optical and determine if there are changes between the two dates."
    files = [("image_1_opt.tif", b"fake"), ("image_2_opt.tif", b"fake2")]
    
    files = [("image_1_opt.tif", b"fake"), ("image_2_opt.tif", b"fake2")]
    
    mock_mc1_profile = {
        "task_executable": True,
        "spatial_overlap": 0.95,
        "coregistration_score": 0.92,
        "image_1": {
            "filename": "image_1_opt.tif", "modality": "optical",
            "crs": "EPSG:32643", "gsd_m": 1.0, "quality_score": 1.0,
            "acquisition_date": "2020-01-01T00:00:00Z"
        },
        "image_2": {
            "filename": "image_2_opt.tif", "modality": "optical",
            "crs": "EPSG:32643", "gsd_m": 1.0, "quality_score": 1.0,
            "acquisition_date": "2024-01-01T00:00:00Z"
        },
        "image_count": 2,
        "affine_transform": [1.0, 0, 300000.0, 0, -1.0, 4000000.0]
    }
    
    # Run the pipeline in fixture mode to prevent live downloads.
    with patch('job_manager.run_mc1_pipeline', return_value=mock_mc1_profile):
        asyncio.run(execute_agentic_pipeline(job_id, files, query, execution_mode="fixture"))
    
    job = job_registry.get_job(job_id)
    
    # 1. Pipeline should complete successfully
    assert job["status"] == "DONE"
    
    # 2. Extract executed tools from the trace
    trace = job["progress_trace"]
    executed_tools = set()
    for t in trace:
        if t["stage"] == "AGENTIC_EXECUTION" and "tool_id" in t.get("message", ""):
            pass # We can check evidence objects directly
            
    # The actual execution state must prove it.
    evidence_objects = job.get("evidence_objects", [])
    assert len(evidence_objects) >= 2, "Must generate at least 2 pieces of evidence"
    
    # Check that the tools belong to distinct capabilities/tasks
    specialists_used = {ev.get("source_model") for ev in evidence_objects if ev.get("source_model")}
    assert len(specialists_used) >= 2, "Must use at least two distinct specialist capabilities"
    
    # Check the graph
    graph_nodes = job.get("evidence_graph", {}).get("nodes", [])
    evidence_nodes = [n for n in graph_nodes if n.get("type") == "evidence"]
    assert len(evidence_nodes) >= 2, "Must insert at least 2 evidence nodes into the Evidence Graph"

    # Verify both outputs survived in AgentState
    assert "evidence_references" in job["result"]
    assert len(job["result"]["evidence_references"]) >= 2
    
def test_multitool_failure_replanning():
    job_id = job_registry.create_job()
    query = "Describe the scene in optical and determine if there are changes between the two dates."
    files = [("image_1_opt.tif", b"fake"), ("image_2_opt.tif", b"fake2")]
    
    # To test failure, we simulate one tool failing.
    # ChangeMambaExecutionAdapter will return MODEL_UNAVAILABLE if run in real mode without weights.
    # Wait, the prompt says "Create one controlled test where Tool A succeeds, Tool B fails or becomes unavailable."
    # Let's mock ChangeMambaExecutionAdapter's execution to raise an error.
    from agent.adapters import ChangeMambaExecutionAdapter
    from agent.schemas import ToolResult
    
    original_execute = ChangeMambaExecutionAdapter.execute
    def mock_execute(self, call, mc1_profile, image_tensors):
        return ToolResult(
            call_id=call.call_id,
            tool_id=call.tool_id,
            status="failed",
            outputs={"status": "MODEL_UNAVAILABLE"},
            evidence_references=[],
            error_information="MODEL_UNAVAILABLE: simulated failure for test",
            execution_metadata={}
        )

    mock_mc1_profile = {
        "task_executable": True,
        "spatial_overlap": 0.95,
        "coregistration_score": 0.92,
        "image_1": {
            "filename": "image_1_opt.tif", "modality": "optical",
            "crs": "EPSG:32643", "gsd_m": 1.0, "quality_score": 1.0,
            "acquisition_date": "2020-01-01T00:00:00Z"
        },
        "image_2": {
            "filename": "image_2_opt.tif", "modality": "optical",
            "crs": "EPSG:32643", "gsd_m": 1.0, "quality_score": 1.0,
            "acquisition_date": "2024-01-01T00:00:00Z"
        },
        "image_count": 2,
        "affine_transform": [1.0, 0, 300000.0, 0, -1.0, 4000000.0]
    }

    with patch.object(ChangeMambaExecutionAdapter, 'execute', mock_execute), \
         patch('job_manager.run_mc1_pipeline', return_value=mock_mc1_profile):
        asyncio.run(execute_agentic_pipeline(job_id, files, query, execution_mode="fixture"))
        
    job = job_registry.get_job(job_id)
    
    # Even if Tool B fails, Tool A's evidence should be preserved, and the system might return PARTIAL or DONE.
    assert job["status"] in ["DONE", "PARTIAL_EXECUTION", "MODEL_UNAVAILABLE", "INSUFFICIENT_EVIDENCE"]
    
    evidence_objects = job.get("evidence_objects", [])
    # Tool A (PaliGemma) should succeed
    assert len(evidence_objects) >= 1
    
    specialists_used = {ev.get("source_model") for ev in evidence_objects if ev.get("source_model")}
    assert any("PALIGEMMA" in s for s in specialists_used), "single_image_vqa (PaliGemma) evidence should be retained despite ChangeMamba failure"
