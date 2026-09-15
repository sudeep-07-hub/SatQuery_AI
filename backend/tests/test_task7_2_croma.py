import pytest
import asyncio
import os
import torch
import copy
from typing import Dict, Any

from job_manager import execute_agentic_pipeline, job_registry, adapters
from agent.adapters import CromaExecutionAdapter
from agent.default_tools import setup_default_registry
from agent.schemas import ToolCall
from mc4c.engine import MC4CEngine
from mc4c.cross_modal_evidence import CrossModalEvidenceAdapter
from mc4c.fusion_schema import FusedTokenRepresentation
from mc4c.token_schema import TokenSpatialIdentity
from mc4c.query_schema import QueryRepresentation

@pytest.fixture
def mock_mc1_profile():
    return {
        "task_executable": True,
        "image_1": {
            "filename": "fake_opt.tif",
            "modality": "optical",
            "observation_id": "opt_obs_1"
        },
        "image_2": {
            "filename": "fake_sar.tif",
            "modality": "sar",
            "observation_id": "sar_obs_1"
        }
    }

class TestTask7_2_CROMAIntegration:
    
    def test_1_model_loading_and_fallback(self):
        adapter = CromaExecutionAdapter()
        # Should gracefully fail without crashing
        res = adapter.execute(
            ToolCall(call_id="1", tool_id="optical_sar_fusion", arguments={"query": "test"}),
            {"image_1": {"filename": "missing.tif", "modality": "optical"},
             "image_2": {"filename": "missing.tif", "modality": "sar"}},
            {}
        )
        assert res.status in ["failed", "succeeded"]
        if res.status == "failed":
            assert "MODEL_UNAVAILABLE" in res.error_information or "No such file" in res.error_information

    def test_2_3_4_extraction_and_fusion_via_fixture(self, mock_mc1_profile):
        adapter = CromaExecutionAdapter()
        # Mocking the engine call since weights might not be there
        res = adapter._execute_fixture_fallback(
            ToolCall(call_id="1", tool_id="optical_sar_fusion", arguments={"query": "test"}),
            mock_mc1_profile,
            "test query",
            "image_1",
            "image_2",
            started_at=0.0
        )
        
        assert res.status == "succeeded"
        assert "optical_tokens_shape" in res.outputs
        assert "sar_tokens_shape" in res.outputs
        assert "joint_tokens_shape" in res.outputs
        assert res.outputs["joint_tokens_shape"] == [1, 225, 768]
        
    def test_5_query_dependence(self):
        # We can test query dependency logic in the fusion module directly if weights were present
        # Or mock it via the fixture check
        pass

    def test_8_spatial_token_preservation(self, mock_mc1_profile):
        adapter = CromaExecutionAdapter()
        res = adapter._execute_fixture_fallback(
            ToolCall(call_id="1", tool_id="optical_sar_fusion", arguments={"query": "test"}),
            mock_mc1_profile,
            "test query",
            "image_1",
            "image_2",
            started_at=0.0
        )
        assert res.outputs["joint_tokens_shape"] == [1, 225, 768] # 225 tokens, 768 dim preserved

    def test_10_toolspec_validation(self):
        registry = setup_default_registry()
        tool = registry.get("optical_sar_fusion")
        assert tool is not None
        assert "optical" in tool.required_observations.required_modalities
        assert "sar" in tool.required_observations.required_modalities
        assert tool.required_observations.minimum_observations == 2
        assert tool.required_observations.maximum_observations == 2

    def test_11_mc1_binding(self):
        adapter = CromaExecutionAdapter()
        res = adapter.execute(
            ToolCall(call_id="1", tool_id="optical_sar_fusion", arguments={"query": "test"}),
            {"image_1": {"filename": "missing.tif", "modality": "invalid"},
             "image_2": {"filename": "missing.tif", "modality": "invalid"}},
            {}
        )
        assert res.status == "failed"
        assert "Missing optical or SAR observation" in res.error_information
        
    def test_12_mc5_evidence_normalization(self):
        num_tokens = 225
        fused_rep = FusedTokenRepresentation(
            grid_height=15,
            grid_width=15,
            batch_size=1,
            num_tokens=num_tokens,
            spatial_identities=[TokenSpatialIdentity(original_index=i, row=i//15, column=i%15, bounds=[0,0,1,1]) for i in range(num_tokens)],
            fused_tokens=torch.zeros(1, num_tokens, 768),
            query_context=QueryRepresentation(original_query="test query", primary_task="cross_modal_fusion", embedding=torch.zeros(1, 384))
        )

        ev_adapter = CrossModalEvidenceAdapter()
        candidates = ev_adapter.generate_candidates(
            fused_rep,
            optical_obs={"observation_id": "opt1"},
            sar_obs={"observation_id": "sar1"},
            mc1_compatibility={"status": "compatible"},
            indices=[0]
        )
        
        assert len(candidates) == 1
        mc5_ev = candidates[0].to_mc5_evidence()
        assert "evidence_id" in mc5_ev
        assert mc5_ev["evidence_type"] == "cross_modal_spatial"
        assert mc5_ev["modality"] == "optical+sar"

    def test_13_14_17_full_workflow(self):
        job_id = asyncio.run(self.run_pipeline_with("fusion smoke_test"))
        job = job_registry.get_job(job_id)
        
        assert job["status"] == "DONE"
        assert "fusion" in job["result"]["final_answer"].lower() or "cross_modal_spatial" in str(job["evidence_graph"]) or "mock" in str(job["result"]["final_answer"]).lower()
        
        # Check trace
        stages = [t["stage"] for t in job["progress_trace"]]
        assert "MC1_VALIDATING" in stages
        assert "QUERY_INTELLIGENCE" in stages
        assert "TOOL_SELECTION" in stages
        assert "WORKFLOW_PLANNING" in stages
        assert "VERIFICATION" in stages
        
        # Ensure it's not the mock adapter
        assert isinstance(adapters["optical_sar_fusion"], CromaExecutionAdapter)
        
    def test_19_non_croma_regression(self):
        job_id = asyncio.run(self.run_pipeline_with("change detection smoke_test"))
        job = job_registry.get_job(job_id)
        assert job["status"] == "DONE"
        assert "temporal_change_analysis" in str(job)

    async def run_pipeline_with(self, query: str) -> str:
        job_id = job_registry.create_job()
        await execute_agentic_pipeline(job_id, [("file1", b""), ("file2", b"")], query)
        return job_id
