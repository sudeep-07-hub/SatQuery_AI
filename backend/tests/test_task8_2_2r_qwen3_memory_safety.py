import os
import pytest
import asyncio
from unittest.mock import patch, MagicMock

from job_manager import execute_agentic_pipeline, job_registry
from qwen.inference import Qwen3Inference


def test_qwen3_single_instance_lifecycle():
    """
    Validates INVARIANT_1, INVARIANT_2, INVARIANT_3:
    - At most one Qwen3Inference object exists per real-mode job.
    - Qwen3 .load() is called exactly once.
    - All Qwen-driven components share the SAME loaded engine.
    """
    async def run_test():
        job_id = job_registry.create_job()
        
        # We will track the id() of the instantiated Qwen3Inference
        created_engine_id = None
        
        # Mock Qwen3Inference.__init__ to track instantiation
        original_init = Qwen3Inference.__init__
        init_calls = 0
        
        def mock_init(self, *args, **kwargs):
            nonlocal init_calls, created_engine_id
            init_calls += 1
            created_engine_id = id(self)
            original_init(self, *args, **kwargs)
            
        with patch.object(Qwen3Inference, "__init__", side_effect=mock_init, autospec=True), \
             patch("qwen.inference.Qwen3Inference.load") as mock_load, \
             patch("job_manager.QueryIntelligencePipeline.__init__", return_value=None) as mock_qi_init, \
             patch("job_manager.QueryIntelligencePipeline.process_query") as mock_qi_process, \
             patch("job_manager.QwenToolSelector.__init__", return_value=None) as mock_selector_init, \
             patch("job_manager.QwenToolSelector.select_tools") as mock_selector_process, \
             patch("agent.recovery_planner.QwenRecoveryPlanner.__init__", return_value=None) as mock_recovery_init, \
             patch("job_manager.run_mc1_pipeline") as mock_mc1:
             
            # Mock MC1 success
            mock_mc1.return_value = {
                "task_executable": True,
                "spatial_overlap": 1.0,
                "coregistration_score": 1.0,
                "image_1": {"filename": "img1.tif", "modality": "optical"}
            }
            
            # Setup QI mock
            from qwen.schemas import TaskSpec, ObservationRequirement
            from qwen.pipeline import QueryIntelligenceResult
            mock_qi_process.return_value = QueryIntelligenceResult(
                original_query="What is here?",
                primary_task_spec=TaskSpec(query="What is here?", primary_task="single_image_vqa", target_entities=[], required_modalities=["optical"], temporal_requirement="none", spatial_output_required=False, textual_output_required=True, ambiguous=False),
                is_compound=False, subtasks=[],
                observation_requirements=[ObservationRequirement(requirement_id="req1", source_subtasks=[], minimum_observations=1, maximum_observations=1, required_modalities=["optical"], temporal_relationship="none")],
                ambiguous=False
            )
            
            # Setup Selector mock
            from agent.schemas import ToolCall
            mock_selector_process.return_value = [ToolCall(call_id="call1", tool_id="single_image_vqa", arguments={"query": "What is here?"})]

            with patch("job_manager.AgentController.execute") as mock_execute:
                mock_execute.return_value = MagicMock()
                mock_execute.return_value.status = "SUCCESS"
                mock_execute.return_value.completed_calls = []
                mock_execute.return_value.failed_calls = []
                mock_execute.return_value.results = {}
                mock_execute.return_value.replan_count = 0
                
                with patch("job_manager.RecoveryManager.execute_with_recovery") as mock_recovery:
                    mock_recovery.return_value = mock_execute.return_value
                    
                    # Also patch generate for final synthesis
                    with patch("qwen.inference.Qwen3Inference.generate") as mock_generate:
                        mock_generate.return_value = {"status": "ok", "response": '{"answer": "A building", "evidence_ids": [], "uncertainty": "none"}'}
                        
                        await execute_agentic_pipeline(job_id, [("test.tif", open(os.path.join(os.path.dirname(__file__), "fixtures", "test_geo1.tif"), "rb").read())], "What is here?", execution_mode="real", qwen_backend="transformers", planner_fallback="strict")
                    
                    # INVARIANT 1: Instantiated at most once
                    assert init_calls == 1, f"Qwen3Inference instantiated {init_calls} times instead of 1."
                    
                    # INVARIANT 2: Loaded exactly once
                    assert mock_load.call_count == 1, f"Qwen3Inference.load() called {mock_load.call_count} times."
                    
                    # INVARIANT 3: Object Identity
                    # Query Intelligence
                    qi_engine = mock_qi_init.call_args[0][0]
                    assert id(qi_engine) == created_engine_id, "Query Intelligence received a different Qwen3Inference object."
                    
                    # Tool Selector
                    selector_engine = mock_selector_init.call_args[0][0]
                    assert id(selector_engine) == created_engine_id, "Tool Selector received a different Qwen3Inference object."
                    
                    # Recovery Planner
                    assert mock_recovery_init.called, "RecoveryPlanner was not instantiated."
                    recovery_engine = mock_recovery_init.call_args[0][0]
                    assert id(recovery_engine) == created_engine_id, "Recovery Planner received a different Qwen3Inference object."
                    
                    job = job_registry.get_job(job_id)
                    assert job["status"] in ["DONE", "INSUFFICIENT_EVIDENCE"], f"Job failed: {job.get('result')}"

    asyncio.run(run_test())

def test_qwen3_explicit_unload():
    """
    Verifies that execute_agentic_pipeline explicitly calls unload() to free VRAM.
    """
    async def run_test():
        job_id = job_registry.create_job()
        
        with patch("qwen.inference.Qwen3Inference.load") as mock_load, \
             patch("qwen.inference.Qwen3Inference.unload") as mock_unload, \
             patch("job_manager.run_mc1_pipeline") as mock_mc1:
             
            mock_mc1.return_value = {
                "task_executable": True,
                "image_1": {"filename": "img1.tif", "modality": "optical"}
            }
            
            # Intentionally cause an error to verify finally block unloads
            with patch("job_manager.QueryIntelligencePipeline.process_query", side_effect=Exception("Trigger failure")):
                await execute_agentic_pipeline(job_id, [("test.tif", open(os.path.join(os.path.dirname(__file__), "fixtures", "test_geo1.tif"), "rb").read())], "What is here?", execution_mode="real", qwen_backend="transformers", planner_fallback="strict")
                
                assert mock_load.called, "Model should be loaded"
                assert mock_unload.called, "Model MUST be unloaded in the finally block"
                
    asyncio.run(run_test())
