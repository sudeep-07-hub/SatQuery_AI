import os
import pytest
import asyncio
from unittest.mock import patch, MagicMock

from job_manager import execute_agentic_pipeline, job_registry
from qwen.inference import Qwen3Inference


def test_qwen3_real_mode_instantiates_inference():
    """
    C1, C2, C3, C6: execution_mode="real" instantiates Qwen3Inference and does not use MockQwenEngine.
    """
    async def run_test():
        job_id = job_registry.create_job()
        
        with patch("qwen.inference.Qwen3Inference.load") as mock_load, \
             patch("job_manager.QueryIntelligencePipeline.__init__", return_value=None) as mock_qi_init, \
             patch("job_manager.QueryIntelligencePipeline.process_query") as mock_qi_process, \
             patch("job_manager.QwenToolSelector.__init__", return_value=None) as mock_selector_init, \
             patch("job_manager.QwenToolSelector.select_tools") as mock_selector_process, \
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
                    
                    # Verify Qwen3Inference was instantiated and loaded
                    assert mock_load.called, "Qwen3Inference.load() should be called"
                    
                    # Verify pipelines received the real engine
                    assert mock_qi_init.called
                    qi_engine_arg = mock_qi_init.call_args[0][0]
                    assert qi_engine_arg.__class__.__name__ == "Qwen3Inference", "QI Pipeline should receive Qwen3Inference"
                    
                    assert mock_selector_init.called
                    selector_engine_arg = mock_selector_init.call_args[0][0]
                    assert selector_engine_arg.__class__.__name__ == "Qwen3Inference", "Tool Selector should receive Qwen3Inference"
                    
                    job = job_registry.get_job(job_id)
                    assert job["status"] in ["DONE", "INSUFFICIENT_EVIDENCE"], f"Job failed: {job.get('result')}"

    asyncio.run(run_test())

def test_qwen3_real_mode_unavailable_safe_failure():
    """
    C7: If Qwen3 becomes unavailable (e.g. load fails), no silent mock fallback occurs.
    The system gracefully sets status to MODEL_UNAVAILABLE.
    """
    async def run_test():
        job_id = job_registry.create_job()
        
        with patch("qwen.inference.Qwen3Inference.load", side_effect=Exception("CUDA out of memory")), \
             patch("job_manager.run_mc1_pipeline") as mock_mc1:
             
            mock_mc1.return_value = {
                "task_executable": True,
                "image_1": {"filename": "img1.tif", "modality": "optical"}
            }
            
            await execute_agentic_pipeline(job_id, [("test.tif", open(os.path.join(os.path.dirname(__file__), "fixtures", "test_geo1.tif"), "rb").read())], "What is here?", execution_mode="real", qwen_backend="transformers", planner_fallback="strict")
            
            job = job_registry.get_job(job_id)
            # Should gracefully abort with MODEL_UNAVAILABLE
            assert job["result"]["execution_status"] == "MODEL_UNAVAILABLE"
            assert any("Qwen3-4B-Instruct-2507" in m for m in job["result"]["model_unavailable"])
            assert "CUDA out of memory" in job["result"]["final_answer"]
            
    asyncio.run(run_test())
