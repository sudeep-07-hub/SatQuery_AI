import pytest
import json
from typing import Dict, Any

from agent.schemas import *
from qwen.schemas import TaskSpec, ObservationRequirement
from qwen.pipeline import QueryIntelligenceResult
from agent.default_tools import setup_default_registry
from agent.selector import QwenToolSelector
from agent.binding import ObservationBinder
from agent.planner import WorkflowPlanner
from agent.controller import AgentController
from agent.recovery import RecoveryManager
from agent.recovery_planner import QwenRecoveryPlanner
from agent.adapters import MockToolAdapter

class DeterministicMockQwenEngine:
    def __init__(self, responses: list):
        self.responses = responses
        self.call_count = 0

    def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        if self.call_count < len(self.responses):
            resp = self.responses[self.call_count]
            self.call_count += 1
            return {"status": "ok", "response": resp}
        return {"status": "error", "error": "Mock exhaustion"}

def build_deterministic_qi_result(task="single_image_vqa", modality="optical") -> QueryIntelligenceResult:
    return QueryIntelligenceResult(
        original_query="Test query",
        primary_task_spec=TaskSpec(
            query="Test query",
            primary_task=task,
            target_entities=["object"],
            required_operations=["test"],
            required_modalities=[modality],
            spatial_output_required=False,
            textual_output_required=True,
            temporal_requirement="none" if task != "change_detection" else "bi_temporal_change",
            ambiguous=False
        ),
        is_compound=False,
        subtasks=[],
        observation_requirements=[
            ObservationRequirement(
                requirement_id="req_1",
                source_subtasks=[],
                minimum_observations=1 if task != "change_detection" else 2,
                maximum_observations=1 if task != "change_detection" else 2,
                required_modalities=[modality],
                temporal_relationship="none" if task != "change_detection" else "before_after"
            )
        ],
        ambiguous=False
    )

def test_scenario_1_single_image_vqa():
    registry = setup_default_registry()
    adapters = {"single_image_vqa": MockToolAdapter()}
    registry.get("single_image_vqa").enabled = True
    
    qi = build_deterministic_qi_result("single_image_vqa")
    mc1_profile = {
        "image_count": 1,
        "image_1": {"filename": "img1.tif", "modality": "optical"}
    }
    
    selector = QwenToolSelector(DeterministicMockQwenEngine([
        '{"tool_id": "single_image_vqa", "arguments": {"query": "Test query"}}'
    ]), registry)
    call = selector.select_tool(qi)
    calls = [call]
    assert len(calls) == 1
    
    from mc1.schemas import RequestObservationProfile, ObservationProfile, SensorProfile, SpatialProfile, QualityProfile
    req_profile = RequestObservationProfile(
        observations=[
            ObservationProfile(
                observation_id="image_1", 
                file_source="test.tif",
                file_format="tiff",
                spatial=SpatialProfile(crs="EPSG:32643", gsd_m=1.0),
                sensor=SensorProfile(modality="optical", sensor="test"),
                quality=QualityProfile(score=1.0)
            )
        ]
    )
    
    binder = ObservationBinder()
    binding_result = binder.bind(call, qi.observation_requirements[0], req_profile)
    bound_calls = [binding_result.bound_call]
    assert bound_calls[0].input_bindings["primary"].observation_id == "image_1"
    
    planner = WorkflowPlanner()
    plan = planner.build_workflow(bound_calls, qi, registry)
    
    controller = AgentController(registry, adapters)
    state = controller.execute(plan, mc1_profile, {})
    
    assert state.status == "completed"
    assert "single_image_vqa" in [state.results[c].tool_id for c in state.completed_calls]
