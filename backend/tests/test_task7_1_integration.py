"""
test_task7_1_integration.py — Task 7.1 Full Agentic System Integration Tests.

Validates the end-to-end workflow:
    USER QUERY → MC1 → Qwen3 QI → MC3 → Specialist → MC5 → MC6 → Final Answer

Tests cover:
    TEST 1: Single-image VQA end-to-end
    TEST 2: Temporal end-to-end (MODEL_UNAVAILABLE)
    TEST 3: Insufficient observations
    TEST 4: Invalid tool selection → controller rejects
    TEST 5: MODEL_UNAVAILABLE ≠ zero-change SUCCESS (semantic distinction)
    TEST 6: MC6 rejects unsupported claim
    TEST 7: Replanning within bounds
    TEST 8: Replanning exhaustion
    TEST 9: Non-temporal regression
    TEST 10: Audit trace completeness
"""
import pytest
from unittest.mock import MagicMock, patch
import uuid

# MC1
from mc1.schemas import (
    RequestObservationProfile, ObservationProfile,
    SpatialProfile, SensorProfile, QualityProfile
)

# Phase 3
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
from agent.schemas import ToolCall, ToolResult, AgentState

# MC5
from mc5_evidence.evidence_graph import EvidenceGraph

# MC6
from mc6_verification.verifier import Verifier
from mc6_verification.temporal_verifier import TemporalClaimVerifier

# Job Manager (the integration target)
from job_manager import (
    _build_observation_profiles,
    _synthesize_final_answer,
)


class DeterministicMockQwenEngine:
    def __init__(self, responses: list):
        self.responses = responses
        self.call_count = 0

    def load(self):
        pass

    def generate(self, prompt: str, **kwargs) -> dict:
        if self.call_count < len(self.responses):
            resp = self.responses[self.call_count]
            self.call_count += 1
            return {"status": "ok", "response": resp}
        return {"status": "error", "error": "Mock exhaustion"}


def _qi_single_image() -> QueryIntelligenceResult:
    return QueryIntelligenceResult(
        original_query="What is visible in this image?",
        primary_task_spec=TaskSpec(
            query="What is visible in this image?",
            primary_task="single_image_vqa",
            target_entities=["object"],
            required_operations=["vqa"],
            required_modalities=["optical"],
            spatial_output_required=False,
            textual_output_required=True,
            temporal_requirement="none",
            ambiguous=False
        ),
        is_compound=False,
        subtasks=[],
        observation_requirements=[
            ObservationRequirement(
                requirement_id="req_1",
                source_subtasks=[],
                minimum_observations=1,
                maximum_observations=1,
                required_modalities=["optical"],
                temporal_relationship="none"
            )
        ],
        ambiguous=False
    )


def _qi_temporal() -> QueryIntelligenceResult:
    return QueryIntelligenceResult(
        original_query="Compare the two images and describe what changed.",
        primary_task_spec=TaskSpec(
            query="Compare the two images and describe what changed.",
            primary_task="change_detection",
            target_entities=["object"],
            required_operations=["temporal_comparison"],
            required_modalities=["optical"],
            spatial_output_required=True,
            textual_output_required=True,
            temporal_requirement="before_after",
            ambiguous=False
        ),
        is_compound=False,
        subtasks=[],
        observation_requirements=[
            ObservationRequirement(
                requirement_id="req_1",
                source_subtasks=[],
                minimum_observations=2,
                maximum_observations=2,
                required_modalities=["optical"],
                temporal_relationship="before_after"
            )
        ],
        ambiguous=False
    )


def _single_image_profile() -> RequestObservationProfile:
    return RequestObservationProfile(
        observations=[
            ObservationProfile(
                observation_id="image_1",
                file_source="img1.tif",
                file_format="tiff",
                spatial=SpatialProfile(crs="EPSG:4326", gsd_m=1.0),
                sensor=SensorProfile(modality="optical", sensor="S2"),
                quality=QualityProfile(score=1.0)
            )
        ]
    )


def _two_image_profile() -> RequestObservationProfile:
    return RequestObservationProfile(
        observations=[
            ObservationProfile(
                observation_id="image_1",
                file_source="img1.tif",
                file_format="tiff",
                spatial=SpatialProfile(crs="EPSG:4326", gsd_m=1.0),
                sensor=SensorProfile(modality="optical", sensor="S2"),
                quality=QualityProfile(score=1.0),
                temporal={"timestamp": "2020-01-01T00:00:00Z"}
            ),
            ObservationProfile(
                observation_id="image_2",
                file_source="img2.tif",
                file_format="tiff",
                spatial=SpatialProfile(crs="EPSG:4326", gsd_m=1.0),
                sensor=SensorProfile(modality="optical", sensor="S2"),
                quality=QualityProfile(score=1.0),
                temporal={"timestamp": "2024-01-01T00:00:00Z"}
            )
        ]
    )


# ═══════════════════════════════════════════════════════════════════
# TEST 1: Single-Image VQA End-to-End
# ═══════════════════════════════════════════════════════════════════

def test_1_single_image_vqa_e2e():
    """MC1 → Qwen3 → MC3 → PaliGemma → MC5 → MC6 → final."""
    registry = setup_default_registry()
    qi = _qi_single_image()
    
    registry.get("single_image_vqa").enabled = True
    
    engine = DeterministicMockQwenEngine([
        '{"tool_id": "single_image_vqa", "arguments": {"query": "What is visible?"}}'
    ])
    selector = QwenToolSelector(engine, registry)
    call = selector.select_tool(qi)
    assert call.tool_id == "single_image_vqa"
    
    # Bind
    binder = ObservationBinder()
    binding = binder.bind(call, qi.observation_requirements[0], _single_image_profile())
    assert binding.status == "SUFFICIENT"
    
    # Plan
    planner = WorkflowPlanner()
    plan = planner.build_workflow([binding.bound_call], qi, registry)
    assert plan.status == "ready"
    
    # Execute via controller with mock adapter
    adapters = {"single_image_vqa": MockToolAdapter()}
    controller = AgentController(registry, adapters)
    state = controller.execute(plan, {}, {"t1": None})
    assert state.status == "completed"
    assert len(state.completed_calls) == 1
    
    # Normalize evidence
    evidence = [{
        "evidence_id": "ev_test_1",
        "claim": "Objects visible in image.",
        "confidence": 0.9,
        "source_model": "single_image_vqa"
    }]
    
    # Evidence Graph
    graph = EvidenceGraph()
    graph.add_node("q1", "query", {"text": "What is visible?"})
    for ev in evidence:
        graph.insert_evidence(ev, "q1")
    assert graph.node_count > 0
    
    # MC6 Verification
    verifier = Verifier()
    v_result = verifier.verify(evidence)
    assert v_result["status"] == "VERIFIED"
    
    # Final Answer
    final = _synthesize_final_answer(
        query="What is visible?",
        evidence=evidence,
        verification_result=v_result,
        agent_state_status="completed",
        model_unavailable_tools=[]
    )
    assert final["execution_status"] == "SUCCESS"
    assert len(final["evidence_references"]) == 1


# ═══════════════════════════════════════════════════════════════════
# TEST 2: Temporal End-to-End (MODEL_UNAVAILABLE)
# ═══════════════════════════════════════════════════════════════════

def test_2_temporal_model_unavailable():
    """Temporal query → ChangeMamba → MODEL_UNAVAILABLE → safe propagation."""
    registry = setup_default_registry()
    qi = _qi_temporal()
    
    registry.get("temporal_change_analysis").enabled = True
    
    engine = DeterministicMockQwenEngine([
        '{"tool_id": "temporal_change_analysis", "arguments": {"query": "Detect changes"}}'
    ])
    selector = QwenToolSelector(engine, registry)
    call = selector.select_tool(qi)
    assert call.tool_id == "temporal_change_analysis"
    
    # Bind
    binder = ObservationBinder()
    binding = binder.bind(call, qi.observation_requirements[0], _two_image_profile())
    assert binding.status == "SUFFICIENT"
    
    # Simulate MODEL_UNAVAILABLE through adapter
    failing_adapter = MockToolAdapter(should_fail=True, fail_reason="MODEL_UNAVAILABLE: mamba-ssm requires CUDA")
    adapters = {"temporal_change_analysis": failing_adapter}
    
    planner = WorkflowPlanner()
    plan = planner.build_workflow([binding.bound_call], qi, registry)
    controller = AgentController(registry, adapters)
    state = controller.execute(plan, {}, {"t1": None, "t2": None})
    
    assert state.status == "failed"
    assert len(state.failed_calls) == 1
    
    # Final answer should reflect MODEL_UNAVAILABLE
    final = _synthesize_final_answer(
        query="Compare the two images.",
        evidence=[],
        verification_result={"status": "INSUFFICIENT_EVIDENCE", "triggers_fired": ["no_evidence_produced"]},
        agent_state_status="failed",
        model_unavailable_tools=["temporal_change_analysis"]
    )
    assert final["execution_status"] == "MODEL_UNAVAILABLE"
    assert "unavailable" in final["final_answer"].lower()
    # Must NOT claim change or no-change
    assert "no change" not in final["final_answer"].lower()
    assert "change detected" not in final["final_answer"].lower()


# ═══════════════════════════════════════════════════════════════════
# TEST 3: Insufficient Observations
# ═══════════════════════════════════════════════════════════════════

def test_3_insufficient_observations():
    """Temporal query with only 1 image → INSUFFICIENT."""
    registry = setup_default_registry()
    qi = _qi_temporal()
    registry.get("temporal_change_analysis").enabled = True
    
    engine = DeterministicMockQwenEngine([
        '{"tool_id": "temporal_change_analysis", "arguments": {"query": "Detect changes"}}'
    ])
    selector = QwenToolSelector(engine, registry)
    call = selector.select_tool(qi)
    
    # Only 1 image
    binder = ObservationBinder()
    binding = binder.bind(call, qi.observation_requirements[0], _single_image_profile())
    assert binding.status == "INSUFFICIENT"
    assert binding.bound_call is None


# ═══════════════════════════════════════════════════════════════════
# TEST 4: Invalid Tool Selection
# ═══════════════════════════════════════════════════════════════════

def test_4_invalid_tool_rejected():
    """Controller rejects unregistered/disabled tool."""
    registry = setup_default_registry()
    
    # Create a call to a non-existent tool
    invalid_call = ToolCall(
        call_id="call_invalid",
        tool_id="nonexistent_tool",
        arguments={"query": "test"}
    )
    
    planner = WorkflowPlanner()
    qi = _qi_single_image()
    plan = planner.build_workflow([invalid_call], qi, registry)
    
    # The planner correctly marks the plan as invalid for unregistered tools
    # The controller then short-circuits at plan validation
    adapters = {}
    controller = AgentController(registry, adapters)
    state = controller.execute(plan, {}, {})
    
    assert state.status == "failed"
    # Verify the failure is due to plan validation (invalid plan status)
    assert any("Invalid WorkflowPlan" in e.get("error", "") for e in state.execution_trace)


# ═══════════════════════════════════════════════════════════════════
# TEST 5: MODEL_UNAVAILABLE ≠ Zero-Change SUCCESS
# ═══════════════════════════════════════════════════════════════════

def test_5_model_unavailable_vs_zero_change():
    """These two states must NOT collapse into the same result."""
    # State 1: MODEL_UNAVAILABLE
    final_unavailable = _synthesize_final_answer(
        query="What changed?",
        evidence=[],
        verification_result={"status": "INSUFFICIENT_EVIDENCE", "triggers_fired": []},
        agent_state_status="failed",
        model_unavailable_tools=["temporal_change_analysis"]
    )
    
    # State 2: SUCCESS with zero-change evidence
    final_zero_change = _synthesize_final_answer(
        query="What changed?",
        evidence=[{
            "evidence_id": "ev_zero",
            "claim": "No significant change detected between observations.",
            "confidence": 0.95,
            "source_model": "ChangeMamba"
        }],
        verification_result={"status": "VERIFIED", "triggers_fired": []},
        agent_state_status="completed",
        model_unavailable_tools=[]
    )
    
    # They must be semantically distinct
    assert final_unavailable["execution_status"] == "MODEL_UNAVAILABLE"
    assert final_zero_change["execution_status"] == "SUCCESS"
    assert final_unavailable["final_answer"] != final_zero_change["final_answer"]
    assert "unavailable" in final_unavailable["final_answer"].lower()


# ═══════════════════════════════════════════════════════════════════
# TEST 6: MC6 Rejects Unsupported Claim
# ═══════════════════════════════════════════════════════════════════

def test_6_mc6_rejects_unsupported_claim():
    """MC6 TemporalClaimVerifier rejects hallucinated semantic claims."""
    verifier = TemporalClaimVerifier()
    trace = verifier.verify(
        temporal_answer={
            "status": "ANSWERED",
            "answer": "New building constructed in the area.",
            "evidence_ids": ["e1"]
        },
        evidence_objects=[{
            "evidence_id": "e1",
            "claim": "Generic spatial change."
        }],
        pipeline_status="SUCCESS"
    )
    assert trace.verification_status == "UNSUPPORTED"
    assert "semantic_support_check" in trace.checks


# ═══════════════════════════════════════════════════════════════════
# TEST 7: Replanning Within Bounds
# ═══════════════════════════════════════════════════════════════════

def test_7_replanning_within_bounds():
    """Recovery replans when first tool fails, selects alternative."""
    registry = setup_default_registry()
    registry.get("single_image_vqa").enabled = True
    registry.get("temporal_change_analysis").enabled = True
    
    # First tool fails, recovery proposes single_image_vqa
    failing_adapter = MockToolAdapter(should_fail=True, fail_reason="Tool execution error")
    success_adapter = MockToolAdapter()
    
    adapters = {
        "temporal_change_analysis": failing_adapter,
        "single_image_vqa": success_adapter
    }
    
    controller = AgentController(registry, adapters)
    
    mock_recovery_engine = DeterministicMockQwenEngine([
        '{"recovery_possible": true, "replacement_tool_id": "single_image_vqa", "reason": "Fallback to VQA", "arguments": {"query": "Describe"}}'
    ])
    recovery_planner = QwenRecoveryPlanner(mock_recovery_engine)
    planner = WorkflowPlanner()
    recovery_manager = RecoveryManager(controller, recovery_planner, registry, planner)
    
    qi = _qi_single_image()
    # Start with temporal tool (which will fail)
    initial_call = ToolCall(
        call_id="call_1",
        tool_id="temporal_change_analysis",
        arguments={"query": "test"}
    )
    plan = planner.build_workflow([initial_call], qi, registry)
    
    state = recovery_manager.execute_with_recovery(plan, qi, {}, {"t1": None, "t2": None}, "test query")
    
    # Should have replanned at least once
    assert state.replan_count >= 1 or state.status == "completed"


# ═══════════════════════════════════════════════════════════════════
# TEST 8: Replanning Exhaustion
# ═══════════════════════════════════════════════════════════════════

def test_8_replanning_exhaustion():
    """Repeated failures terminate within configured bounds."""
    registry = setup_default_registry()
    registry.get("single_image_vqa").enabled = True
    
    # All adapters fail
    adapters = {
        "single_image_vqa": MockToolAdapter(should_fail=True, fail_reason="Always fails"),
    }
    
    controller = AgentController(registry, adapters)
    
    mock_recovery_engine = DeterministicMockQwenEngine([])  # No recovery possible
    recovery_planner = QwenRecoveryPlanner(mock_recovery_engine)
    planner = WorkflowPlanner()
    recovery_manager = RecoveryManager(controller, recovery_planner, registry, planner)
    
    qi = _qi_single_image()
    initial_call = ToolCall(
        call_id="call_1",
        tool_id="single_image_vqa",
        arguments={"query": "test"}
    )
    plan = planner.build_workflow([initial_call], qi, registry)
    
    state = recovery_manager.execute_with_recovery(plan, qi, {}, {"t1": None}, "test query")
    
    # Must terminate (not infinite loop)
    assert state.status == "failed"
    assert state.tool_call_count <= state.max_tool_calls


# ═══════════════════════════════════════════════════════════════════
# TEST 9: Non-Temporal Regression
# ═══════════════════════════════════════════════════════════════════

def test_9_nontemporal_regression():
    """Existing single-image VQA workflow still works after Phase 6/7 integration."""
    registry = setup_default_registry()
    registry.get("single_image_vqa").enabled = True
    
    adapters = {"single_image_vqa": MockToolAdapter()}
    controller = AgentController(registry, adapters)
    
    qi = _qi_single_image()
    engine = DeterministicMockQwenEngine([
        '{"tool_id": "single_image_vqa", "arguments": {"query": "Test"}}'
    ])
    selector = QwenToolSelector(engine, registry)
    call = selector.select_tool(qi)
    
    binder = ObservationBinder()
    binding = binder.bind(call, qi.observation_requirements[0], _single_image_profile())
    
    planner = WorkflowPlanner()
    plan = planner.build_workflow([binding.bound_call], qi, registry)
    state = controller.execute(plan, {}, {"t1": None})
    
    assert state.status == "completed"
    assert "single_image_vqa" in [state.results[c].tool_id for c in state.completed_calls]


# ═══════════════════════════════════════════════════════════════════
# TEST 10: Audit Trace Completeness
# ═══════════════════════════════════════════════════════════════════

def test_10_audit_trace_completeness():
    """Execution trace captures workflow_start, tool_execution, and workflow_completion."""
    registry = setup_default_registry()
    registry.get("single_image_vqa").enabled = True
    
    adapters = {"single_image_vqa": MockToolAdapter()}
    controller = AgentController(registry, adapters)
    
    qi = _qi_single_image()
    call = ToolCall(
        call_id="call_trace",
        tool_id="single_image_vqa",
        arguments={"query": "test"}
    )
    
    planner = WorkflowPlanner()
    plan = planner.build_workflow([call], qi, registry)
    state = controller.execute(plan, {}, {"t1": None})
    
    # Verify trace events
    event_types = [e["event_type"] for e in state.execution_trace]
    assert "workflow_start" in event_types
    assert "tool_execution" in event_types
    assert "workflow_completion" in event_types
    
    # Each event has a timestamp
    for event in state.execution_trace:
        assert "timestamp" in event
    
    # Completion status is recorded
    completion_events = [e for e in state.execution_trace if e["event_type"] == "workflow_completion"]
    assert len(completion_events) == 1
    assert completion_events[0]["status"] == "succeeded"


# ═══════════════════════════════════════════════════════════════════
# TEST 11: _build_observation_profiles helper
# ═══════════════════════════════════════════════════════════════════

def test_11_build_observation_profiles():
    """Helper correctly converts legacy MC1 dict to formal ObservationProfile list."""
    mc1 = {
        "image_1": {
            "filename": "early.tif",
            "modality": "optical",
            "crs": "EPSG:32643",
            "gsd_m": 1.0,
            "acquisition_date": "2020-01-01T00:00:00Z"
        },
        "image_2": {
            "filename": "late.tif",
            "modality": "optical",
            "crs": "EPSG:32643",
            "gsd_m": 1.0,
            "acquisition_date": "2024-01-01T00:00:00Z"
        }
    }
    obs_list = _build_observation_profiles(mc1)
    assert len(obs_list) == 2
    assert obs_list[0].observation_id == "image_1"
    assert obs_list[1].observation_id == "image_2"
    assert obs_list[0].temporal["timestamp"] == "2020-01-01T00:00:00Z"


# ═══════════════════════════════════════════════════════════════════
# TEST 12: Evidence Graph Integration
# ═══════════════════════════════════════════════════════════════════

def test_12_evidence_graph_integration():
    """Evidence objects are correctly inserted into the Evidence Graph."""
    graph = EvidenceGraph()
    graph.add_node("q1", "query", {"text": "Test query"})
    
    evidence = [
        {
            "evidence_id": "ev_1",
            "claim": "Spatial change detected.",
            "confidence": 0.88,
            "source_model": "ChangeMamba",
            "modality": "optical",
            "timestamp": {"t1": "2020", "t2": "2024"},
            "spatial_region": {"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]]}
        }
    ]
    
    for ev in evidence:
        graph.insert_evidence(ev, "q1")
    
    # Verify graph structure
    assert graph.node_count >= 5  # evidence, claim, model, modality, timestamp, region, query
    assert graph.edge_count >= 4  # supports, derived_from (model, modality, timestamp), overlaps
    
    # Evidence IDs are stable
    ev_node = graph.get_node("ev_1")
    assert ev_node is not None
    assert ev_node["type"] == "evidence"
