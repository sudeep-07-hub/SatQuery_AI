import pytest
from unittest.mock import MagicMock

# MC1
from mc1.schemas import RequestObservationProfile, ObservationProfile, SpatialProfile, SensorProfile, QualityProfile

# MC3
from qwen.schemas import TaskSpec, ObservationRequirement
from qwen.pipeline import QueryIntelligenceResult
from agent.default_tools import setup_default_registry
from agent.selector import QwenToolSelector
from agent.binding import ObservationBinder

# Temporal (MC4B)
from mc4b_temporal.pipeline import TemporalPipeline
from mc4b_temporal.result import TemporalResult, RawTemporalOutput
from mc4b_temporal.tool_adapter import ChangeMambaAdapter

# Normalizer (MC5)
from mc4b_temporal.evidence_normalizer import normalize_to_evidence

# Interpreter (Qwen3 Layer)
from mc4b_temporal.semantic_interpreter import TemporalSemanticInterpreter, TemporalEvidenceContext, TemporalAnswer

# Verifier (MC6)
from mc6_verification.temporal_verifier import TemporalClaimVerifier


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


def build_deterministic_qi_result() -> QueryIntelligenceResult:
    return QueryIntelligenceResult(
        original_query="Compare the two images and describe what changed between them.",
        primary_task_spec=TaskSpec(
            query="Compare the two images and describe what changed between them.",
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

def test_a_b_query_routing_and_binding():
    # 1. Routing to temporal_change_analysis
    registry = setup_default_registry()
    qi = build_deterministic_qi_result()
    
    # Enable tools for test
    registry.get("single_image_vqa").enabled = True
    registry.get("temporal_change_analysis").enabled = True
    
    selector = QwenToolSelector(DeterministicMockQwenEngine([
        '{"tool_id": "temporal_change_analysis", "arguments": {"query": "Compare the two images."}}'
    ]), registry)
    
    call = selector.select_tool(qi)
    assert call.tool_id == "temporal_change_analysis"
    
    # 2. Binding and Temporal Ordering
    req_profile = RequestObservationProfile(
        observations=[
            ObservationProfile(
                observation_id="img_late",
                file_source="late.tif",
                file_format="tiff",
                spatial=SpatialProfile(crs="EPSG:4326", gsd_m=1.0),
                sensor=SensorProfile(modality="optical", sensor="S2"),
                quality=QualityProfile(score=1.0),
                temporal={"timestamp": "2024-01-01T00:00:00Z"}
            ),
            ObservationProfile(
                observation_id="img_early",
                file_source="early.tif",
                file_format="tiff",
                spatial=SpatialProfile(crs="EPSG:4326", gsd_m=1.0),
                sensor=SensorProfile(modality="optical", sensor="S2"),
                quality=QualityProfile(score=1.0),
                temporal={"timestamp": "2020-01-01T00:00:00Z"}
            )
        ]
    )
    
    binder = ObservationBinder()
    binding = binder.bind(call, qi.observation_requirements[0], req_profile)
    
    # It should chronologically sort them: early is t1, late is t2
    assert binding.bound_call.input_bindings["before"].observation_id == "img_early"
    assert binding.bound_call.input_bindings["after"].observation_id == "img_late"


def test_c_changemamba_unavailable_path():
    # Force TemporalPipeline to return MODEL_UNAVAILABLE
    pipeline = TemporalPipeline()
    mc1_profile = {
        "image_1": {"filename": "early.tif", "acquisition_date": "2020-01-01"},
        "image_2": {"filename": "late.tif", "acquisition_date": "2024-01-01"}
    }
    
    # Because mamba-ssm is missing on MPS, pipeline execute will natively yield MODEL_UNAVAILABLE
    # But just in case test environment changes, we mock it explicitly for this test logic
    pipeline.execute = MagicMock(return_value=TemporalResult(status="MODEL_UNAVAILABLE", before_observation_id="img_early", after_observation_id="img_late"))
    
    result = pipeline.execute(mc1_profile=mc1_profile, t1_tensor=None, t2_tensor=None)
    assert result.status == "MODEL_UNAVAILABLE"
    
    # Normalization (MC5)
    evidence = normalize_to_evidence(result, "job_123")
    assert evidence == []
    
    # Qwen3 Interpretation
    interpreter = TemporalSemanticInterpreter(inference_engine=MagicMock())
    context = TemporalEvidenceContext(
        query="What changed?",
        task_type="change_description",
        before_observation="img_early",
        after_observation="img_late",
        evidence_objects=evidence,
        pipeline_status=result.status
    )
    answer = interpreter.generate_answer(context)
    assert answer.status == "TEMPORAL_EVIDENCE_UNAVAILABLE"
    assert "unavailable" in answer.answer.lower()
    
    # MC6 Verification
    verifier = TemporalClaimVerifier()
    trace = verifier.verify(
        temporal_answer=answer.model_dump() if hasattr(answer, "model_dump") else answer.dict(),
        evidence_objects=evidence,
        pipeline_status=result.status
    )
    assert trace.verification_status == "TEMPORAL_EVIDENCE_UNAVAILABLE"


def test_d_e_f_g_successful_fixture_path():
    # 1. Pipeline returns SUCCESS with a CONTROLLED TEMPORAL FIXTURE
    raw = RawTemporalOutput(
        binary_mask=[[0, 1], [0, 0]], # one changed pixel
        output_shape=[2, 2],
        dtype="int",
        device="cpu",
        confidence=0.88
    )
    result = TemporalResult(
        status="SUCCESS",
        before_observation_id="img_early",
        after_observation_id="img_late",
        crs="EPSG:4326",
        affine_transform=[1.0, 0.0, 100.0, 0.0, 1.0, 200.0],
        raw_output=raw,
        model_id="ChangeMamba"
    )
    
    # 2. Normalization (MC5)
    evidence = normalize_to_evidence(result, "job_123")
    assert len(evidence) == 1
    e = evidence[0]
    assert e["evidence_type"] == "bitemporal_change_detection"
    assert e["timestamp"]["t1"] == "img_early"
    assert e["timestamp"]["t2"] == "img_late"
    assert e["spatial_region"]["type"] == "Polygon" # confirms pixel_to_geo ran
    
    # 3. Qwen3 Semantic Interpretation
    mock_qwen = DeterministicMockQwenEngine([
        "There is a Model-predicted change region detected."
    ])
    interpreter = TemporalSemanticInterpreter(inference_engine=mock_qwen)
    context = TemporalEvidenceContext(
        query="What changed?",
        task_type="change_description",
        before_observation="img_early",
        after_observation="img_late",
        evidence_objects=evidence,
        pipeline_status=result.status
    )
    answer = interpreter.generate_answer(context)
    assert answer.status == "ANSWERED"
    assert answer.evidence_ids == [e["evidence_id"]]
    
    # 4. MC6 Verification
    verifier = TemporalClaimVerifier()
    trace = verifier.verify(
        temporal_answer=answer.model_dump() if hasattr(answer, "model_dump") else answer.dict(),
        evidence_objects=evidence,
        pipeline_status=result.status
    )
    assert trace.verification_status == "VERIFIED"
    assert "all_checks_passed" in trace.checks
    

def test_j_no_fallback_regression():
    # Ensure tool config doesn't fall back to CNN
    from mc4b_temporal.config import BACKBONE
    assert BACKBONE == "changemamba"
