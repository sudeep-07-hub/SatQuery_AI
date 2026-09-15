import pytest
from unittest.mock import MagicMock
from mc4b_temporal.semantic_interpreter import TemporalSemanticInterpreter, TemporalEvidenceContext

@pytest.fixture
def mock_qwen():
    mock = MagicMock()
    mock.generate.return_value = {"status": "ok", "response": "Mocked Qwen response."}
    return mock

def test_semantic_model_unavailable(mock_qwen):
    interpreter = TemporalSemanticInterpreter(inference_engine=mock_qwen)
    context = TemporalEvidenceContext(
        query="What changed?",
        task_type="change_vqa",
        before_observation="obs_1",
        after_observation="obs_2",
        evidence_objects=[],
        pipeline_status="MODEL_UNAVAILABLE"
    )
    
    answer = interpreter.generate_answer(context)
    assert answer.status == "TEMPORAL_EVIDENCE_UNAVAILABLE"
    assert "currently unavailable" in answer.answer
    mock_qwen.generate.assert_not_called()

def test_semantic_no_change(mock_qwen):
    interpreter = TemporalSemanticInterpreter(inference_engine=mock_qwen)
    context = TemporalEvidenceContext(
        query="What changed?",
        task_type="change_vqa",
        before_observation="obs_1",
        after_observation="obs_2",
        evidence_objects=[],
        pipeline_status="SUCCESS"
    )
    
    answer = interpreter.generate_answer(context)
    assert answer.status == "ANSWERED"
    assert "No change detected" in answer.answer
    mock_qwen.generate.assert_not_called()

def test_semantic_insufficient_evidence(mock_qwen):
    interpreter = TemporalSemanticInterpreter(inference_engine=mock_qwen)
    context = TemporalEvidenceContext(
        query="Did the building change?",
        task_type="change_vqa",
        before_observation="obs_1",
        after_observation="obs_2",
        evidence_objects=[{"evidence_id": "e1", "claim": "Spatial change detected."}],
        pipeline_status="SUCCESS"
    )
    
    answer = interpreter.generate_answer(context)
    assert answer.status == "INSUFFICIENT_EVIDENCE"
    assert "does not establish specific semantic land-cover types like buildings" in answer.answer
    mock_qwen.generate.assert_not_called()

def test_semantic_grounded_answer(mock_qwen):
    interpreter = TemporalSemanticInterpreter(inference_engine=mock_qwen)
    context = TemporalEvidenceContext(
        query="Is there change?",
        task_type="change_vqa",
        before_observation="obs_1",
        after_observation="obs_2",
        evidence_objects=[{"evidence_id": "e1", "claim": "Spatial change detected."}],
        pipeline_status="SUCCESS"
    )
    
    answer = interpreter.generate_answer(context)
    assert answer.status == "ANSWERED"
    assert answer.answer == "Mocked Qwen response."
    assert "e1" in answer.evidence_ids
    mock_qwen.generate.assert_called_once()
    
    prompt = mock_qwen.generate.call_args[0][0]
    assert "Evidence ID: e1" in prompt
    assert "Do not claim visual observations that are not represented in the evidence." in prompt
