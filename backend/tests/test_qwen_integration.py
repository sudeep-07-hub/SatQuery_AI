import pytest
from unittest.mock import MagicMock
from qwen.pipeline import QueryIntelligencePipeline
from qwen.schemas import TaskSpec, SubtaskSpec, DecompositionResult, ObservationRequirement
import json

def _mock_inference(extraction_json, decomposition_json):
    mock = MagicMock()
    
    # Simple stateful mock to return extraction then decomposition
    responses = [
        {"status": "ok", "response": json.dumps(extraction_json)},
        {"status": "ok", "response": json.dumps(decomposition_json)}
    ]
    
    def side_effect(*args, **kwargs):
        if responses:
            return responses.pop(0)
        return {"status": "error"}
        
    mock.generate.side_effect = side_effect
    return mock

def test_integration_single_image_vqa():
    ext_json = {
        "query": "What is visible?",
        "primary_task": "single_image_vqa",
        "target_entities": [],
        "required_modalities": ["unspecified"],
        "temporal_requirement": "none",
        "spatial_output_required": False,
        "textual_output_required": True,
        "ambiguous": False
    }
    dec_json = {
        "is_compound": False,
        "ambiguous": False,
        "subtasks": [
            {
                "subtask_id": "st1",
                "description": "Answer VQA",
                "primary_task": "single_image_vqa",
                "target_entities": [],
                "required_modalities": ["unspecified"],
                "temporal_requirement": "none",
                "spatial_output_required": False,
                "textual_output_required": True,
                "depends_on": []
            }
        ]
    }
    
    pipeline = QueryIntelligencePipeline(_mock_inference(ext_json, dec_json))
    result = pipeline.process_query("What is visible?")
    
    assert result.primary_task_spec.primary_task == "single_image_vqa"
    assert result.is_compound is False
    assert len(result.subtasks) == 1
    assert result.subtasks[0].primary_task == "single_image_vqa"
    assert len(result.observation_requirements) == 1
    assert result.observation_requirements[0].minimum_observations == 1

def test_integration_compound_change_grounding():
    ext_json = {
        "query": "Find new buildings and locate them.",
        "primary_task": "change_detection",
        "target_entities": ["new buildings"],
        "required_modalities": ["unspecified"],
        "temporal_requirement": "before_after",
        "spatial_output_required": False,
        "textual_output_required": True,
        "ambiguous": False
    }
    dec_json = {
        "is_compound": True,
        "ambiguous": False,
        "subtasks": [
            {
                "subtask_id": "st1",
                "description": "Find changes",
                "primary_task": "change_detection",
                "target_entities": ["new buildings"],
                "required_modalities": ["unspecified"],
                "temporal_requirement": "before_after",
                "spatial_output_required": False,
                "textual_output_required": True,
                "depends_on": []
            },
            {
                "subtask_id": "st2",
                "description": "Locate them",
                "primary_task": "grounding",
                "target_entities": ["new buildings"],
                "required_modalities": ["unspecified"],
                "temporal_requirement": "before_after",
                "spatial_output_required": True,
                "textual_output_required": False,
                "depends_on": ["st1"]
            }
        ]
    }
    
    pipeline = QueryIntelligencePipeline(_mock_inference(ext_json, dec_json))
    result = pipeline.process_query("Find new buildings and locate them.")
    
    assert result.is_compound is True
    assert len(result.subtasks) == 2
    assert result.subtasks[1].depends_on == ["st1"]
    
    # Requirement mapping should merge them into 1 unified 2-image requirement
    assert len(result.observation_requirements) == 1
    assert result.observation_requirements[0].minimum_observations == 2
    assert result.observation_requirements[0].temporal_relationship == "before_after"
    assert "st1" in result.observation_requirements[0].source_subtasks
    assert "st2" in result.observation_requirements[0].source_subtasks

def test_integration_cross_modal():
    ext_json = {
        "query": "Compare optical and SAR.",
        "primary_task": "cross_modal_fusion",
        "target_entities": [],
        "required_modalities": ["optical", "sar"],
        "temporal_requirement": "none",
        "spatial_output_required": False,
        "textual_output_required": True,
        "ambiguous": False
    }
    dec_json = {
        "is_compound": False,
        "ambiguous": False,
        "subtasks": [
            {
                "subtask_id": "st1",
                "description": "Compare optical and SAR",
                "primary_task": "cross_modal_fusion",
                "target_entities": [],
                "required_modalities": ["optical", "sar"],
                "temporal_requirement": "none",
                "spatial_output_required": False,
                "textual_output_required": True,
                "depends_on": []
            }
        ]
    }
    
    pipeline = QueryIntelligencePipeline(_mock_inference(ext_json, dec_json))
    result = pipeline.process_query("Compare optical and SAR.")
    
    assert result.primary_task_spec.primary_task == "cross_modal_fusion"
    assert len(result.observation_requirements) == 1
    assert "optical" in result.observation_requirements[0].required_modalities
    assert "sar" in result.observation_requirements[0].required_modalities
    assert result.observation_requirements[0].minimum_observations == 2
    assert result.observation_requirements[0].co_registration_required is True

def test_serialization():
    ext_json = {
        "query": "Where are buildings?",
        "primary_task": "grounding",
        "target_entities": ["buildings"],
        "required_modalities": ["unspecified"],
        "temporal_requirement": "none",
        "spatial_output_required": True,
        "textual_output_required": False,
        "ambiguous": False
    }
    dec_json = {
        "is_compound": False,
        "ambiguous": False,
        "subtasks": [
            {
                "subtask_id": "st1",
                "description": "Locate buildings",
                "primary_task": "grounding",
                "target_entities": ["buildings"],
                "required_modalities": ["unspecified"],
                "temporal_requirement": "none",
                "spatial_output_required": True,
                "textual_output_required": False,
                "depends_on": []
            }
        ]
    }
    pipeline = QueryIntelligencePipeline(_mock_inference(ext_json, dec_json))
    result = pipeline.process_query("Where are buildings?")
    
    j = result.model_dump_json()
    assert "grounding" in j
    assert "st1" in j
    assert "minimum_observations" in j
    
    # We don't have a direct parser for QueryIntelligenceResult since it's just the top-level aggregator,
    # but let's verify it can be loaded back properly
    import json
    loaded = json.loads(j)
    assert loaded["original_query"] == "Where are buildings?"
