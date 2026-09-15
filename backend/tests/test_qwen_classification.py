import pytest
import json
from unittest.mock import MagicMock
from qwen.extractor import TaskSpecExtractor

def _mock_extractor(response_dict):
    mock_inference = MagicMock()
    mock_inference.generate.return_value = {
        "status": "ok",
        "response": json.dumps(response_dict)
    }
    return TaskSpecExtractor(mock_inference)

def test_single_image_vqa():
    extractor = _mock_extractor({
        "query": "What is visible in this image?",
        "primary_task": "single_image_vqa",
        "target_entities": [],
        "required_modalities": ["unspecified"],
        "temporal_requirement": "none",
        "spatial_output_required": False,
        "textual_output_required": True,
        "ambiguous": False
    })
    spec = extractor.extract("What is visible in this image?")
    assert spec.primary_task == "single_image_vqa"

def test_captioning():
    extractor = _mock_extractor({
        "query": "Describe this satellite image.",
        "primary_task": "captioning",
        "target_entities": [],
        "required_modalities": ["unspecified"],
        "temporal_requirement": "none",
        "spatial_output_required": False,
        "textual_output_required": True,
        "ambiguous": False
    })
    spec = extractor.extract("Describe this satellite image.")
    assert spec.primary_task == "captioning"

def test_grounding():
    extractor = _mock_extractor({
        "query": "Where are the buildings?",
        "primary_task": "grounding",
        "target_entities": ["buildings"],
        "required_modalities": ["unspecified"],
        "temporal_requirement": "none",
        "spatial_output_required": True,
        "textual_output_required": False,
        "ambiguous": False
    })
    spec = extractor.extract("Where are the buildings?")
    assert spec.primary_task == "grounding"
    assert spec.spatial_output_required is True

def test_change_detection():
    extractor = _mock_extractor({
        "query": "What changed between these two images?",
        "primary_task": "change_detection",
        "target_entities": [],
        "required_modalities": ["unspecified"],
        "temporal_requirement": "before_after",
        "spatial_output_required": False,
        "textual_output_required": True,
        "ambiguous": False
    })
    spec = extractor.extract("What changed between these two images?")
    assert spec.primary_task == "change_detection"
    assert spec.temporal_requirement == "before_after"

def test_change_vqa():
    extractor = _mock_extractor({
        "query": "Did the built-up area increase?",
        "primary_task": "change_vqa",
        "target_entities": ["built-up area"],
        "required_modalities": ["unspecified"],
        "temporal_requirement": "before_after",
        "spatial_output_required": False,
        "textual_output_required": True,
        "ambiguous": False
    })
    spec = extractor.extract("Did the built-up area increase?")
    assert spec.primary_task == "change_vqa"
    assert spec.temporal_requirement == "before_after"

def test_cross_modal_fusion():
    extractor = _mock_extractor({
        "query": "Compare the optical and SAR imagery.",
        "primary_task": "cross_modal_fusion",
        "target_entities": [],
        "required_modalities": ["optical", "sar"],
        "temporal_requirement": "none",
        "spatial_output_required": False,
        "textual_output_required": True,
        "ambiguous": False
    })
    spec = extractor.extract("Compare the optical and SAR imagery.")
    assert spec.primary_task == "cross_modal_fusion"
    assert "optical" in spec.required_modalities
    assert "sar" in spec.required_modalities

def test_unknown_ambiguous():
    extractor = _mock_extractor({
        "query": "Analyze this.",
        "primary_task": "unknown",
        "target_entities": [],
        "required_modalities": ["unspecified"],
        "temporal_requirement": "unknown",
        "spatial_output_required": False,
        "textual_output_required": True,
        "ambiguous": True
    })
    spec = extractor.extract("Analyze this.")
    assert spec.primary_task == "unknown"
    assert spec.ambiguous is True

def test_sar_vqa():
    extractor = _mock_extractor({
        "query": "What is visible in the SAR image?",
        "primary_task": "single_image_vqa",
        "target_entities": [],
        "required_modalities": ["sar"],
        "temporal_requirement": "none",
        "spatial_output_required": False,
        "textual_output_required": True,
        "ambiguous": False
    })
    spec = extractor.extract("What is visible in the SAR image?")
    assert spec.primary_task == "single_image_vqa"
    assert spec.required_modalities == ["sar"]

def test_sar_captioning():
    extractor = _mock_extractor({
        "query": "Describe the SAR image.",
        "primary_task": "captioning",
        "target_entities": [],
        "required_modalities": ["sar"],
        "temporal_requirement": "none",
        "spatial_output_required": False,
        "textual_output_required": True,
        "ambiguous": False
    })
    spec = extractor.extract("Describe the SAR image.")
    assert spec.primary_task == "captioning"
    assert spec.required_modalities == ["sar"]

def test_semantic_overlap_where_change():
    extractor = _mock_extractor({
        "query": "Where are the changes?",
        "primary_task": "change_detection",
        "target_entities": [],
        "required_modalities": ["unspecified"],
        "temporal_requirement": "before_after",
        "spatial_output_required": True,
        "textual_output_required": False,
        "ambiguous": False
    })
    spec = extractor.extract("Where are the changes?")
    # This proves the negative test: "where" doesn't force grounding if change is primary
    assert spec.primary_task == "change_detection"

def test_invalid_classification_enum():
    # Model hallucinates a task
    extractor = _mock_extractor({
        "query": "Hello",
        "primary_task": "magic_task",
    })
    spec = extractor.extract("Hello")
    assert spec.primary_task == "unknown"
    assert spec.ambiguous is True

def test_malformed_structured_output():
    mock_inference = MagicMock()
    mock_inference.generate.return_value = {
        "status": "ok",
        "response": "I think it is change detection"
    }
    extractor = TaskSpecExtractor(mock_inference)
    spec = extractor.extract("what changed")
    assert spec.primary_task == "unknown"
    assert spec.ambiguous is True

def test_semantic_consistency_validation_failure():
    # Mock model outputs change_vqa but forgets temporal_requirement
    extractor = _mock_extractor({
        "query": "Did the buildings increase?",
        "primary_task": "change_vqa",
        "target_entities": ["buildings"],
        "required_modalities": ["unspecified"],
        "temporal_requirement": "none",
        "spatial_output_required": False,
        "textual_output_required": True,
        "ambiguous": False
    })
    spec = extractor.extract("Did the buildings increase?")
    # Validator should catch this and flip ambiguous
    assert spec.primary_task == "change_vqa"
    assert spec.ambiguous is True

def test_no_tool_leakage():
    # Prompt explicitly bans tools, but if model spits out something weird that parses cleanly 
    # except it doesn't match the schema, it should fallback to unknown.
    # The schema guarantees we only get what we asked for.
    extractor = _mock_extractor({
        "query": "Use ChangeMamba",
        "primary_task": "change_detection",
        "target_entities": ["ChangeMamba"],
        "tool_call": "ChangeMamba",
        "required_modalities": ["unspecified"],
        "temporal_requirement": "before_after",
        "spatial_output_required": False,
        "textual_output_required": True,
        "ambiguous": False
    })
    spec = extractor.extract("Use ChangeMamba")
    # Pydantic ignores extra fields, but the key is we don't store or execute it
    assert not hasattr(spec, "tool_call")
    assert spec.primary_task == "change_detection"
