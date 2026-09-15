import pytest
import json
from unittest.mock import MagicMock
from qwen.schemas import TaskSpec
from qwen.extractor import TaskSpecExtractor

def test_taskspec_schema_validation():
    # Valid TaskSpec
    spec = TaskSpec(
        query="What is in this image?",
        primary_task="single_image_vqa"
    )
    assert spec.primary_task == "single_image_vqa"
    assert spec.ambiguous is False

def test_taskspec_semantic_consistency():
    # Inconsistent: change detection without temporal requirement
    spec = TaskSpec(
        query="Did it change?",
        primary_task="change_detection",
        temporal_requirement="none"
    )
    # The validator should mark it ambiguous
    assert spec.ambiguous is True
    
    # Inconsistent: fusion but only optical
    spec2 = TaskSpec(
        query="Fuse it",
        primary_task="cross_modal_fusion",
        required_modalities=["optical"]
    )
    assert spec2.ambiguous is True

def test_extractor_success():
    mock_inference = MagicMock()
    mock_inference.generate.return_value = {
        "status": "ok",
        "response": json.dumps({
            "query": "Where is the building?",
            "primary_task": "grounding",
            "target_entities": ["building"],
            "required_modalities": ["unspecified"],
            "temporal_requirement": "none",
            "spatial_output_required": True,
            "textual_output_required": True,
            "ambiguous": False
        })
    }
    
    extractor = TaskSpecExtractor(mock_inference)
    spec = extractor.extract("Where is the building?")
    
    assert spec.primary_task == "grounding"
    assert "building" in spec.target_entities
    assert spec.spatial_output_required is True
    assert spec.ambiguous is False

def test_extractor_json_failure():
    mock_inference = MagicMock()
    mock_inference.generate.return_value = {
        "status": "ok",
        "response": "This is just plain text, not JSON!"
    }
    
    extractor = TaskSpecExtractor(mock_inference)
    spec = extractor.extract("Find the car.")
    
    # Should fallback gracefully to unknown + ambiguous
    assert spec.primary_task == "unknown"
    assert spec.ambiguous is True
    
def test_extractor_validation_failure():
    mock_inference = MagicMock()
    mock_inference.generate.return_value = {
        "status": "ok",
        "response": json.dumps({
            "query": "Hello",
            "primary_task": "not_a_real_task",  # Invalid enum
        })
    }
    
    extractor = TaskSpecExtractor(mock_inference)
    spec = extractor.extract("Hello")
    
    # Pydantic validation will fail, should fallback
    assert spec.primary_task == "unknown"
    assert spec.ambiguous is True

def test_extractor_ambiguity():
    mock_inference = MagicMock()
    mock_inference.generate.return_value = {
        "status": "ok",
        "response": json.dumps({
            "query": "Analyze this.",
            "primary_task": "unknown",
            "target_entities": [],
            "required_modalities": ["unspecified"],
            "temporal_requirement": "unknown",
            "spatial_output_required": False,
            "textual_output_required": True,
            "ambiguous": True
        })
    }
    
    extractor = TaskSpecExtractor(mock_inference)
    spec = extractor.extract("Analyze this.")
    
    assert spec.ambiguous is True
    assert spec.primary_task == "unknown"

def test_json_roundtrip():
    spec = TaskSpec(
        query="Find the car",
        primary_task="grounding",
        target_entities=["car"],
        spatial_output_required=True
    )
    
    j = spec.model_dump_json()
    spec_restored = TaskSpec.model_validate_json(j)
    
    assert spec_restored.primary_task == "grounding"
    assert spec_restored.spatial_output_required is True
