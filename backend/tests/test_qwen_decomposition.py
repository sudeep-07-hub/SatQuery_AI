import pytest
import json
from unittest.mock import MagicMock
from qwen.schemas import TaskSpec, SubtaskSpec, DecompositionResult
from qwen.decomposer import QueryDecomposer

def _mock_decomposer(response_dict):
    mock_inference = MagicMock()
    mock_inference.generate.return_value = {
        "status": "ok",
        "response": json.dumps(response_dict)
    }
    return QueryDecomposer(mock_inference)

def _dummy_task_spec(primary_task="single_image_vqa", temporal_requirement="none", required_modalities=["unspecified"]):
    return TaskSpec(
        query="dummy",
        primary_task=primary_task,
        temporal_requirement=temporal_requirement,
        required_modalities=required_modalities
    )

def test_atomic_vqa():
    decomposer = _mock_decomposer({
        "is_compound": False,
        "ambiguous": False,
        "subtasks": [{
            "subtask_id": "subtask_1",
            "description": "Answer VQA question",
            "primary_task": "single_image_vqa",
            "target_entities": [],
            "required_modalities": ["unspecified"],
            "temporal_requirement": "none",
            "spatial_output_required": False,
            "textual_output_required": True,
            "depends_on": []
        }]
    })
    res = decomposer.decompose("What is visible in this image?", _dummy_task_spec("single_image_vqa"))
    assert res.is_compound is False
    assert len(res.subtasks) == 1
    assert res.subtasks[0].primary_task == "single_image_vqa"

def test_atomic_captioning():
    decomposer = _mock_decomposer({
        "is_compound": False,
        "ambiguous": False,
        "subtasks": [{
            "subtask_id": "st1",
            "description": "Describe scene",
            "primary_task": "captioning",
            "target_entities": [],
            "required_modalities": ["unspecified"],
            "temporal_requirement": "none",
            "spatial_output_required": False,
            "textual_output_required": True,
            "depends_on": []
        }]
    })
    res = decomposer.decompose("Describe this scene.", _dummy_task_spec("captioning"))
    assert res.is_compound is False
    assert res.subtasks[0].primary_task == "captioning"

def test_atomic_grounding():
    decomposer = _mock_decomposer({
        "is_compound": False,
        "ambiguous": False,
        "subtasks": [{
            "subtask_id": "st1",
            "description": "Locate buildings",
            "primary_task": "grounding",
            "target_entities": ["buildings"],
            "required_modalities": ["unspecified"],
            "temporal_requirement": "none",
            "spatial_output_required": True,
            "textual_output_required": False,
            "depends_on": []
        }]
    })
    res = decomposer.decompose("Where are the buildings?", _dummy_task_spec("grounding"))
    assert res.is_compound is False
    assert res.subtasks[0].primary_task == "grounding"

def test_atomic_change_detection():
    decomposer = _mock_decomposer({
        "is_compound": False,
        "ambiguous": False,
        "subtasks": [{
            "subtask_id": "st1",
            "description": "Detect changes",
            "primary_task": "change_detection",
            "target_entities": [],
            "required_modalities": ["unspecified"],
            "temporal_requirement": "before_after",
            "spatial_output_required": False,
            "textual_output_required": True,
            "depends_on": []
        }]
    })
    res = decomposer.decompose("What changed between these images?", _dummy_task_spec("change_detection", "before_after"))
    assert res.is_compound is False
    assert res.subtasks[0].primary_task == "change_detection"

def test_atomic_change_vqa():
    decomposer = _mock_decomposer({
        "is_compound": False,
        "ambiguous": False,
        "subtasks": [{
            "subtask_id": "st1",
            "description": "Detect built up area increase",
            "primary_task": "change_vqa",
            "target_entities": ["built-up area"],
            "required_modalities": ["unspecified"],
            "temporal_requirement": "before_after",
            "spatial_output_required": False,
            "textual_output_required": True,
            "depends_on": []
        }]
    })
    res = decomposer.decompose("Did the built-up area increase?", _dummy_task_spec("change_vqa", "before_after"))
    assert res.is_compound is False
    assert res.subtasks[0].primary_task == "change_vqa"

def test_atomic_cross_modal():
    decomposer = _mock_decomposer({
        "is_compound": False,
        "ambiguous": False,
        "subtasks": [{
            "subtask_id": "st1",
            "description": "Compare SAR and optical",
            "primary_task": "cross_modal_fusion",
            "target_entities": [],
            "required_modalities": ["optical", "sar"],
            "temporal_requirement": "none",
            "spatial_output_required": False,
            "textual_output_required": True,
            "depends_on": []
        }]
    })
    res = decomposer.decompose("Compare the SAR and optical imagery.", _dummy_task_spec("cross_modal_fusion", required_modalities=["optical", "sar"]))
    assert res.is_compound is False
    assert res.subtasks[0].primary_task == "cross_modal_fusion"

def test_compound_change_grounding():
    decomposer = _mock_decomposer({
        "is_compound": True,
        "ambiguous": False,
        "subtasks": [
            {
                "subtask_id": "st1",
                "description": "Identify new buildings",
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
                "description": "Locate new buildings",
                "primary_task": "grounding",
                "target_entities": ["new buildings"],
                "required_modalities": ["unspecified"],
                "temporal_requirement": "none",
                "spatial_output_required": True,
                "textual_output_required": False,
                "depends_on": ["st1"]
            }
        ]
    })
    res = decomposer.decompose("Identify new buildings and show where they are.", _dummy_task_spec("change_detection", "before_after"))
    assert res.is_compound is True
    assert len(res.subtasks) == 2
    assert res.subtasks[1].depends_on == ["st1"]

def test_compound_change_vqa_localization():
    decomposer = _mock_decomposer({
        "is_compound": True,
        "ambiguous": False,
        "subtasks": [
            {
                "subtask_id": "st1",
                "description": "Did area increase",
                "primary_task": "change_vqa",
                "target_entities": ["built-up area"],
                "required_modalities": ["unspecified"],
                "temporal_requirement": "before_after",
                "spatial_output_required": False,
                "textual_output_required": True,
                "depends_on": []
            },
            {
                "subtask_id": "st2",
                "description": "Locate development",
                "primary_task": "grounding",
                "target_entities": ["new development"],
                "required_modalities": ["unspecified"],
                "temporal_requirement": "before_after",
                "spatial_output_required": True,
                "textual_output_required": False,
                "depends_on": ["st1"]
            }
        ]
    })
    res = decomposer.decompose("Did the built-up area increase and where did the new development occur?", _dummy_task_spec("change_vqa", "before_after"))
    assert res.is_compound is True
    assert res.subtasks[0].primary_task == "change_vqa"
    assert res.subtasks[1].primary_task == "grounding"
    assert res.subtasks[1].temporal_requirement == "before_after" # Temporal context propagated

def test_compound_caption_grounding():
    decomposer = _mock_decomposer({
        "is_compound": True,
        "ambiguous": False,
        "subtasks": [
            {
                "subtask_id": "st1",
                "description": "Describe scene",
                "primary_task": "captioning",
                "target_entities": [],
                "required_modalities": ["unspecified"],
                "temporal_requirement": "none",
                "spatial_output_required": False,
                "textual_output_required": True,
                "depends_on": []
            },
            {
                "subtask_id": "st2",
                "description": "Locate fields",
                "primary_task": "grounding",
                "target_entities": ["agricultural fields"],
                "required_modalities": ["unspecified"],
                "temporal_requirement": "none",
                "spatial_output_required": True,
                "textual_output_required": False,
                "depends_on": []
            }
        ]
    })
    res = decomposer.decompose("Describe the scene and locate the agricultural fields.", _dummy_task_spec("captioning"))
    assert res.is_compound is True
    assert len(res.subtasks) == 2

def test_temporal_context_propagation():
    decomposer = _mock_decomposer({
        "is_compound": True,
        "ambiguous": False,
        "subtasks": [
            {
                "subtask_id": "st1",
                "description": "Identify new buildings",
                "primary_task": "change_detection",
                "target_entities": ["buildings"],
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
                "target_entities": ["buildings"],
                "required_modalities": ["unspecified"],
                "temporal_requirement": "before_after",
                "spatial_output_required": True,
                "textual_output_required": False,
                "depends_on": ["st1"]
            }
        ]
    })
    res = decomposer.decompose("Between 2024 and 2025, identify new buildings and locate them.", _dummy_task_spec("change_detection", "before_after"))
    assert res.subtasks[0].temporal_requirement == "before_after"
    assert res.subtasks[1].temporal_requirement == "before_after"

def test_entity_propagation():
    decomposer = _mock_decomposer({
        "is_compound": True,
        "ambiguous": False,
        "subtasks": [
            {
                "subtask_id": "st1",
                "description": "Find buildings",
                "primary_task": "change_detection",
                "target_entities": ["buildings"],
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
                "target_entities": ["buildings"], # Entity propagates
                "required_modalities": ["unspecified"],
                "temporal_requirement": "none",
                "spatial_output_required": True,
                "textual_output_required": False,
                "depends_on": ["st1"]
            }
        ]
    })
    res = decomposer.decompose("Find new buildings and locate them.", _dummy_task_spec("change_detection", "before_after"))
    assert "buildings" in res.subtasks[0].target_entities
    assert "buildings" in res.subtasks[1].target_entities

def test_no_over_decomposition():
    decomposer = _mock_decomposer({
        "is_compound": False, # Should be false
        "ambiguous": False,
        "subtasks": [{
            "subtask_id": "st1",
            "description": "Describe what changed",
            "primary_task": "change_detection",
            "target_entities": [],
            "required_modalities": ["unspecified"],
            "temporal_requirement": "before_after",
            "spatial_output_required": False,
            "textual_output_required": True,
            "depends_on": []
        }]
    })
    res = decomposer.decompose("Describe what changed between the images.", _dummy_task_spec("change_detection", "before_after"))
    assert res.is_compound is False
    assert len(res.subtasks) == 1

def test_no_under_decomposition():
    # Model generates 2 tasks correctly instead of collapsing
    decomposer = _mock_decomposer({
        "is_compound": True,
        "ambiguous": False,
        "subtasks": [
            {
                "subtask_id": "st1",
                "description": "Compare images",
                "primary_task": "change_detection",
                "target_entities": ["buildings"],
                "required_modalities": ["unspecified"],
                "temporal_requirement": "before_after",
                "spatial_output_required": False,
                "textual_output_required": True,
                "depends_on": []
            },
            {
                "subtask_id": "st2",
                "description": "Locate buildings",
                "primary_task": "grounding",
                "target_entities": ["buildings"],
                "required_modalities": ["unspecified"],
                "temporal_requirement": "none",
                "spatial_output_required": True,
                "textual_output_required": False,
                "depends_on": ["st1"]
            }
        ]
    })
    res = decomposer.decompose("Compare the images, identify new buildings, and tell me where they are.", _dummy_task_spec("change_detection", "before_after"))
    assert len(res.subtasks) == 2

def test_ambiguous_query():
    decomposer = _mock_decomposer({
        "is_compound": False,
        "ambiguous": True,
        "subtasks": []
    })
    # Will fallback to dummy task spec
    dummy = TaskSpec(query="dummy", primary_task="unknown", ambiguous=True)
    res = decomposer.decompose("Analyze these images and tell me something useful.", dummy)
    assert res.ambiguous is True
    # Fallback populates one atomic task matching the primary unknown task
    assert res.subtasks[0].primary_task == "unknown"

def test_tool_leakage_rejected():
    # If a model hallucinates tool names in extra JSON keys, pydantic strips it
    # If it tries to use them as primary_task, enum validation fails -> fallback
    decomposer = _mock_decomposer({
        "is_compound": True,
        "ambiguous": False,
        "subtasks": [
            {
                "subtask_id": "st1",
                "description": "Run ChangeMamba",
                "primary_task": "magic_tool_mamba", # INVALID ENUM
                "target_entities": [],
                "required_modalities": ["unspecified"],
                "temporal_requirement": "none",
                "spatial_output_required": False,
                "textual_output_required": True,
                "depends_on": []
            }
        ]
    })
    res = decomposer.decompose("Run ChangeMamba", _dummy_task_spec("unknown"))
    # The invalid enum causes Pydantic to fail and the decomposer to fallback safely
    assert res.ambiguous is True
    assert res.subtasks[0].primary_task == "unknown"

def test_json_validation_fallback():
    mock_inference = MagicMock()
    mock_inference.generate.return_value = {
        "status": "ok",
        "response": "I think we should decompose this..."
    }
    decomposer = QueryDecomposer(mock_inference)
    dummy = _dummy_task_spec("captioning")
    res = decomposer.decompose("describe", dummy)
    # Safe fallback
    assert res.is_compound is False
    assert res.subtasks[0].primary_task == "captioning"
    assert res.ambiguous is True # Correctly marked as ambiguous due to parse failure

def test_dependency_validation():
    # Check that schema successfully loads depends_on
    decomposer = _mock_decomposer({
        "is_compound": True,
        "ambiguous": False,
        "subtasks": [
            {
                "subtask_id": "st1",
                "description": "A",
                "primary_task": "captioning",
                "target_entities": [],
                "required_modalities": ["unspecified"],
                "temporal_requirement": "none",
                "spatial_output_required": False,
                "textual_output_required": True,
                "depends_on": []
            },
            {
                "subtask_id": "st2",
                "description": "B",
                "primary_task": "grounding",
                "target_entities": [],
                "required_modalities": ["unspecified"],
                "temporal_requirement": "none",
                "spatial_output_required": True,
                "textual_output_required": False,
                "depends_on": ["st1"]
            }
        ]
    })
    res = decomposer.decompose("a and b", _dummy_task_spec("captioning"))
    assert res.subtasks[1].depends_on == ["st1"]

def test_duplicate_subtask_detection():
    # If the LLM generates identical subtasks, they are returned. Duplicate collapsing is Phase 3. 
    # For now, we test the schema supports the raw parse.
    decomposer = _mock_decomposer({
        "is_compound": True,
        "ambiguous": False,
        "subtasks": [
            {
                "subtask_id": "st1",
                "description": "A",
                "primary_task": "captioning",
                "target_entities": [],
                "required_modalities": ["unspecified"],
                "temporal_requirement": "none",
                "spatial_output_required": False,
                "textual_output_required": True,
                "depends_on": []
            },
            {
                "subtask_id": "st2",
                "description": "A",
                "primary_task": "captioning",
                "target_entities": [],
                "required_modalities": ["unspecified"],
                "temporal_requirement": "none",
                "spatial_output_required": False,
                "textual_output_required": True,
                "depends_on": []
            }
        ]
    })
    res = decomposer.decompose("a and a", _dummy_task_spec("captioning"))
    assert len(res.subtasks) == 2

def test_json_round_trip():
    res = DecompositionResult(
        original_query="Where is it",
        primary_task_spec=_dummy_task_spec("grounding"),
        is_compound=False,
        subtasks=[
            SubtaskSpec(
                subtask_id="st1",
                description="Find it",
                primary_task="grounding",
                spatial_output_required=True
            )
        ]
    )
    j = res.model_dump_json()
    res2 = DecompositionResult.model_validate_json(j)
    assert res2.subtasks[0].primary_task == "grounding"

def test_semantic_consistency_grounding_coercion():
    # Model generates a grounding task but forgets spatial_output_required=True
    decomposer = _mock_decomposer({
        "is_compound": False,
        "ambiguous": False,
        "subtasks": [{
            "subtask_id": "st1",
            "description": "Locate",
            "primary_task": "grounding",
            "target_entities": [],
            "required_modalities": ["unspecified"],
            "temporal_requirement": "none",
            "spatial_output_required": False, # FORGOT THIS
            "textual_output_required": True,
            "depends_on": []
        }]
    })
    res = decomposer.decompose("Where", _dummy_task_spec("grounding"))
    # Pydantic validator should have coerced it to True
    assert res.subtasks[0].spatial_output_required is True
