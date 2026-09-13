import pytest
from mc4b_temporal.tool_adapter import validate_preconditions
from mc3_planner.dispatcher import execute_plan

def test_preconditions_modality_mismatch():
    # Test that validate_preconditions catches a modality mismatch
    mc1_profile_mismatch = {
        "image_count": 2,
        "spatial_overlap": 0.95,
        "coregistration_score": 0.92,
        "image_1": {"crs": "EPSG:4326", "modality": "optical"},
        "image_2": {"crs": "EPSG:4326", "modality": "sar"}
    }
    
    result = validate_preconditions(mc1_profile_mismatch)
    assert not result["passed"]
    assert any(f["check"] == "same_modality" for f in result["failed"])

def test_preconditions_unsupported_modality():
    # Test that validate_preconditions catches unsupported modality (e.g., thermal)
    mc1_profile_unsupported = {
        "image_count": 2,
        "spatial_overlap": 0.95,
        "coregistration_score": 0.92,
        "image_1": {"crs": "EPSG:4326", "modality": "thermal"},
        "image_2": {"crs": "EPSG:4326", "modality": "thermal"}
    }
    
    result = validate_preconditions(mc1_profile_unsupported)
    assert not result["passed"]
    assert any(f["check"] == "supported_modality" for f in result["failed"])

def test_preconditions_sar_sar_success():
    # Test that validate_preconditions succeeds for SAR+SAR
    mc1_profile_success = {
        "image_count": 2,
        "spatial_overlap": 0.95,
        "coregistration_score": 0.92,
        "image_1": {"crs": "EPSG:4326", "modality": "sar"},
        "image_2": {"crs": "EPSG:4326", "modality": "sar"}
    }
    
    result = validate_preconditions(mc1_profile_success)
    assert result["passed"]

def test_dispatcher_modality_assertion():
    # Test that the dispatcher raises RuntimeError if the planner's resolved_modality
    # doesn't match the actual input profile modality.
    plan = {
        "status": "PLAN_READY",
        "execution_order": ["CHANGE_MAMBA"],
        "tool_parameters": {
            "CHANGE_MAMBA": {
                "resolved_modality": "optical"  # Planner says optical
            }
        }
    }
    
    mc1_profile_sar = {
        "image_count": 2,
        "image_1": {"modality": "sar"},
        "image_2": {"modality": "sar"}
    }
    
    # Missing tensors will normally cause TOOL_FAILED, but we want to see the assertion fail.
    # The assertion happens before tool execution but after tensor check, so we need dummy tensors.
    import torch
    tensors = {"t1": torch.rand(1,1,1), "t2": torch.rand(1,1,1)}
    
    with pytest.raises(RuntimeError, match="Modality mismatch assertion failed"):
        execute_plan(plan, mc1_profile_sar, tensors, query="")
