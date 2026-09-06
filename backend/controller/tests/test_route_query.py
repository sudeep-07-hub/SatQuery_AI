"""
test_route_query.py — Phase 3 tests for the Dispatcher.
"""

import pytest
from controller.controller import route_query
from controller.engine_registry import ENGINE_REGISTRY

def _base_profile(**overrides):
    base = {
        "image_count": 1,
        "query": "test query",
        "spatial_overlap": None,
        "coregistration_score": None,
        "relationship": "single_image",
        "quality": {},
        "warnings": [],
        "task_executable": True,
        "image_1": {
            "modality": "optical",
        }
    }
    base.update(overrides)
    return base

def test_rejected_case():
    # Single image + change detection wording (which requires 2 images)
    profile = _base_profile(image_count=1, image_1={"modality": "optical"})
    query = "has the built-up area increased?"
    
    result = route_query(query, profile)
    assert result["status"] == "rejected"
    assert "Change detection requires exactly 2 images" in result["reason"]
    assert result["answer"] is None

def test_success_case():
    # 2-image optical+SAR + fusion query
    profile = _base_profile(
        image_count=2,
        spatial_overlap=0.8,
        image_1={"modality": "optical"},
        image_2={"modality": "sar"}
    )
    query = "fuse optical and sar together"
    
    result = route_query(query, profile)
    assert result["status"] == "ok"
    assert result["task_type"] == "cross_modal_fusion"
    assert result["answer"] is not None

def test_grounding_regression():
    # Ensure grounding query runs without KeyError
    profile = _base_profile(image_count=1)
    query = "highlight the buildings"
    
    result = route_query(query, profile)
    assert result["status"] == "ok"
    assert result["task_type"] == "grounding"
    assert result["answer"] is not None

def test_error_case(monkeypatch):
    # Mock the handler to raise an exception
    def mock_handler(*args, **kwargs):
        raise ValueError("Simulated engine crash")
        
    monkeypatch.setitem(
        ENGINE_REGISTRY["single_image_vqa"], 
        "handler", 
        mock_handler
    )
    
    profile = _base_profile(image_count=1)
    query = "what is this?"
    
    result = route_query(query, profile)
    assert result["status"] == "error"
    assert result["error"] == "Internal engine error occurred."
    assert "Simulated engine crash" not in result.get("error", "")

def test_trace_shape():
    profile = _base_profile(image_count=1)
    query = "describe this"
    
    result = route_query(query, profile)
    expected_keys = {
        "status", "task_type", "engine", 
        "confidence", "answer", "evidence_region"
    }
    # Reason and error are optional depending on status, but checking expected subset
    assert expected_keys.issubset(set(result.keys()))
