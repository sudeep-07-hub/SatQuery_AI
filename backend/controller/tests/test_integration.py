"""
test_integration.py — Phase 4 tests for End-to-End and Demo Readiness.
"""

import pytest
from controller.controller import route_query

def _build_realistic_profile(image_count, modality1, modality2=None, overlap=None):
    """Helper to build realistic Stage 1 output profiles."""
    profile = {
        "image_count": image_count,
        "query": "placeholder",
        "spatial_overlap": overlap,
        "coregistration_score": 0.8 if overlap else None,
        "relationship": "single_image" if image_count == 1 else "multi_temporal",
        "quality": {modality1: 0.9},
        "warnings": [],
        "task_executable": True,
        "image_1": {
            "filename": "img1.tif",
            "modality": modality1,
            "sensor": f"Generic {modality1.capitalize()}",
            "gsd_m": 10.0,
            "crs": "EPSG:32643",
            "acquisition_date": "2024-01-01 10:00:00",
        }
    }
    
    if image_count == 2 and modality2:
        profile["image_2"] = {
            "filename": "img2.tif",
            "modality": modality2,
            "sensor": f"Generic {modality2.capitalize()}",
            "gsd_m": 10.0,
            "crs": "EPSG:32643",
            "acquisition_date": "2024-02-01 10:00:00",
        }
        profile["quality"][modality2] = 0.85
        if modality1 != modality2:
            profile["relationship"] = "cross_modal"
            
    return profile

def _is_ui_renderable(result):
    """Helper to check if the dict matches the expected UI trace contract."""
    expected_keys = {"status"}
    if result["status"] == "rejected":
        expected_keys.update({"reason", "answer"})
    elif result["status"] == "ok":
        expected_keys.update({"task_type", "engine", "confidence", "answer", "evidence_region"})
    elif result["status"] == "error":
        expected_keys.update({"error"})
        
    return expected_keys.issubset(set(result.keys()))


def test_demo_reject_single_image_change():
    """
    Demo scenario: User asks a change question but only uploads one image.
    Should reject gracefully, not crash.
    """
    profile = _build_realistic_profile(1, "optical")
    query = "has the built-up area changed?"
    
    result = route_query(query, profile)
    
    assert result["status"] == "rejected"
    assert result["answer"] is None
    assert "Change detection requires exactly 2 images" in result["reason"]
    assert _is_ui_renderable(result)

def test_demo_ambiguous_cross_modal():
    """
    Demo scenario: User uploads optical + SAR, but query doesn't mention them.
    Should still route to cross_modal_fusion via input shape.
    """
    profile = _build_realistic_profile(2, "optical", "sar", overlap=0.7)
    query = "find the ships in the water"
    
    result = route_query(query, profile)
    
    assert result["status"] == "ok"
    assert result["task_type"] == "cross_modal_fusion"
    assert _is_ui_renderable(result)

def test_smoke_all_5_task_types():
    """
    Runs all 5 task types back-to-back with valid profiles to ensure none raise.
    """
    # 1. VQA
    q_vqa = "what is this?"
    p_vqa = _build_realistic_profile(1, "optical")
    assert route_query(q_vqa, p_vqa)["status"] == "ok"
    
    # 2. Captioning
    q_cap = "describe this"
    assert route_query(q_cap, p_vqa)["status"] == "ok"
    
    # 3. Grounding
    q_grd = "highlight the road"
    assert route_query(q_grd, p_vqa)["status"] == "ok"
    
    # 4. Change Detection
    q_chg = "where did it increase?"
    p_chg = _build_realistic_profile(2, "optical", "optical", overlap=0.8)
    assert route_query(q_chg, p_chg)["status"] == "ok"
    
    # 5. Cross Modal Fusion
    q_fus = "fuse sar and optical"
    p_fus = _build_realistic_profile(2, "optical", "sar", overlap=0.7)
    assert route_query(q_fus, p_fus)["status"] == "ok"

def test_audit_trail_shape_contracts():
    """
    Confirm the audit-trail/logging layer receives a well-formed trace object
    for both a success and a rejection case.
    """
    # Success case
    q_success = "describe this scene"
    p_success = _build_realistic_profile(1, "optical")
    res_success = route_query(q_success, p_success)
    
    assert _is_ui_renderable(res_success)
    assert res_success["status"] == "ok"
    assert "engine" in res_success
    
    # Rejection case
    q_reject = "what changed?"
    p_reject = _build_realistic_profile(1, "optical")
    res_reject = route_query(q_reject, p_reject)
    
    assert _is_ui_renderable(res_reject)
    assert res_reject["status"] == "rejected"
    assert "reason" in res_reject
