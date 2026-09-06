"""
test_classify_task.py — Phase 2 tests for classify_task.

Tests:
1. One test per cue-table row using a representative query.
2. Input-shape fallback for cross_modal_fusion.
3. Default fallback for unmatched queries.
4. Precedence tests (matches two cues -> higher priority wins).
5. Edge cases: empty/whitespace queries.
"""

from controller.controller import classify_task

def _dummy_profile():
    return {"image_count": 1, "image_1": {"modality": "optical"}}

def _optical_sar_profile():
    return {
        "image_count": 2,
        "image_1": {"modality": "optical"},
        "image_2": {"modality": "sar"}
    }

def test_cue_change_detection():
    query = "has the built-up area increased between these dates?"
    assert classify_task(query, _dummy_profile()) == "change_detection"

def test_cue_cross_modal_fusion():
    query = "fuse the sar and optical images together"
    assert classify_task(query, _dummy_profile()) == "cross_modal_fusion"

def test_cue_grounding():
    query = "highlight the flooded regions"
    assert classify_task(query, _dummy_profile()) == "grounding"

def test_cue_captioning():
    query = "describe this scene"
    assert classify_task(query, _dummy_profile()) == "captioning"

def test_cue_single_image_vqa_default():
    # A query matching no cues
    query = "what is the dominant crop type?"
    assert classify_task(query, _dummy_profile()) == "single_image_vqa"

def test_input_shape_fallback_fusion():
    # Query has NO modality keywords, but profile is optical+SAR
    query = "find the ships"
    assert classify_task(query, _optical_sar_profile()) == "cross_modal_fusion"

def test_precedence_change_over_grounding():
    # Both "highlight" (grounding) and "changed" (change_detection)
    query = "highlight the areas that have changed"
    assert classify_task(query, _dummy_profile()) == "change_detection"

def test_precedence_fusion_over_grounding():
    # "locate" (grounding) + "sar" and "optical" (fusion)
    query = "locate the building using sar and optical"
    assert classify_task(query, _dummy_profile()) == "cross_modal_fusion"

def test_edge_case_empty_string():
    assert classify_task("", _dummy_profile()) == "single_image_vqa"
    assert classify_task("   ", _dummy_profile()) == "single_image_vqa"
