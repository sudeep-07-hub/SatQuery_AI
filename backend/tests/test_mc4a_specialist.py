import pytest
from PIL import Image
from mc4a_vqa.specialist import PaliGemmaVQASpecialist

def test_mc4a_contract():
    specialist = PaliGemmaVQASpecialist()
    # Force mock for speed
    specialist.is_mock = True
    from mc4a_vqa.mock_adapter import MockPaliGemmaVQAAdapter
    specialist.adapter = MockPaliGemmaVQAAdapter()

    mc1_profile = {
        "image_count": 1,
        "image_1": {"modality": "optical", "crs": "EPSG:32643", "gsd_m": 10.0, "filename": "test.tif"},
        "quality": {"optical": 0.9}
    }
    mc2_task_spec = {"primary_task": "visual_question_answering"}
    
    # Dummy image
    image = Image.new("RGB", (224, 224), color="green")
    query = "what crop is visible in the first image"

    res = specialist.run(image, query, mc1_profile, mc2_task_spec)
    
    # Verify contract keys
    expected_keys = {
        "textual_answer", "caption", "detected_entities", 
        "bounding_boxes", "segmentation_masks", "model_confidences", 
        "spatial_evidence", "domain_mismatch_flag"
    }
    assert expected_keys.issubset(res.keys())
    assert res["textual_answer"] == "wheat"  # from mock deterministic responses
    assert res["domain_mismatch_flag"] == False
    assert res["model_confidences"]["paligemma_vqa"] == 0.95
    assert res["spatial_evidence"] is not None

def test_modality_mismatch():
    specialist = PaliGemmaVQASpecialist()
    specialist.is_mock = True
    from mc4a_vqa.mock_adapter import MockPaliGemmaVQAAdapter
    specialist.adapter = MockPaliGemmaVQAAdapter()

    mc1_profile = {
        "image_count": 1,
        "image_1": {"modality": "sar", "crs": "EPSG:32643", "gsd_m": 10.0},
        "quality": {"sar": 0.9}
    }
    mc2_task_spec = {"primary_task": "visual_question_answering"}
    image = Image.new("RGB", (224, 224), color="green")
    query = "what crop is visible in the first image"

    res = specialist.run(image, query, mc1_profile, mc2_task_spec)
    
    assert res["domain_mismatch_flag"] == True
    # Confidence penalty 0.95 * 0.3 = 0.285
    assert abs(res["model_confidences"]["paligemma_vqa"] - 0.285) < 1e-5

def test_quality_gate():
    specialist = PaliGemmaVQASpecialist(quality_threshold=0.5)
    
    mc1_profile = {
        "image_count": 1,
        "image_1": {"modality": "optical", "crs": "EPSG:32643", "gsd_m": 10.0},
        "quality": {"optical": 0.4} # Below threshold
    }
    mc2_task_spec = {"primary_task": "visual_question_answering"}
    image = Image.new("RGB", (224, 224), color="green")
    query = "what crop is visible in the first image"

    res = specialist.run(image, query, mc1_profile, mc2_task_spec)
    
    assert res.get("blocked_reason") == "input_quality_below_threshold"
    assert res.get("textual_answer") is None
