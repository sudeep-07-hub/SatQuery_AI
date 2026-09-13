import pytest
from mc4a_vqa.evidence_normalizer import to_evidence_object
from mc4b_temporal.evidence_normalizer import normalize_to_evidence
from mc5_evidence.schemas import validate_evidence_object

def test_mc4a_normalizer():
    job_id = "testjob"
    mc4a_output = {
        "textual_answer": "This is a sentence. This is another.",
        "model_confidences": {"paligemma_vqa": 0.8},
        "spatial_evidence": None
    }
    mc1_profile = {
        "image_count": 1,
        "image_1": {
            "filename": "img.tif",
            "modality": "sar",
            "acquisition_date": "2024-01-01T00:00:00Z"
        },
        "footprint": {"type": "Polygon", "coordinates": []}
    }
    mc2_spec = {"query": "test"}
    
    evs = to_evidence_object(mc4a_output, mc1_profile, mc2_spec, job_id)
    assert len(evs) == 1
    ev = evs[0]
    
    assert ev["evidence_id"] == f"{job_id}_mc4a_0"
    assert ev["claim"] == "This is a sentence."
    assert ev["processing_parameters"]["claim_is_truncated"] is True
    assert ev["evidence_type"] == "vqa_grounded_detection"
    assert ev["modality"] == "sar"
    assert ev["modality_contribution"]["sar"] == 1.0
    assert ev["source_model"] == "PALIGEMMA_VQA_TOOL"
    assert validate_evidence_object(ev)

def test_mc4b_normalizer():
    job_id = "testjob"
    mc4b_output = {
        "status": "SUCCESS",
        "changed_region_coordinates": [{"geometry": {"type": "Polygon"}}],
        "semantics": {"description": "Forest loss"},
        "change_statistics": {"changed_area_m2": 500, "changed_pixel_pct": 5},
        "model_confidence": 0.9,
        "source_model": "CHANGE_MAMBA",
        "change_score": 0.5
    }
    mc1_profile = {
        "image_count": 2,
        "image_1": {"modality": "sar", "acquisition_date": "t1", "filename": "1.tif"},
        "image_2": {"modality": "sar", "acquisition_date": "t2", "filename": "2.tif"}
    }
    query = "test"
    
    evs = normalize_to_evidence(mc4b_output, mc1_profile, query, job_id)
    assert len(evs) == 1
    ev = evs[0]
    
    assert ev["evidence_id"] == f"{job_id}_mc4b_0"
    assert "Forest loss" in ev["claim"]
    assert ev["processing_parameters"]["claim_source"] == "synthesized"
    assert ev["evidence_type"] == "bitemporal_change_detection"
    assert ev["modality"] == "sar"
    assert ev["modality_contribution"]["sar"] == 1.0
    assert ev["timestamp"] == {"t1": "t1", "t2": "t2"}
    assert ev["source_model"] == "CHANGE_MAMBA_TOOL"
    assert validate_evidence_object(ev)
