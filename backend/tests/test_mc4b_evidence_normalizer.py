import pytest
from mc4b_temporal.evidence_normalizer import normalize_to_evidence, _pixel_to_geo
from mc4b_temporal.result import TemporalResult, RawTemporalOutput

def test_model_unavailable_returns_no_evidence():
    result = TemporalResult(status="MODEL_UNAVAILABLE")
    evidence = normalize_to_evidence(result, "job_123")
    assert evidence == []

def test_pixel_to_geo_transformation():
    # 2x2 mask with active pixel at (1, 1)
    mask = [
        [0, 0],
        [0, 1]
    ]
    # affine: [a, b, c0, d, e, f0]
    # Identity transform with origin 0,0 and 1-unit spacing
    affine = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    
    regions = _pixel_to_geo(mask, affine, "EPSG:4326")
    assert len(regions) == 1
    coords = regions[0]["geometry"]["coordinates"][0]
    
    # The active pixel is (c=1, r=1)
    # The bounds should center around c+0.5=1.5, r+0.5=1.5
    # Since we mapped a single pixel, min_c=1, min_r=1, max_c=1, max_r=1
    # x = 1.5, y = 1.5
    assert [1.5, 1.5] in coords

def test_semantic_safety_and_provenance():
    raw = RawTemporalOutput(
        binary_mask=[[1]],
        output_shape=[1, 1],
        dtype="int",
        device="cpu"
    )
    result = TemporalResult(
        status="SUCCESS",
        before_observation_id="obs_1",
        after_observation_id="obs_2",
        crs="EPSG:4326",
        affine_transform=[1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        raw_output=raw,
        model_id="ChangeMamba"
    )
    
    evidence = normalize_to_evidence(result, "job_123")
    assert len(evidence) == 1
    
    e = evidence[0]
    assert e["timestamp"] == {"t1": "obs_1", "t2": "obs_2"}
    assert "building" not in e["claim"].lower()
    assert e["claim"] == "Model-predicted change region detected between obs_1 and obs_2."
    assert e["source_model"] == "ChangeMamba"
