import pytest
import torch
from mc4b_temporal.pipeline import TemporalPipeline
from mc4b_temporal.tool_adapter import ChangeMambaAdapter

def test_pipeline_chronological_ordering():
    pipeline = TemporalPipeline()
    img1 = {"acquisition_date": "2023-01-01", "filename": "old.tif"}
    img2 = {"acquisition_date": "2023-02-01", "filename": "new.tif"}
    
    # Correct order
    earlier, later = pipeline._determine_chronological_order(img1, img2)
    assert earlier["filename"] == "old.tif"
    assert later["filename"] == "new.tif"
    
    # Reversed order
    earlier, later = pipeline._determine_chronological_order(img2, img1)
    assert earlier["filename"] == "old.tif"
    assert later["filename"] == "new.tif"
    
    # Missing order
    earlier, later = pipeline._determine_chronological_order({"filename": "a"}, {"filename": "b"})
    assert earlier is None
    assert later is None

def test_pipeline_model_unavailable():
    pipeline = TemporalPipeline()
    # It should correctly identify that ChangeMamba is not available
    assert pipeline.model_unavailable is True
    assert "ChangeMamba backbone requires CUDA" in pipeline.unavailable_reason
    
    # Calling execute should return MODEL_UNAVAILABLE
    res = pipeline.execute({}, torch.randn(1, 3, 256, 256), torch.randn(1, 3, 256, 256))
    assert res.status == "MODEL_UNAVAILABLE"
    assert "ChangeMamba" in res.model_id

def test_tool_adapter_no_fallback():
    adapter = ChangeMambaAdapter()
    
    mc1_profile = {
        "image_count": 2,
        "image_1": {"modality": "optical", "crs": "EPSG:32633"},
        "image_2": {"modality": "optical", "crs": "EPSG:32633"},
        "spatial_overlap": 0.8,
        "coregistration_score": 0.9
    }
    
    # The adapter should return passed=False and status=MODEL_UNAVAILABLE
    # rather than succeeding with a fake CNN mask.
    t1 = torch.randn(1, 3, 256, 256)
    t2 = torch.randn(1, 3, 256, 256)
    
    result = adapter.execute(mc1_profile, t1, t2)
    assert result.get("passed") is False
    assert result.get("status") == "MODEL_UNAVAILABLE"
    assert "ChangeMamba backbone requires CUDA" in result.get("reason", "")
