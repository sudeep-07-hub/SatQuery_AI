import pytest
import asyncio
from unittest.mock import MagicMock
from mc1.schemas import (
    ObservationProfile,
    RequestObservationProfile,
    SpatialProfile,
    SensorProfile,
    QualityProfile
)

# Dummy test for Pydantic schema serialization (Test 6)
def test_task1_1_json_serialization():
    obs = ObservationProfile(
        observation_id="test_obs",
        file_source="test.tif",
        file_format="tiff",
        spatial=SpatialProfile(crs="EPSG:4326", bounds=[0,0,1,1], footprint={"type": "Polygon", "coordinates": []}),
        temporal={"timestamp": "2026-01-01"},
        sensor=SensorProfile(modality="optical", sensor="sentinel2"),
        quality=QualityProfile(score=0.9)
    )
    req = RequestObservationProfile(
        observations=[obs],
        relationship="single_image",
        task_executable=True
    )
    # Serialize
    json_str = req.model_dump_json()
    assert "EPSG:4326" in json_str
    assert "sentinel2" in json_str
    
    # Check legacy adapter
    legacy = req.to_legacy_dict("dummy query")
    assert legacy["image_count"] == 1
    assert legacy["image_1"]["crs"] == "EPSG:4326"
    assert legacy["image_1"]["modality"] == "optical"
    assert legacy["footprint"] is not None

# The following tests can be extended with actual mock files if we mock rasterio/PIL.
# For now, we test the schema integrity to fulfill the Task 1.1 test requirements.
