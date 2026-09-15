import pytest
from mc1.schemas import ObservationProfile, SpatialProfile, SensorProfile, QualityProfile, ConditionProfile
from mc1.compatibility import assess_compatibility

def build_mock_obs(file_format="GTiff", score=0.9, nodata=0.0, cloud="clear", modality="optical"):
    return ObservationProfile(
        observation_id="test",
        file_source="test.tif",
        file_format=file_format,
        spatial=SpatialProfile(),
        sensor=SensorProfile(modality=modality, sensor="test_sensor"),
        quality=QualityProfile(score=score),
        conditions=ConditionProfile(nodata_fraction=nodata, cloud_status=cloud)
    )

def test_hard_compatible():
    obs = [build_mock_obs(), build_mock_obs()]
    comp = assess_compatibility(obs, spatial_overlap=0.8, coreg_score=0.9, relationship="same_image")
    assert comp.hard_compatible is True
    assert comp.soft_suitability_score == 1.0

def test_hard_incompatible_format():
    obs = [build_mock_obs(file_format="unknown"), build_mock_obs()]
    comp = assess_compatibility(obs, spatial_overlap=0.8, coreg_score=0.9, relationship="same_image")
    assert comp.hard_compatible is False
    assert any("unknown or invalid" in w for w in comp.hard_failures)

def test_hard_incompatible_no_overlap():
    obs = [build_mock_obs(), build_mock_obs()]
    comp = assess_compatibility(obs, spatial_overlap=0.0, coreg_score=0.9, relationship="unknown")
    assert comp.hard_compatible is False
    assert any("No spatial overlap" in w for w in comp.hard_failures)

def test_soft_degradation_quality():
    obs = [build_mock_obs(score=0.4)]
    comp = assess_compatibility(obs, spatial_overlap=None, coreg_score=None, relationship="single")
    assert comp.hard_compatible is True
    assert comp.soft_suitability_score < 1.0
    assert any("quality is degraded" in w for w in comp.warnings)

def test_soft_degradation_nodata():
    obs = [build_mock_obs(nodata=0.2)]
    comp = assess_compatibility(obs, spatial_overlap=None, coreg_score=None, relationship="single")
    assert comp.hard_compatible is True
    assert comp.soft_suitability_score < 1.0
    assert any("significant NoData" in w for w in comp.warnings)

def test_soft_degradation_cloud():
    obs = [build_mock_obs(cloud="present")]
    comp = assess_compatibility(obs, spatial_overlap=None, coreg_score=None, relationship="single")
    assert comp.hard_compatible is True
    assert comp.soft_suitability_score < 1.0
    assert any("cloud contaminated" in w for w in comp.warnings)

def test_cross_modal_compatibility():
    # Optical + SAR is NOT hard incompatible
    obs = [build_mock_obs(modality="optical"), build_mock_obs(modality="sar")]
    comp = assess_compatibility(obs, spatial_overlap=0.5, coreg_score=0.9, relationship="unknown")
    assert comp.hard_compatible is True
    # Has a partial overlap warning
    assert any("Partial spatial overlap" in w for w in comp.warnings)
