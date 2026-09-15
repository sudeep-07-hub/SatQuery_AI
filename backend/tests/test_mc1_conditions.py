import pytest
from mc1.classifier import assess_conditions
from mc1.schemas import ConditionProfile

def test_nodata_fully_valid():
    # 100% valid
    meta = {
        "image_statistics": {
            "valid_fraction": 1.0
        }
    }
    cond = assess_conditions(meta, "optical")
    assert cond.nodata_fraction == 0.0
    assert cond.nodata_status == "clear"
    assert cond.nodata_source == "dataset_mask"

def test_nodata_partial():
    # Explicit missing fraction
    meta = {
        "image_statistics": {
            "valid_fraction": 0.8
        }
    }
    cond = assess_conditions(meta, "optical")
    assert abs(cond.nodata_fraction - 0.2) < 1e-5
    assert cond.nodata_status == "present"
    assert cond.nodata_source == "dataset_mask"

def test_cloud_optical_metadata():
    # Correctly parses explicit product cloud cover
    meta = {
        "cloud_fraction": 0.15
    }
    cond = assess_conditions(meta, "optical")
    assert cond.cloud_fraction == 0.15
    assert cond.cloud_status == "present"
    assert cond.cloud_source == "product_metadata"

def test_cloud_optical_unknown():
    # Missing metadata yields "unknown"
    meta = {}
    cond = assess_conditions(meta, "optical")
    assert cond.cloud_fraction is None
    assert cond.cloud_status == "unknown"
    assert cond.cloud_source == "unknown"

def test_cloud_sar_not_applicable():
    # SAR gets "not_applicable"
    meta = {}
    cond = assess_conditions(meta, "sar")
    assert cond.cloud_status == "not_applicable"
    assert cond.cloud_source == "not_applicable"

def test_missing_stats_safe_handling():
    # Graceful defaults when stats fail to extract
    meta = {}
    cond = assess_conditions(meta, "optical")
    assert cond.nodata_fraction == 0.0
    assert cond.nodata_status == "unknown"
    assert cond.nodata_source == "unknown"
