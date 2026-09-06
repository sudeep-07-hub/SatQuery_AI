import os
import pytest
from mc1.format_validator import validate_format
from mc1.metadata_extractor import extract_metadata
from mc1.spatial_analyzer import calculate_overlap, get_footprint
from mc1.temporal_analyzer import determine_temporal_relationship

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), 'fixtures')

def test_format_validator():
    # Test valid extension but invalid magic bytes
    is_valid, err, fmt = validate_format("test.tif", b"invalid_bytes_here")
    assert not is_valid
    assert fmt == "unknown"

    # Test valid PNG
    with open(os.path.join(FIXTURES_DIR, "test_plain.png"), "rb") as f:
        is_valid, err, fmt = validate_format("test_plain.png", f.read())
        assert is_valid
        assert fmt == "png"

def test_metadata_extractor():
    with open(os.path.join(FIXTURES_DIR, "test_geo1.tif"), "rb") as f:
        meta = extract_metadata(f.read(), "tiff")
        assert meta["crs"] == "EPSG:32643"
        assert meta["band_count"] == 4
        assert meta["gsd_m"] == 10.0

    with open(os.path.join(FIXTURES_DIR, "test_plain.png"), "rb") as f:
        meta = extract_metadata(f.read(), "png")
        assert meta["crs"] is None
        assert meta["band_count"] == 3

def test_spatial_analyzer():
    with open(os.path.join(FIXTURES_DIR, "test_geo1.tif"), "rb") as f1, \
         open(os.path.join(FIXTURES_DIR, "test_geo2.tif"), "rb") as f2:
        meta1 = extract_metadata(f1.read(), "tiff")
        meta2 = extract_metadata(f2.read(), "tiff")
        
        foot1 = get_footprint(meta1)
        foot2 = get_footprint(meta2)
        
        overlap = calculate_overlap(foot1, foot2)
        # footprint 1: 300000 to 301000, 3999000 to 4000000 (area = 1,000,000)
        # footprint 2: 300500 to 301500, 3998500 to 3999500 (area = 1,000,000)
        # Intersection: 300500 to 301000 (500), 3999000 to 3999500 (500) -> area 250,000
        # Union: 1,000,000 + 1,000,000 - 250,000 = 1,750,000
        # IoU: 250,000 / 1,750,000 = 1/7 ~= 0.1428
        assert overlap is not None
        assert 0.14 < overlap < 0.15

def test_temporal_analyzer():
    # Same date
    meta1 = {"acquisition_date": "2024-11-15 10:00:00"}
    meta2 = {"acquisition_date": "2024-11-15 12:00:00"}
    assert determine_temporal_relationship(meta1, meta2, "optical", "optical") == "same_date"
    
    # Cross modal
    assert determine_temporal_relationship(meta1, meta2, "optical", "sar") == "cross_modal"
    
    # Multi temporal
    meta3 = {"acquisition_date": "2024-11-18 10:00:00"}
    assert determine_temporal_relationship(meta1, meta3, "optical", "optical") == "multi_temporal"
