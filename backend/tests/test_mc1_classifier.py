import pytest
from mc1.classifier import EvidenceBasedClassifier

def test_classifier_optical_metadata():
    # TEST 1 & 3 - Optical observation with explicit metadata
    classifier = EvidenceBasedClassifier()
    meta = {
        "tags": {"SENSOR": "Sentinel-2"}
    }
    modality, sensor, conf, source = classifier.classify(meta, "unknown.tif")
    assert modality == "optical"
    assert sensor == "Sentinel-2"
    assert conf == "high"
    assert source == "metadata"

def test_classifier_sar_metadata():
    # TEST 2 & 3 - SAR observation with explicit metadata
    classifier = EvidenceBasedClassifier()
    meta = {
        "tags": {"platform": "Sentinel-1A"}
    }
    modality, sensor, conf, source = classifier.classify(meta, "unknown.tif")
    assert modality == "sar"
    assert sensor == "Sentinel-1"
    assert conf == "high"
    assert source == "metadata"

def test_classifier_unknown_ambiguous():
    # TEST 4 - Unknown / ambiguous observation
    classifier = EvidenceBasedClassifier()
    meta = {
        "band_count": 1,
        "dtypes": ["uint8"]
    }
    modality, sensor, conf, source = classifier.classify(meta, "image.tif")
    assert modality == "unknown"
    assert sensor == "unknown"
    assert conf == "unknown"
    assert source == "unknown"

def test_classifier_filename_heuristic():
    # TEST 5 - Filename heuristic
    classifier = EvidenceBasedClassifier()
    meta = {}
    modality, sensor, conf, source = classifier.classify(meta, "LC08_L1TP_123456_20210101_20210101_02_T1.tif")
    assert modality == "optical"
    assert sensor == "Landsat 8"
    assert conf == "low"
    assert source == "filename_heuristic"

def test_classifier_conflicting_evidence():
    # TEST 6 - Conflicting evidence (Metadata should beat raster properties and filename)
    classifier = EvidenceBasedClassifier()
    meta = {
        "tags": {"sensor": "Sentinel-1"}, # Metadata says SAR
        "colorinterp": ["red", "green", "blue"], # Raster says optical
    }
    # Filename says optical
    modality, sensor, conf, source = classifier.classify(meta, "S2A_image.tif")
    
    assert modality == "sar"
    assert sensor == "Sentinel-1"
    assert conf == "high"
    assert source == "metadata"
