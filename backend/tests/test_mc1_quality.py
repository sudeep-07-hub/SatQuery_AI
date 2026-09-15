import pytest
from mc1.classifier import assess_quality

def test_quality_missing_stats():
    # TEST 3 - Empty / unreadable raster
    meta = {}
    q = assess_quality(meta, "optical")
    assert q.score == 0.0
    assert q.status == "unknown"
    assert "missing or unreadable" in q.reasons[0]

def test_quality_invalid_fraction():
    # Very few valid pixels
    meta = {
        "image_statistics": {
            "valid_fraction": 0.05,
            "std": 50.0
        }
    }
    q = assess_quality(meta, "optical")
    assert q.score == 0.2
    assert q.status == "poor"
    assert "Very few valid pixels" in q.reasons[0]

def test_quality_constant():
    # TEST 2 - Constant raster
    meta = {
        "image_statistics": {
            "valid_fraction": 1.0,
            "std": 0.0
        }
    }
    q = assess_quality(meta, "optical")
    assert q.score == 0.1
    assert q.status == "poor"
    assert "Low information content" in q.reasons[0]

def test_quality_low_variance():
    # Low dynamic range
    meta = {
        "image_statistics": {
            "valid_fraction": 1.0,
            "std": 1.5
        }
    }
    q = assess_quality(meta, "optical")
    assert q.score == 0.6
    assert q.status == "degraded"
    assert "Low dynamic range" in q.reasons[0]

def test_quality_nominal():
    # TEST 1 - Normal usable raster
    meta = {
        "image_statistics": {
            "valid_fraction": 0.95,
            "std": 25.4
        }
    }
    q = assess_quality(meta, "optical")
    assert q.score == 0.9
    assert q.status == "good"
    assert "Nominal variance" in q.reasons[0]

def test_quality_different_modalities():
    # Ensures the function doesn't crash on different modalities
    meta = {
        "image_statistics": {
            "valid_fraction": 1.0,
            "std": 10.0
        }
    }
    q1 = assess_quality(meta, "sar")
    q2 = assess_quality(meta, "optical")
    assert q1.score == 0.9
    assert q2.score == 0.9
