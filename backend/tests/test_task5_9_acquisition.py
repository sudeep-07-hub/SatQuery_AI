import os
import torch
import pytest
from backend.datasets.bigearthnet.pairing import validate_pair
from backend.datasets.bigearthnet.manifest import DatasetManifest

def test_validate_pair_success():
    visual = {
        "sample_id": "123",
        "joint_encodings": torch.randn(1, 225, 768)
    }
    text = {
        "sample_id": "123",
        "pooled_embedding": torch.randn(2560)
    }
    is_valid, reason = validate_pair(visual, text)
    assert is_valid
    assert reason == "valid"

def test_validate_pair_id_mismatch():
    visual = {
        "sample_id": "123",
        "joint_encodings": torch.randn(1, 225, 768)
    }
    text = {
        "sample_id": "456",
        "pooled_embedding": torch.randn(2560)
    }
    is_valid, reason = validate_pair(visual, text)
    assert not is_valid
    assert "ID_mismatch" in reason

def test_validate_pair_malformed_shape():
    visual = {
        "sample_id": "123",
        "joint_encodings": torch.randn(1, 225, 768)
    }
    text = {
        "sample_id": "123",
        "pooled_embedding": torch.randn(512) # Wrong shape
    }
    is_valid, reason = validate_pair(visual, text)
    assert not is_valid
    assert "malformed_feature" in reason

def test_validate_pair_nan():
    visual = {
        "sample_id": "123",
        "joint_encodings": torch.full((1, 225, 768), float('nan'))
    }
    text = {
        "sample_id": "123",
        "pooled_embedding": torch.randn(2560)
    }
    is_valid, reason = validate_pair(visual, text)
    assert not is_valid
    assert "NaN/Inf" in reason

def test_manifest_duplicate_detection(tmp_path):
    manifest = DatasetManifest(str(tmp_path / "cache"), str(tmp_path / "meta"))
    
    manifest.add_valid_pair("123", "v.pt", "t.pt")
    assert manifest.stats["valid_pairs"] == 1
    
    manifest.add_valid_pair("123", "v.pt", "t.pt")
    assert manifest.stats["valid_pairs"] == 1 # unchanged
    assert manifest.stats["duplicate_samples"] == 1
    assert manifest.stats["rejected_samples"] == 1
    
def test_manifest_rejection_tracking(tmp_path):
    manifest = DatasetManifest(str(tmp_path / "cache"), str(tmp_path / "meta"))
    manifest.add_rejection("456", "Qwen3_failure: OOM", "text_extraction")
    assert manifest.stats["Qwen3_failures"] == 1
    assert manifest.stats["rejected_samples"] == 1
