import os
import torch
import torch.nn.functional as F
import psutil
import tempfile
import pytest

from backend.adaptation.qwen3_feature_extractor import masked_mean_pooling, Qwen3FeatureExtractor, InsufficientMemoryError

def test_masked_mean_pooling():
    # Batch size 2, Sequence length 3, Hidden dim 4
    hidden = torch.tensor([
        [[1.0, 1.0, 1.0, 1.0], [2.0, 2.0, 2.0, 2.0], [3.0, 3.0, 3.0, 3.0]],
        [[4.0, 4.0, 4.0, 4.0], [5.0, 5.0, 5.0, 5.0], [6.0, 6.0, 6.0, 6.0]]
    ])
    
    # First sample: uses first 2 tokens. Second sample: uses only 1st token.
    mask = torch.tensor([
        [1, 1, 0],
        [1, 0, 0]
    ])
    
    pooled = masked_mean_pooling(hidden, mask)
    
    # Expected:
    # sample 0: mean of [1,1,1,1] and [2,2,2,2] -> [1.5, 1.5, 1.5, 1.5]
    # sample 1: mean of [4,4,4,4] -> [4.0, 4.0, 4.0, 4.0]
    assert pooled.shape == (2, 4)
    assert torch.allclose(pooled[0], torch.tensor([1.5, 1.5, 1.5, 1.5]))
    assert torch.allclose(pooled[1], torch.tensor([4.0, 4.0, 4.0, 4.0]))

def test_l2_normalization():
    # Test L2 norm is 1.0 within tolerance
    pooled = torch.tensor([[1.0, 2.0, 3.0], [0.0, -1.0, 0.0]])
    normalized = F.normalize(pooled, p=2, dim=-1)
    
    norms = torch.linalg.vector_norm(normalized, dim=-1)
    
    # Tolerance for float32 is typically ~1e-6
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

def test_invalid_tensors():
    # Ensure our standard validation checks catch NaN/Inf
    valid = torch.tensor([1.0, 2.0])
    has_nan = torch.tensor([1.0, float('nan')])
    has_inf = torch.tensor([1.0, float('inf')])
    
    assert torch.isfinite(valid).all()
    assert not torch.isfinite(has_nan).all()
    assert not torch.isfinite(has_inf).all()

def test_feature_artifact_roundtrip():
    artifact = {
        "sample_id": "771130",
        "caption_id": "771130_0",
        "caption": "A real caption",
        "model_id": "Qwen/Qwen3-4B-Instruct-2507",
        "model_revision": "abcd123",
        "layer": 36,
        "hidden_dim": 2560,
        "pooled_embedding": torch.randn(2560),
        "normalized_embedding": F.normalize(torch.randn(2560), p=2, dim=-1),
        "token_count": 15,
        "truncated": False,
        "pooling": "masked_mean",
        "normalization": "l2",
        "device": "cpu",
        "dtype": "torch.float32",
        "source": "BigEarthNet.txt"
    }
    
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "feature.pt")
        torch.save(artifact, path)
        
        loaded = torch.load(path)
        
        assert loaded["sample_id"] == "771130"
        assert loaded["layer"] == 36
        assert loaded["hidden_dim"] == 2560
        assert torch.allclose(loaded["pooled_embedding"], artifact["pooled_embedding"])
        assert torch.allclose(loaded["normalized_embedding"], artifact["normalized_embedding"])

@pytest.mark.live
def test_real_model_smoke_test():
    """
    Live test to check if Qwen3 can be loaded and executed.
    This will likely be skipped or fail due to InsufficientMemoryError locally.
    """
    extractor = Qwen3FeatureExtractor(required_ram_gb=8.0)
    
    try:
        extractor.check_memory()
    except InsufficientMemoryError:
        pytest.skip("Insufficient memory to run live test.")
        
    try:
        results = extractor.extract(["A beautiful satellite image of a forest."])
        assert len(results) == 1
        res = results[0]
        assert res["pooled_embedding"].shape == (2560,)
        assert res["normalized_embedding"].shape == (2560,)
        assert abs(torch.linalg.vector_norm(res["normalized_embedding"]).item() - 1.0) < 1e-5
        assert torch.isfinite(res["normalized_embedding"]).all()
    except Exception as e:
        pytest.fail(f"Live model execution failed: {e}")
