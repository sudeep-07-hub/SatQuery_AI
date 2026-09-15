import os
import tempfile
import torch
import pytest

from backend.adaptation.config import AdaptationConfig
from backend.adaptation.dataset import BigEarthNetAdaptationDataset
from backend.adaptation.projections import TextProjectionHead, VisualProjectionHead
from backend.adaptation.contrastive import SymmetricInfoNCE
from backend.adaptation.checkpoint import save_checkpoint, load_checkpoint

def test_dataset_pairing(tmp_path):
    v_dir = tmp_path / "features"
    t_dir = tmp_path / "text_features"
    v_dir.mkdir()
    t_dir.mkdir()
    
    # 1 valid pair (123)
    torch.save({"sample_id": "123", "joint_encodings": torch.randn(1, 225, 768)}, v_dir / "sample_123.pt")
    torch.save({"sample_id": "123", "pooled_embedding": torch.randn(2560)}, t_dir / "qwen3_text_123.pt")
    
    # 1 missing text (456)
    torch.save({"sample_id": "456", "joint_encodings": torch.randn(1, 225, 768)}, v_dir / "sample_456.pt")
    
    # 1 missing visual (789)
    torch.save({"sample_id": "789", "pooled_embedding": torch.randn(2560)}, t_dir / "qwen3_text_789.pt")
    
    dataset = BigEarthNetAdaptationDataset(str(v_dir), str(t_dir))
    
    assert len(dataset) == 1
    assert dataset[0]["sample_id"] == "123"
    assert dataset.stats["paired_records"] == 1
    assert dataset.stats["missing_text"] == 1
    assert dataset.stats["missing_visual"] == 1

def test_projections_and_normalization():
    t_head = TextProjectionHead(2560, 512)
    v_head = VisualProjectionHead(768, 512)
    
    t_input = torch.randn(2, 2560)
    v_input = torch.randn(2, 225, 768)
    
    t_out = t_head(t_input)
    v_out = v_head(v_input)
    
    # Check dimensionality
    assert t_out.shape == (2, 512)
    assert v_out.shape == (2, 512)
    
    # Check L2 normalization
    assert torch.allclose(torch.linalg.vector_norm(t_out, dim=-1), torch.ones(2), atol=1e-5)
    assert torch.allclose(torch.linalg.vector_norm(v_out, dim=-1), torch.ones(2), atol=1e-5)
    
    # Check finite
    assert torch.isfinite(t_out).all()
    assert torch.isfinite(v_out).all()

def test_contrastive_loss():
    loss_fn = SymmetricInfoNCE(temperature=0.07)
    
    # Batch size 2
    img_emb = torch.nn.functional.normalize(torch.randn(2, 512), dim=-1)
    txt_emb = torch.nn.functional.normalize(torch.randn(2, 512), dim=-1)
    
    loss = loss_fn(img_emb, txt_emb)
    assert torch.isfinite(loss)
    assert loss.item() >= 0
    
    # Batch size 1 - should not crash, returns 0.0 mathematically but cross entropy returns ~0.0
    img_emb_1 = torch.nn.functional.normalize(torch.randn(1, 512), dim=-1)
    txt_emb_1 = torch.nn.functional.normalize(torch.randn(1, 512), dim=-1)
    
    loss_1 = loss_fn(img_emb_1, txt_emb_1)
    assert torch.isfinite(loss_1)
    assert torch.allclose(loss_1, torch.tensor(0.0), atol=1e-5)

def test_gradient_propagation_and_optimizer():
    t_head = TextProjectionHead(2560, 512)
    v_head = VisualProjectionHead(768, 512)
    loss_fn = SymmetricInfoNCE()
    
    optimizer = torch.optim.SGD(list(t_head.parameters()) + list(v_head.parameters()), lr=0.1)
    
    t_input = torch.randn(2, 2560)
    v_input = torch.randn(2, 225, 768)
    
    # Capture initial weights
    t_weight_initial = t_head.projection.weight.clone()
    
    optimizer.zero_grad()
    loss = loss_fn(v_head(v_input), t_head(t_input))
    loss.backward()
    
    # Verify gradients exist
    assert t_head.projection.weight.grad is not None
    assert v_head.projection.weight.grad is not None
    
    optimizer.step()
    
    # Verify weights changed
    assert not torch.allclose(t_head.projection.weight, t_weight_initial)

def test_checkpoint_roundtrip(tmp_path):
    t_head = TextProjectionHead()
    v_head = VisualProjectionHead()
    optimizer = torch.optim.AdamW(t_head.parameters())
    
    config = AdaptationConfig(embedding_dim=512, temperature=0.1)
    ckpt_path = str(tmp_path / "model.pt")
    
    save_checkpoint(ckpt_path, t_head, v_head, optimizer, config, {}, 1, 0.5)
    
    # Modify weights to ensure loading overwrites them
    t_head.projection.weight.data.fill_(0.0)
    
    load_checkpoint(ckpt_path, t_head, v_head)
    
    # Weight should not be 0 anymore (unless initialized to 0, which it isn't)
    assert not torch.allclose(t_head.projection.weight, torch.zeros_like(t_head.projection.weight))
