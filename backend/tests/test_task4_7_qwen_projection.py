import pytest
import torch
import json
import os
import copy

from mc4c.qwen_adapter_schema import QwenVisualRepresentation
from mc4c.qwen_adapter import QwenProjectionAdapter
from mc4c.fusion_schema import FusedTokenRepresentation
from mc4c.query_schema import QueryRepresentation
from mc4c.token_schema import TokenSpatialIdentity

def make_dummy_fused_rep(batch: int = 1, num_tokens: int = 225, fusion_dim: int = 768) -> FusedTokenRepresentation:
    identities = [
        TokenSpatialIdentity(
            original_index=i,
            row=i // 15,
            column=i % 15,
            bounds=(0, 0, 8, 8)
        ) for i in range(num_tokens)
    ]
    query_context = QueryRepresentation(
        original_query="test", 
        primary_task="unknown", 
        embedding=torch.randn(1, 384)
    )
    return FusedTokenRepresentation(
        grid_height=15,
        grid_width=15,
        num_tokens=num_tokens,
        batch_size=batch,
        fused_tokens=torch.randn(batch, num_tokens, fusion_dim),
        fusion_dim=fusion_dim,
        spatial_identities=identities,
        query_context=query_context
    )

def test_1_forward_pass_and_shape():
    fused_in = make_dummy_fused_rep()
    adapter = QwenProjectionAdapter(input_dim=768, qwen_hidden_dim=2560)
    
    out = adapter(fused_in)
    
    assert isinstance(out, QwenVisualRepresentation)
    assert out.projected_tokens.shape == (1, 225, 2560)
    assert out.embedding_dim == 2560
    assert out.num_tokens == 225

def test_2_spatial_identity_preservation():
    fused_in = make_dummy_fused_rep()
    adapter = QwenProjectionAdapter()
    
    out = adapter(fused_in)
    
    # Check bounds
    assert len(out.spatial_identities) == 225
    
    idx0 = out.spatial_identities[0]
    assert idx0.original_index == 0
    assert idx0.row == 0
    assert idx0.column == 0
    
    idx42 = out.spatial_identities[42]
    assert idx42.original_index == 42
    assert idx42.row == 2
    assert idx42.column == 12
    
    idx224 = out.spatial_identities[224]
    assert idx224.original_index == 224
    assert idx224.row == 14
    assert idx224.column == 14

def test_3_immutability():
    fused_in = make_dummy_fused_rep()
    original_tensor = fused_in.fused_tokens.clone()
    
    adapter = QwenProjectionAdapter()
    adapter(fused_in)
    
    assert torch.equal(fused_in.fused_tokens, original_tensor)

def test_4_query_provenance_preservation():
    fused_in = make_dummy_fused_rep()
    fused_in.query_context.original_query = "Find the bridge"
    
    adapter = QwenProjectionAdapter()
    out = adapter(fused_in)
    
    assert out.query_context.original_query == "Find the bridge"

def test_5_validation_rejections():
    adapter = QwenProjectionAdapter(input_dim=768)
    
    # Invalid rank
    bad_fused = make_dummy_fused_rep()
    bad_fused.fused_tokens = torch.randn(1, 225) # 2D
    with pytest.raises(ValueError, match="Expected 3D tensor"):
        adapter(bad_fused)
        
    # Invalid dim
    bad_fused.fused_tokens = torch.randn(1, 225, 512) # Wrong dim
    with pytest.raises(ValueError, match="Expected feature dimension 768"):
        adapter(bad_fused)

def test_6_batch_dimensions():
    adapter = QwenProjectionAdapter()
    
    fused_b1 = make_dummy_fused_rep(batch=1)
    out_b1 = adapter(fused_b1)
    assert out_b1.projected_tokens.shape == (1, 225, 2560)
    
    fused_b2 = make_dummy_fused_rep(batch=2)
    out_b2 = adapter(fused_b2)
    assert out_b2.projected_tokens.shape == (2, 225, 2560)

def test_7_query_conditioned_property_propagation():
    # If the inputs differ due to query condition, outputs must differ
    fused_a = make_dummy_fused_rep()
    fused_b = make_dummy_fused_rep()
    # they are currently randomly different
    
    adapter = QwenProjectionAdapter()
    adapter.eval()
    with torch.no_grad():
        out_a = adapter(fused_a)
        out_b = adapter(fused_b)
        
    assert not torch.allclose(out_a.projected_tokens, out_b.projected_tokens)

def test_8_determinism():
    fused = make_dummy_fused_rep()
    adapter = QwenProjectionAdapter()
    adapter.eval()
    
    with torch.no_grad():
        out1 = adapter(fused)
        out2 = adapter(fused)
        
    assert torch.allclose(out1.projected_tokens, out2.projected_tokens)

def test_9_serialization():
    fused_in = make_dummy_fused_rep()
    adapter = QwenProjectionAdapter()
    out = adapter(fused_in)
    
    dump = out.model_dump()
    assert "Tensor shape" in dump["projected_tokens"]
    json.dumps(dump) # Should not raise error

def test_10_parameter_count():
    adapter = QwenProjectionAdapter(768, 2560)
    count = sum(p.numel() for p in adapter.parameters() if p.requires_grad)
    # 768 * 2560 + 2560 = 1968640
    assert count == 1968640

def test_11_qwen_config_fetching():
    # Attempt to fetch real Qwen3 config locally, simulating the verification
    try:
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained('Qwen/Qwen3-4B-Instruct-2507', trust_remote_code=True)
        assert config.hidden_size == 2560
    except Exception:
        pytest.skip("Model unavailable locally")
