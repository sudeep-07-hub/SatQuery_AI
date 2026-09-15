import pytest
import torch
import json
import os

from mc4c.query_schema import QueryRepresentation
from mc4c.embedding_schema import TokenRepresentation
from mc4c.fusion_schema import FusedTokenRepresentation
from mc4c.query_fusion import QueryConditionedSpatialFusion
from mc4c.token_schema import TokenSpatialIdentity
from mc4c.token_extractor import CROMATokenExtractor
from mc4c.modality_spatial import TokenRepresentationLayer
from mc4c.query_encoder import QueryEncoder
from qwen.schemas import TaskSpec

def make_dummy_token_rep(modality: str, batch: int = 1, num_tokens: int = 225) -> TokenRepresentation:
    identities = [
        TokenSpatialIdentity(
            original_index=i,
            row=i // 15,
            column=i % 15,
            bounds=(0, 0, 8, 8)
        ) for i in range(num_tokens)
    ]
    return TokenRepresentation(
        modality=modality,
        grid_height=15,
        grid_width=15,
        num_tokens=num_tokens,
        batch_size=batch,
        original_tokens=torch.randn(batch, num_tokens, 768),
        modality_embeddings=torch.randn(batch, num_tokens, 16),
        spatial_embeddings=torch.randn(batch, num_tokens, 2),
        spatial_identities=identities
    )

def test_1_forward_pass_and_shape():
    fusion = QueryConditionedSpatialFusion(fusion_dim=768)
    opt = make_dummy_token_rep("optical")
    sar = make_dummy_token_rep("sar")
    
    query = QueryRepresentation(
        original_query="test",
        primary_task="unknown",
        embedding=torch.randn(1, 384)
    )
    
    fused = fusion(query, opt, sar)
    
    assert isinstance(fused, FusedTokenRepresentation)
    assert fused.fused_tokens.shape == (1, 225, 768)
    assert fused.fusion_dim == 768
    assert fused.num_tokens == 225

def test_2_spatial_identity_preservation():
    fusion = QueryConditionedSpatialFusion()
    opt = make_dummy_token_rep("optical")
    sar = make_dummy_token_rep("sar")
    query = QueryRepresentation(original_query="test", primary_task="unknown", embedding=torch.randn(1, 384))
    
    fused = fusion(query, opt, sar)
    
    # Check token 42
    idx42 = fused.spatial_identities[42]
    assert idx42.original_index == 42
    assert idx42.row == 2
    assert idx42.column == 12

def test_3_query_dependence():
    # MANDATORY: fusion output must change when query changes
    fusion = QueryConditionedSpatialFusion()
    opt = make_dummy_token_rep("optical")
    sar = make_dummy_token_rep("sar")
    
    query_a = QueryRepresentation(original_query="A", primary_task="unknown", embedding=torch.randn(1, 384))
    query_b = QueryRepresentation(original_query="B", primary_task="unknown", embedding=torch.randn(1, 384))
    
    # Must use eval mode to ensure no dropout variability
    fusion.eval()
    with torch.no_grad():
        out_a = fusion(query_a, opt, sar)
        out_b = fusion(query_b, opt, sar)
        
    assert not torch.allclose(out_a.fused_tokens, out_b.fused_tokens, atol=1e-5)

def test_4_modality_dependence():
    # Output must change when image tokens change
    fusion = QueryConditionedSpatialFusion()
    opt = make_dummy_token_rep("optical")
    sar_a = make_dummy_token_rep("sar")
    sar_b = make_dummy_token_rep("sar")
    
    query = QueryRepresentation(original_query="test", primary_task="unknown", embedding=torch.randn(1, 384))
    
    fusion.eval()
    with torch.no_grad():
        out_a = fusion(query, opt, sar_a)
        out_b = fusion(query, opt, sar_b)
        
    assert not torch.allclose(out_a.fused_tokens, out_b.fused_tokens, atol=1e-5)

def test_5_validation_rejections():
    fusion = QueryConditionedSpatialFusion()
    
    query = QueryRepresentation(original_query="test", primary_task="unknown", embedding=torch.randn(1, 384))
    
    # Mismatched tokens
    opt = make_dummy_token_rep("optical", num_tokens=225)
    sar = make_dummy_token_rep("sar", num_tokens=100)
    with pytest.raises(ValueError, match="Token count mismatch"):
        fusion(query, opt, sar)
        
    # Mismatched batch
    opt2 = make_dummy_token_rep("optical", batch=2)
    sar2 = make_dummy_token_rep("sar", batch=1)
    with pytest.raises(ValueError, match="Batch size mismatch"):
        fusion(query, opt2, sar2)

def test_6_selection_preservation():
    fusion = QueryConditionedSpatialFusion()
    opt = make_dummy_token_rep("optical")
    sar = make_dummy_token_rep("sar")
    query = QueryRepresentation(original_query="test", primary_task="unknown", embedding=torch.randn(1, 384))
    
    fused = fusion(query, opt, sar)
    subset = fused.select([42, 224])
    
    assert subset.num_tokens == 2
    assert subset.fused_tokens.shape == (1, 2, 768)
    assert subset.spatial_identities[0].original_index == 42
    assert subset.spatial_identities[1].original_index == 224

def test_7_serialization():
    fusion = QueryConditionedSpatialFusion()
    opt = make_dummy_token_rep("optical")
    sar = make_dummy_token_rep("sar")
    query = QueryRepresentation(original_query="test", primary_task="unknown", embedding=torch.randn(1, 384))
    
    fused = fusion(query, opt, sar)
    dump = fused.model_dump()
    assert "Tensor shape" in dump["fused_tokens"]
    json.dumps(dump) # Should not raise error

def test_8_parameter_count():
    fusion = QueryConditionedSpatialFusion()
    count = sum(p.numel() for p in fusion.parameters() if p.requires_grad)
    # query_proj (384*768 + 768) = 295680
    # opt_proj (796*768 + 768) = 612096
    # sar_proj (796*768 + 768) = 612096
    # gate_net (2304*768 + 768 + 768*2 + 2) = 1771778
    # Total ~ 3.3M parameters
    assert count < 5_000_000 # ensure it is lightweight

CROMA_WEIGHTS = os.path.join(os.path.dirname(__file__), "..", "mc4c", "weights", "CROMA_base.pt")

@pytest.mark.skipif(not os.path.exists(CROMA_WEIGHTS), reason="CROMA weights unavailable")
def test_real_croma_e2e_validation():
    from mc4c.croma import PretrainedCROMA
    
    # 1. Image features
    model = PretrainedCROMA(pretrained_path=CROMA_WEIGHTS, size='base', modality='both')
    model.eval()
    
    opt_input = torch.randn(1, 12, 120, 120)
    sar_input = torch.randn(1, 2, 120, 120)
    
    with torch.no_grad():
        out = model(SAR_images=sar_input, optical_images=opt_input)
        
    extractor = CROMATokenExtractor()
    bundle = extractor.extract(out)
    
    rep_layer = TokenRepresentationLayer()
    opt_rep = rep_layer(bundle.optical)
    sar_rep = rep_layer(bundle.sar)
    
    # 2. Query features
    query_enc = QueryEncoder()
    try:
        query_enc.load()
    except Exception:
        pytest.skip("Query encoder weights unavailable locally")
        
    task_a = TaskSpec(query="Find the building", primary_task="grounding")
    task_b = TaskSpec(query="Where are the roads?", primary_task="grounding")
    
    query_a = query_enc.encode(task_a.query, task_a)
    query_b = query_enc.encode(task_b.query, task_b)
    
    # 3. Fusion
    fusion = QueryConditionedSpatialFusion()
    fusion.eval()
    
    with torch.no_grad():
        fused_a = fusion(query_a, opt_rep, sar_rep)
        fused_b = fusion(query_b, opt_rep, sar_rep)
        
    assert fused_a.fused_tokens.shape == (1, 225, 768)
    assert fused_b.fused_tokens.shape == (1, 225, 768)
    
    # Query dependence
    assert not torch.allclose(fused_a.fused_tokens, fused_b.fused_tokens, atol=1e-5)
    
    # Spatial preservation
    assert fused_a.spatial_identities[42].original_index == 42
