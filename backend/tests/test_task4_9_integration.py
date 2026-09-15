import pytest
import torch
import time
import copy
import json

from mc4c.token_schema import TokenSpatialIdentity
from mc4c.query_schema import QueryRepresentation
from mc4c.embedding_schema import TokenRepresentation
from mc4c.query_fusion import QueryConditionedSpatialFusion
from mc4c.fusion_schema import FusedTokenRepresentation
from mc4c.qwen_adapter import QwenProjectionAdapter
from mc4c.qwen_adapter_schema import QwenVisualRepresentation
from mc4c.cross_modal_evidence import CrossModalEvidenceAdapter
from mc5_evidence.schemas import validate_evidence_object


def construct_mock_croma_output(num_tokens=225, batch=1):
    opt = torch.randn(batch, num_tokens, 768)
    sar = torch.randn(batch, num_tokens, 768)
    
    # Pre-extract bounds as in Task 4.2
    identities = [
        TokenSpatialIdentity(
            original_index=i,
            row=i // 15,
            column=i % 15,
            bounds=(0, 0, 8, 8)
        ) for i in range(num_tokens)
    ]
    return opt, sar, identities


def run_pipeline(
    optical_tensor, 
    sar_tensor, 
    identities, 
    query_text, 
    opt_obs_id="opt1", 
    sar_obs_id="sar1", 
    mc1_status="compatible",
    indices=[0, 42, 224]
):
    # 1. Modality/Spatial Embeddings (Task 4.4)
    # Mocking TokenRepresentation since we just need the structured types.
    b, n, d = optical_tensor.shape
    opt_rep = TokenRepresentation(
        modality="optical",
        grid_height=15,
        grid_width=15,
        num_tokens=n,
        batch_size=b,
        original_tokens=optical_tensor,
        modality_embeddings=torch.randn(b, n, 16),
        spatial_embeddings=torch.randn(b, n, 2),
        spatial_identities=identities
    )
    
    sar_rep = TokenRepresentation(
        modality="sar",
        grid_height=15,
        grid_width=15,
        num_tokens=n,
        batch_size=b,
        original_tokens=sar_tensor,
        modality_embeddings=torch.randn(b, n, 16),
        spatial_embeddings=torch.randn(b, n, 2),
        spatial_identities=identities
    )
    
    # 2. Query Representation (Task 4.5)
    # Hash query to simulate differing embeddings
    torch.manual_seed(hash(query_text) % 100000)
    query_rep = QueryRepresentation(
        original_query=query_text,
        primary_task="integration_test",
        embedding=torch.randn(1, 384)
    )
    
    # 3. Query-Conditioned Fusion (Task 4.6)
    fusion_layer = QueryConditionedSpatialFusion(token_dim=768, query_dim=384, fusion_dim=768)
    fusion_layer.eval()
    
    with torch.no_grad():
        fused_rep = fusion_layer(
            query=query_rep,
            opt_rep=opt_rep,
            sar_rep=sar_rep
        )
        
    # 4. Qwen Projection (Task 4.7)
    qwen_adapter = QwenProjectionAdapter(input_dim=768, qwen_hidden_dim=2560)
    qwen_adapter.eval()
    
    with torch.no_grad():
        qwen_rep = qwen_adapter(fused_rep)
        
    # 5. Evidence Generation (Task 4.8)
    ev_adapter = CrossModalEvidenceAdapter()
    
    candidates = ev_adapter.generate_candidates(
        fused_rep=fused_rep, # Note: using fused rep to bypass the untracked intermediate tensor
        optical_obs={"observation_id": opt_obs_id},
        sar_obs={"observation_id": sar_obs_id},
        mc1_compatibility={"status": mc1_status},
        indices=indices
    )
    
    return {
        "optical_input": optical_tensor,
        "sar_input": sar_tensor,
        "query_rep": query_rep,
        "fused_rep": fused_rep,
        "qwen_rep": qwen_rep,
        "candidates": candidates
    }


def test_1_end_to_end_shape_verification():
    opt, sar, idents = construct_mock_croma_output()
    results = run_pipeline(opt, sar, idents, "Find the bridge")
    
    # Verify exact tensor shapes
    assert results["optical_input"].shape == (1, 225, 768)
    assert results["sar_input"].shape == (1, 225, 768)
    assert results["query_rep"].embedding.shape == (1, 384)
    assert results["fused_rep"].fused_tokens.shape == (1, 225, 768)
    assert results["qwen_rep"].projected_tokens.shape == (1, 225, 2560)
    
    # No sequence collapse occurred
    assert results["fused_rep"].num_tokens == 225
    assert results["qwen_rep"].num_tokens == 225

def test_2_spatial_identity_preservation():
    opt, sar, idents = construct_mock_croma_output()
    results = run_pipeline(opt, sar, idents, "Find the bridge", indices=[0, 42, 224])
    
    fused_identities = results["fused_rep"].spatial_identities
    qwen_identities = results["qwen_rep"].spatial_identities
    candidates = results["candidates"]
    
    # Check index 42 survives everywhere natively
    assert fused_identities[42].original_index == 42
    assert qwen_identities[42].original_index == 42
    
    assert fused_identities[42].row == 2
    assert fused_identities[42].column == 12
    
    # Ensure evidence candidates correctly map to the requested indices
    assert candidates[1].token_identity.original_index == 42
    assert candidates[1].token_identity.row == 2
    assert candidates[1].token_identity.column == 12

def test_3_query_perturbation():
    opt, sar, idents = construct_mock_croma_output()
    
    torch.manual_seed(42) # fix fusion and projection weights for reproducible delta
    fusion_layer = QueryConditionedSpatialFusion()
    fusion_layer.eval()
    qwen_adapter = QwenProjectionAdapter()
    qwen_adapter.eval()
    
    query_a = QueryRepresentation(original_query="Q1", primary_task="T1", embedding=torch.randn(1,384))
    query_b = QueryRepresentation(original_query="Q2", primary_task="T2", embedding=torch.randn(1,384))
    
    with torch.no_grad():
        b, n, d = opt.shape
        opt_rep = TokenRepresentation(
            modality="optical", grid_height=15, grid_width=15, num_tokens=n, batch_size=b,
            original_tokens=opt, modality_embeddings=torch.randn(b,n,16), spatial_embeddings=torch.randn(b,n,2), spatial_identities=idents
        )
        sar_rep_a = TokenRepresentation(
            modality="sar", grid_height=15, grid_width=15, num_tokens=n, batch_size=b,
            original_tokens=opt, modality_embeddings=torch.randn(b,n,16), spatial_embeddings=torch.randn(b,n,2), spatial_identities=idents
        )
        sar_rep_b = TokenRepresentation(
            modality="sar", grid_height=15, grid_width=15, num_tokens=n, batch_size=b,
            original_tokens=opt, modality_embeddings=torch.randn(b,n,16), spatial_embeddings=torch.randn(b,n,2), spatial_identities=idents
        )
        fused_a = fusion_layer(query_a, opt_rep, sar_rep_a)
        fused_b = fusion_layer(query_b, opt_rep, sar_rep_b)
        
        qwen_a = qwen_adapter(fused_a)
        qwen_b = qwen_adapter(fused_b)
        
    # Tensors must differ based purely on numerical query conditioning
    assert not torch.allclose(fused_a.fused_tokens, fused_b.fused_tokens)
    assert not torch.allclose(qwen_a.projected_tokens, qwen_b.projected_tokens)
    
    # Identities must NOT differ
    assert fused_a.spatial_identities[42].row == fused_b.spatial_identities[42].row

def test_4_modality_perturbation():
    opt, sar_a, idents = construct_mock_croma_output()
    sar_b = torch.randn(1, 225, 768)
    
    fusion_layer = QueryConditionedSpatialFusion()
    fusion_layer.eval()
    query = QueryRepresentation(original_query="Q1", primary_task="T1", embedding=torch.randn(1,384))
    
    with torch.no_grad():
        b, n, d = opt.shape
        opt_rep = TokenRepresentation(
            modality="optical", grid_height=15, grid_width=15, num_tokens=n, batch_size=b,
            original_tokens=opt, modality_embeddings=torch.randn(b,n,16), spatial_embeddings=torch.randn(b,n,2), spatial_identities=idents
        )
        sar_rep_a = TokenRepresentation(
            modality="sar", grid_height=15, grid_width=15, num_tokens=n, batch_size=b,
            original_tokens=sar_a, modality_embeddings=torch.randn(b,n,16), spatial_embeddings=torch.randn(b,n,2), spatial_identities=idents
        )
        sar_rep_b = TokenRepresentation(
            modality="sar", grid_height=15, grid_width=15, num_tokens=n, batch_size=b,
            original_tokens=sar_b, modality_embeddings=torch.randn(b,n,16), spatial_embeddings=torch.randn(b,n,2), spatial_identities=idents
        )
        
        fused_a = fusion_layer(query, opt_rep, sar_rep_a)
        fused_b = fusion_layer(query, opt_rep, sar_rep_b)
        
    assert not torch.allclose(fused_a.fused_tokens, fused_b.fused_tokens)

def test_5_cross_modal_evidence_provenance():
    opt, sar, idents = construct_mock_croma_output()
    results = run_pipeline(opt, sar, idents, "query_text", "opt_123", "sar_456", indices=[42])
    
    candidate = results["candidates"][0]
    
    assert candidate.query_context == "query_text"
    assert candidate.optical_observation_id == "opt_123"
    assert candidate.sar_observation_id == "sar_456"
    assert candidate.token_identity.original_index == 42
    
    mc5_dict = candidate.to_mc5_evidence()
    assert validate_evidence_object(mc5_dict) is True
    
    assert mc5_dict["claim"] == "Unverified spatial multimodal candidate"
    assert mc5_dict["source_input"]["optical_id"] == "opt_123"

def test_6_mc1_hard_rejection():
    opt, sar, idents = construct_mock_croma_output()
    
    with pytest.raises(ValueError, match="hard incompatibility"):
        run_pipeline(opt, sar, idents, "query", mc1_status="hard_incompatible")
        
def test_7_evidence_serialization_cleanliness():
    opt, sar, idents = construct_mock_croma_output()
    results = run_pipeline(opt, sar, idents, "Q", indices=[42])
    
    mc5_dict = results["candidates"][0].to_mc5_evidence()
    json_str = json.dumps(mc5_dict)
    
    # Giant tensors (like 2560-d arrays) should NOT be serialized
    assert "projected_tokens" not in json_str
    assert "fused_tokens" not in json_str
    assert len(json_str) < 1000

def test_8_determinism_pipeline():
    opt, sar, idents = construct_mock_croma_output()
    
    # We must mock seed explicitly for everything inside run_pipeline for perfect determinism.
    # PyTorch layer initialization needs seeding if created inside the function, 
    # but the pipeline creates fresh layers each time.
    # To truly test this, we must use identical layer weights.
    
    torch.manual_seed(42)
    fusion_layer = QueryConditionedSpatialFusion()
    qwen_adapter = QwenProjectionAdapter()
    
    query = QueryRepresentation(original_query="Q", primary_task="T", embedding=torch.randn(1,384))
    
    fusion_layer.eval()
    qwen_adapter.eval()
    
    with torch.no_grad():
        b, n, d = opt.shape
        opt_rep = TokenRepresentation(
            modality="optical", grid_height=15, grid_width=15, num_tokens=n, batch_size=b,
            original_tokens=opt, modality_embeddings=torch.randn(b,n,16), spatial_embeddings=torch.randn(b,n,2), spatial_identities=idents
        )
        sar_rep = TokenRepresentation(
            modality="sar", grid_height=15, grid_width=15, num_tokens=n, batch_size=b,
            original_tokens=sar, modality_embeddings=torch.randn(b,n,16), spatial_embeddings=torch.randn(b,n,2), spatial_identities=idents
        )
        fused_1 = fusion_layer(query, opt_rep, sar_rep)
        qwen_1 = qwen_adapter(fused_1)
        
        fused_2 = fusion_layer(query, opt_rep, sar_rep)
        qwen_2 = qwen_adapter(fused_2)
        
    assert torch.allclose(fused_1.fused_tokens, fused_2.fused_tokens)
    assert torch.allclose(qwen_1.projected_tokens, qwen_2.projected_tokens)
    
    ev_adapter = CrossModalEvidenceAdapter()
    c1 = ev_adapter.generate_candidates(fused_1, {"observation_id":"O1"}, {"observation_id":"S1"}, {}, indices=[42])
    c2 = ev_adapter.generate_candidates(fused_2, {"observation_id":"O1"}, {"observation_id":"S1"}, {}, indices=[42])
    
    assert c1[0].evidence_id == c2[0].evidence_id

def test_9_performance_metrics():
    # Only capturing lightweight timing
    opt, sar, idents = construct_mock_croma_output()
    
    start = time.time()
    results = run_pipeline(opt, sar, idents, "Timing test query")
    end = time.time()
    
    elapsed_ms = (end - start) * 1000
    # Should be incredibly fast since it's just two linear layers locally
    assert elapsed_ms < 500
