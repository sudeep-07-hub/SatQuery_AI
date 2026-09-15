import pytest
import torch
import copy
from typing import Dict, Any

from mc4c.cross_modal_evidence import CrossModalEvidenceAdapter, CrossModalEvidenceCandidate
from mc4c.fusion_schema import FusedTokenRepresentation
from mc4c.token_schema import TokenSpatialIdentity
from mc4c.query_schema import QueryRepresentation
from mc5_evidence.schemas import validate_evidence_object

def make_dummy_fused_rep(num_tokens: int = 225) -> FusedTokenRepresentation:
    identities = [
        TokenSpatialIdentity(
            original_index=i,
            row=i // 15,
            column=i % 15,
            bounds=(0, 0, 8, 8)
        ) for i in range(num_tokens)
    ]
    query_context = QueryRepresentation(
        original_query="Find the building", 
        primary_task="unknown", 
        embedding=torch.randn(1, 384)
    )
    return FusedTokenRepresentation(
        grid_height=15,
        grid_width=15,
        num_tokens=num_tokens,
        batch_size=1,
        fused_tokens=torch.randn(1, num_tokens, 768),
        fusion_dim=768,
        spatial_identities=identities,
        query_context=query_context
    )

def test_1_adapter_construction():
    adapter = CrossModalEvidenceAdapter()
    assert adapter is not None

def test_2_valid_pair_accepted():
    fused_rep = make_dummy_fused_rep()
    opt_obs = {"observation_id": "opt123"}
    sar_obs = {"observation_id": "sar456"}
    mc1_compat = {"status": "compatible"}
    
    adapter = CrossModalEvidenceAdapter()
    candidates = adapter.generate_candidates(fused_rep, opt_obs, sar_obs, mc1_compat, indices=[0, 42])
    
    assert len(candidates) == 2
    assert candidates[0].optical_observation_id == "opt123"
    assert candidates[0].sar_observation_id == "sar456"

def test_3_incompatible_pair_rejected():
    fused_rep = make_dummy_fused_rep()
    opt_obs = {"observation_id": "opt123"}
    sar_obs = {"observation_id": "sar456"}
    mc1_compat = {"status": "hard_incompatible"}
    
    adapter = CrossModalEvidenceAdapter()
    with pytest.raises(ValueError, match="hard incompatibility"):
        adapter.generate_candidates(fused_rep, opt_obs, sar_obs, mc1_compat, indices=[0])

def test_4_missing_observations_rejected():
    fused_rep = make_dummy_fused_rep()
    opt_obs = {"observation_id": "opt123"}
    sar_obs = {"observation_id": "sar456"}
    mc1_compat = {"status": "compatible"}
    
    adapter = CrossModalEvidenceAdapter()
    with pytest.raises(ValueError, match="Missing or invalid optical observation"):
        adapter.generate_candidates(fused_rep, {}, sar_obs, mc1_compat, indices=[0])
        
    with pytest.raises(ValueError, match="Missing or invalid SAR observation"):
        adapter.generate_candidates(fused_rep, opt_obs, {}, mc1_compat, indices=[0])

def test_5_specific_token_indices():
    fused_rep = make_dummy_fused_rep()
    opt_obs = {"observation_id": "opt"}
    sar_obs = {"observation_id": "sar"}
    mc1_compat = {}
    
    adapter = CrossModalEvidenceAdapter()
    candidates = adapter.generate_candidates(fused_rep, opt_obs, sar_obs, mc1_compat, indices=[0, 42, 224])
    
    assert len(candidates) == 3
    assert candidates[0].token_identity.original_index == 0
    assert candidates[1].token_identity.original_index == 42
    assert candidates[2].token_identity.original_index == 224

def test_6_invalid_token_index_rejected():
    fused_rep = make_dummy_fused_rep()
    opt_obs = {"observation_id": "opt"}
    sar_obs = {"observation_id": "sar"}
    
    adapter = CrossModalEvidenceAdapter()
    with pytest.raises(IndexError, match="out of bounds"):
        adapter.generate_candidates(fused_rep, opt_obs, sar_obs, {}, indices=[225])

def test_7_mc5_evidence_contract():
    fused_rep = make_dummy_fused_rep()
    opt_obs = {"observation_id": "opt123"}
    sar_obs = {"observation_id": "sar456"}
    
    adapter = CrossModalEvidenceAdapter()
    candidates = adapter.generate_candidates(fused_rep, opt_obs, sar_obs, {}, indices=[42])
    
    evidence_dict = candidates[0].to_mc5_evidence()
    
    # Must pass existing MC5.1 validation
    assert validate_evidence_object(evidence_dict) is True
    
    assert evidence_dict["evidence_type"] == "cross_modal_spatial"
    assert evidence_dict["claim"] == "Unverified spatial multimodal candidate"
    assert evidence_dict["modality"] == "optical+sar"
    
    # Ensure source input contains IDs and index
    assert evidence_dict["source_input"]["optical_id"] == "opt123"
    assert evidence_dict["source_input"]["sar_id"] == "sar456"
    assert evidence_dict["source_input"]["token_index"] == 42
    
    # Ensure region matches spatial identities
    bounds = evidence_dict["spatial_region"]["bounds"]
    assert len(bounds) == 4
    
def test_8_determinism():
    fused_rep = make_dummy_fused_rep()
    opt_obs = {"observation_id": "opt1"}
    sar_obs = {"observation_id": "sar1"}
    
    adapter = CrossModalEvidenceAdapter()
    
    c1 = adapter.generate_candidates(fused_rep, opt_obs, sar_obs, {}, indices=[42])
    c2 = adapter.generate_candidates(fused_rep, opt_obs, sar_obs, {}, indices=[42])
    
    assert c1[0].evidence_id == c2[0].evidence_id
    
    # Change index
    c3 = adapter.generate_candidates(fused_rep, opt_obs, sar_obs, {}, indices=[43])
    assert c1[0].evidence_id != c3[0].evidence_id
    
    # Change observation
    c4 = adapter.generate_candidates(fused_rep, {"observation_id": "opt2"}, sar_obs, {}, indices=[42])
    assert c1[0].evidence_id != c4[0].evidence_id
    
    # Change query
    fused_rep_changed = copy.deepcopy(fused_rep)
    fused_rep_changed.query_context.original_query = "Find cars"
    c5 = adapter.generate_candidates(fused_rep_changed, opt_obs, sar_obs, {}, indices=[42])
    assert c1[0].evidence_id != c5[0].evidence_id

def test_9_soft_suitability():
    # Should not reject if soft compatibility warning exists
    fused_rep = make_dummy_fused_rep()
    opt_obs = {"observation_id": "opt"}
    sar_obs = {"observation_id": "sar"}
    mc1_compat = {"status": "compatible_with_warnings"}
    
    adapter = CrossModalEvidenceAdapter()
    candidates = adapter.generate_candidates(fused_rep, opt_obs, sar_obs, mc1_compat, indices=[0])
    assert len(candidates) == 1

def test_10_no_tensor_serialization():
    fused_rep = make_dummy_fused_rep()
    opt_obs = {"observation_id": "opt123"}
    sar_obs = {"observation_id": "sar456"}
    
    adapter = CrossModalEvidenceAdapter()
    candidates = adapter.generate_candidates(fused_rep, opt_obs, sar_obs, {}, indices=[42])
    
    evidence_dict = candidates[0].to_mc5_evidence()
    
    # Ensure no massive tensors inside the dictionary
    import json
    # Should be perfectly serializable to JSON
    json_str = json.dumps(evidence_dict)
    assert len(json_str) < 1000 # should be tiny
