import pytest
import torch
import os
import json
from mc4c.token_extractor import CROMATokenExtractor

def test_1_to_9_exhaustive_identity_mapping():
    extractor = CROMATokenExtractor()
    opt = torch.randn(1, 225, 768)
    bundle = extractor.extract({"optical_encodings": opt})
    grid = bundle.optical
    
    # 8. Every token index maps to exactly one grid position
    # 9. Every grid position maps to exactly one token index
    mapped_positions = set()
    mapped_indices = set()
    
    for i in range(225):
        identity = grid.spatial_identities[i]
        assert identity.original_index == i
        
        # 1. Full 15x15 mapping invariant
        assert identity.original_index == identity.row * 15 + identity.column
        
        mapped_positions.add((identity.row, identity.column))
        mapped_indices.add(identity.original_index)
        
    assert len(mapped_positions) == 225
    assert len(mapped_indices) == 225
    
    # Specific points
    # 2. token 0
    assert grid.spatial_identities[0].row == 0
    assert grid.spatial_identities[0].column == 0
    
    # 3. token 14
    assert grid.spatial_identities[14].row == 0
    assert grid.spatial_identities[14].column == 14
    
    # 4. token 15
    assert grid.spatial_identities[15].row == 1
    assert grid.spatial_identities[15].column == 0
    
    # 5. token 16
    assert grid.spatial_identities[16].row == 1
    assert grid.spatial_identities[16].column == 1
    
    # 6. token 224
    assert grid.spatial_identities[224].row == 14
    assert grid.spatial_identities[224].column == 14
    
    # 7. Reverse mapping is handled implicitly by the identity structure and schema mapping
    assert grid.coord_to_index(14, 14) == 224

def test_10_to_11_patch_bounds():
    extractor = CROMATokenExtractor()
    bundle = extractor.extract({"optical_encodings": torch.randn(1, 225, 768)})
    grid = bundle.optical
    
    # 10. corners
    # Top-left (token 0)
    assert grid.spatial_identities[0].bounds == (0, 0, 8, 8)
    # Top-right (token 14)
    assert grid.spatial_identities[14].bounds == (0, 112, 8, 120)
    # Bottom-left (token 210)
    assert grid.spatial_identities[210].bounds == (112, 0, 120, 8)
    # Bottom-right (token 224)
    assert grid.spatial_identities[224].bounds == (112, 112, 120, 120)
    
    # 11. Center token (7, 7) -> index 7*15+7 = 112
    assert grid.spatial_identities[112].bounds == (56, 56, 64, 64)

def test_12_to_16_modality_correspondence():
    extractor = CROMATokenExtractor()
    out = {
        "optical_encodings": torch.randn(1, 225, 768),
        "SAR_encodings": torch.randn(1, 225, 768),
        "joint_encodings": torch.randn(1, 225, 768),
    }
    bundle = extractor.extract(out)
    
    opt_ids = bundle.optical.spatial_identities
    sar_ids = bundle.sar.spatial_identities
    joint_ids = bundle.joint.spatial_identities
    
    # Test random sampling of indices for identical correspondence
    for i in [0, 37, 112, 224]:
        # 12, 13, 14. Modality identity
        assert opt_ids[i].original_index == i
        assert sar_ids[i].original_index == i
        assert joint_ids[i].original_index == i
        
        # 15. Optical and SAR resolve to same patch
        assert opt_ids[i].bounds == sar_ids[i].bounds
        assert opt_ids[i].row == sar_ids[i].row
        
        # 16. Joint resolves to same patch
        assert joint_ids[i].bounds == opt_ids[i].bounds

def test_17_to_20_token_selection():
    extractor = CROMATokenExtractor()
    opt = torch.randn(1, 225, 768)
    bundle = extractor.extract({"optical_encodings": opt})
    grid = bundle.optical
    
    # 19. Token reordering and 17, 18. Token selection preservation
    # Select non-sequential, out-of-order tokens
    indices = [224, 0, 15]
    subset = grid.select(indices)
    
    # 20. Metadata/tensor count aligned
    assert subset.num_tokens == 3
    assert subset.tokens.shape[1] == 3
    assert len(subset.spatial_identities) == 3
    
    # Identity preservation check
    assert subset.spatial_identities[0].original_index == 224
    assert subset.spatial_identities[0].row == 14
    assert subset.spatial_identities[0].column == 14
    
    assert subset.spatial_identities[1].original_index == 0
    assert subset.spatial_identities[1].row == 0
    
    assert subset.spatial_identities[2].original_index == 15
    assert subset.spatial_identities[2].row == 1
    
    # 24. Token embeddings not modified
    assert torch.equal(subset.tokens[:, 0, :], opt[:, 224, :])
    assert torch.equal(subset.tokens[:, 1, :], opt[:, 0, :])

def test_21_to_22_selection_validation():
    extractor = CROMATokenExtractor()
    bundle = extractor.extract({"optical_encodings": torch.randn(1, 225, 768)})
    grid = bundle.optical
    
    # 21. Invalid index rejected
    with pytest.raises(IndexError):
        grid.select([225])
        
    with pytest.raises(IndexError):
        grid.select([-1])
        
    # 22. Duplicate index is allowed in tensors normally (advanced indexing)
    # We explicitly define that duplicate indices yield duplicate identical metadata
    subset = grid.select([5, 5])
    assert subset.num_tokens == 2
    assert subset.spatial_identities[0] == subset.spatial_identities[1]
    assert subset.spatial_identities[0].original_index == 5

def test_23_batch_dimension_preserved():
    extractor = CROMATokenExtractor()
    # Batch size 4
    opt = torch.randn(4, 225, 768)
    bundle = extractor.extract({"optical_encodings": opt})
    grid = bundle.optical
    
    assert grid.batch_size == 4
    assert grid.tokens.shape == (4, 225, 768)
    # Spatial identity is across the spatial dimension, metadata count is 225
    assert len(grid.spatial_identities) == 225
    
    subset = grid.select([10, 20])
    assert subset.batch_size == 4
    assert subset.tokens.shape == (4, 2, 768)

def test_25_no_gap_occurs():
    extractor = CROMATokenExtractor()
    bundle = extractor.extract({"optical_encodings": torch.randn(1, 225, 768)})
    grid = bundle.optical
    subset = grid.select([0, 1, 2])
    
    # Tensor rank is preserved, sequence dim is modified, no collapse to (B, D)
    assert subset.tokens.ndim == 3
    assert subset.tokens.shape == (1, 3, 768)

def test_26_metadata_serialization():
    extractor = CROMATokenExtractor()
    bundle = extractor.extract({"optical_encodings": torch.randn(1, 225, 768)})
    
    dump = bundle.model_dump()
    assert "optical" in dump
    assert dump["optical"]["num_tokens"] == 225
    assert len(dump["optical"]["spatial_identities"]) == 225
    
    # Tensor is safely stringified
    assert isinstance(dump["optical"]["tokens"], str)
    assert "Tensor shape" in dump["optical"]["tokens"]
    
    # Completely json serializable
    assert json.dumps(dump)

def test_27_invalid_metadata_alignment():
    extractor = CROMATokenExtractor()
    bundle = extractor.extract({"optical_encodings": torch.randn(1, 225, 768)})
    grid = bundle.optical
    
    # Maliciously misalign the metadata
    grid.spatial_identities.pop()
    
    # The select method should guard against this
    with pytest.raises(RuntimeError, match="Metadata misalignment"):
        grid.select([0])

def test_28_deterministic_repeated_mapping():
    extractor = CROMATokenExtractor()
    bundle = extractor.extract({"optical_encodings": torch.randn(1, 225, 768)})
    grid = bundle.optical
    
    subset1 = grid.select([50, 100])
    subset2 = grid.select([50, 100])
    
    assert subset1.spatial_identities[0] == subset2.spatial_identities[0]
    assert subset1.spatial_identities[1] == subset2.spatial_identities[1]

CROMA_WEIGHTS = os.path.join(os.path.dirname(__file__), "..", "mc4c", "weights", "CROMA_base.pt")
WEIGHTS_AVAILABLE = os.path.exists(CROMA_WEIGHTS)
skip_no_weights = pytest.mark.skipif(not WEIGHTS_AVAILABLE, reason="CROMA weights unavailable")

@skip_no_weights
def test_real_croma_validation_spatial_preservation():
    """
    Validates that real model execution produces spatial tokens that survive the extraction and selection layer.
    """
    from mc4c.croma import PretrainedCROMA
    model = PretrainedCROMA(pretrained_path=CROMA_WEIGHTS, size='base', modality='both')
    model.eval()
    
    opt_input = torch.randn(1, 12, 120, 120)
    sar_input = torch.randn(1, 2, 120, 120)
    
    with torch.no_grad():
        out = model(SAR_images=sar_input, optical_images=opt_input)
        
    extractor = CROMATokenExtractor()
    bundle = extractor.extract(out)
    
    opt_grid = bundle.optical
    assert opt_grid.tokens.shape == (1, 225, 768)
    
    # Simulate an agent or downstream logic selecting tokens [0, 42, 224]
    selected_subset = opt_grid.select([0, 42, 224])
    
    assert selected_subset.num_tokens == 3
    assert selected_subset.tokens.shape == (1, 3, 768)
    
    # Verify the selected tensor exactly matches the real raw outputs at those indices
    assert torch.equal(selected_subset.tokens[:, 0, :], out["optical_encodings"][:, 0, :])
    assert torch.equal(selected_subset.tokens[:, 1, :], out["optical_encodings"][:, 42, :])
    assert torch.equal(selected_subset.tokens[:, 2, :], out["optical_encodings"][:, 224, :])
    
    # Verify spatial identity survives
    assert selected_subset.spatial_identities[0].original_index == 0
    assert selected_subset.spatial_identities[1].original_index == 42
    assert selected_subset.spatial_identities[2].original_index == 224
