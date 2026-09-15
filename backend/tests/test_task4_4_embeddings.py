import pytest
import torch
import os
import json
from mc4c.token_extractor import CROMATokenExtractor
from mc4c.modality_spatial import TokenRepresentationLayer
from mc4c.embedding_schema import ModalitySpatialConfig

def test_1_to_5_modality_representations():
    extractor = CROMATokenExtractor()
    layer = TokenRepresentationLayer()
    
    out = {
        "optical_encodings": torch.randn(1, 225, 768),
        "SAR_encodings": torch.randn(1, 225, 768),
        "joint_encodings": torch.randn(1, 225, 768),
    }
    bundle = extractor.extract(out)
    
    rep_opt = layer(bundle.optical)
    rep_sar = layer(bundle.sar)
    rep_joint = layer(bundle.joint)
    
    # 1. Optical exists
    assert rep_opt.modality_embeddings.shape == (1, 225, 16)
    # 2. SAR exists
    assert rep_sar.modality_embeddings.shape == (1, 225, 16)
    # 3. Joint exists
    assert rep_joint.modality_embeddings.shape == (1, 225, 16)
    
    # Check that they are distinct (optical != SAR)
    assert not torch.allclose(rep_opt.modality_embeddings, rep_sar.modality_embeddings)
    assert not torch.allclose(rep_opt.modality_embeddings, rep_joint.modality_embeddings)
    assert not torch.allclose(rep_sar.modality_embeddings, rep_joint.modality_embeddings)

def test_6_to_10_spatial_representations():
    extractor = CROMATokenExtractor()
    layer = TokenRepresentationLayer()
    
    bundle = extractor.extract({"optical_encodings": torch.randn(1, 225, 768)})
    rep = layer(bundle.optical)
    
    # 10. All 225 tokens receive spatial representations
    assert rep.spatial_embeddings.shape == (1, 225, 2)
    
    # 6. Spatial representation for token 0 (0, 0)
    assert torch.allclose(rep.spatial_embeddings[:, 0, :], torch.tensor([0.0, 0.0]))
    
    # 7. Spatial representation for token 14 (0, 14) -> row 0/14, col 14/14
    assert torch.allclose(rep.spatial_embeddings[:, 14, :], torch.tensor([0.0, 1.0]))
    
    # 8. Spatial representation for token 15 (1, 0) -> row 1/14, col 0/14
    assert torch.allclose(rep.spatial_embeddings[:, 15, :], torch.tensor([1/14.0, 0.0]))
    
    # 9. Spatial representation for token 224 (14, 14) -> row 14/14, col 14/14
    assert torch.allclose(rep.spatial_embeddings[:, 224, :], torch.tensor([1.0, 1.0]))

def test_11_and_26_deterministic_representations():
    extractor = CROMATokenExtractor()
    bundle = extractor.extract({"optical_encodings": torch.randn(1, 225, 768)})
    
    layer1 = TokenRepresentationLayer(ModalitySpatialConfig(seed=42))
    rep1 = layer1(bundle.optical)
    
    layer2 = TokenRepresentationLayer(ModalitySpatialConfig(seed=42))
    rep2 = layer2(bundle.optical)
    
    assert torch.allclose(rep1.modality_embeddings, rep2.modality_embeddings)
    assert torch.allclose(rep1.spatial_embeddings, rep2.spatial_embeddings)

def test_12_and_13_identity_preservation_on_subset():
    extractor = CROMATokenExtractor()
    layer = TokenRepresentationLayer()
    
    bundle = extractor.extract({"optical_encodings": torch.randn(1, 225, 768)})
    rep = layer(bundle.optical)
    
    # Select [224, 0, 15]
    subset = rep.select([224, 0, 15])
    
    # Verify spatial embeddings reorder perfectly
    assert torch.allclose(subset.spatial_embeddings[:, 0, :], torch.tensor([1.0, 1.0])) # 224
    assert torch.allclose(subset.spatial_embeddings[:, 1, :], torch.tensor([0.0, 0.0])) # 0
    assert torch.allclose(subset.spatial_embeddings[:, 2, :], torch.tensor([1/14.0, 0.0])) # 15
    
    # Verify modality embeddings are preserved (all optical)
    assert torch.allclose(subset.modality_embeddings[:, 0, :], rep.modality_embeddings[:, 224, :])
    assert torch.allclose(subset.modality_embeddings[:, 1, :], rep.modality_embeddings[:, 0, :])
    
    assert len(subset.spatial_identities) == 3

def test_14_to_16_modality_spatial_correspondence():
    extractor = CROMATokenExtractor()
    layer = TokenRepresentationLayer()
    
    out = {
        "optical_encodings": torch.randn(1, 225, 768),
        "SAR_encodings": torch.randn(1, 225, 768),
        "joint_encodings": torch.randn(1, 225, 768),
    }
    bundle = extractor.extract(out)
    
    opt = layer(bundle.optical)
    sar = layer(bundle.sar)
    joint = layer(bundle.joint)
    
    idx = 42
    
    # 14. Shared spatial identity
    assert torch.allclose(opt.spatial_embeddings[:, idx, :], sar.spatial_embeddings[:, idx, :])
    
    # 15. Distinct modality identity
    assert not torch.allclose(opt.modality_embeddings[:, idx, :], sar.modality_embeddings[:, idx, :])
    
    # 16. Joint token has correct spatial identity
    assert torch.allclose(joint.spatial_embeddings[:, idx, :], opt.spatial_embeddings[:, idx, :])
    assert joint.modality == "joint"

def test_17_batch_dimension():
    extractor = CROMATokenExtractor()
    layer = TokenRepresentationLayer()
    
    bundle = extractor.extract({"optical_encodings": torch.randn(4, 225, 768)})
    rep = layer(bundle.optical)
    
    assert rep.original_tokens.shape == (4, 225, 768)
    assert rep.modality_embeddings.shape == (4, 225, 16)
    assert rep.spatial_embeddings.shape == (4, 225, 2)

def test_18_to_20_no_modifications():
    extractor = CROMATokenExtractor()
    layer = TokenRepresentationLayer()
    
    raw = torch.randn(1, 225, 768)
    bundle = extractor.extract({"optical_encodings": raw})
    rep = layer(bundle.optical)
    
    # 18. Values unchanged
    assert torch.equal(rep.original_tokens, raw)
    
    # 19. No GAP (shape still B, N, D)
    assert rep.original_tokens.shape == (1, 225, 768)
    
    # 20. No Qwen projection (dim still 768)
    assert rep.original_tokens.shape[-1] == 768

def test_21_metadata_alignment_checks():
    extractor = CROMATokenExtractor()
    layer = TokenRepresentationLayer()
    bundle = extractor.extract({"optical_encodings": torch.randn(1, 225, 768)})
    rep = layer(bundle.optical)
    
    # manually break alignment
    rep.spatial_identities.pop()
    with pytest.raises(RuntimeError, match="Metadata misalignment"):
        rep.select([0])

def test_23_one_cell_grid():
    from mc4c.token_schema import SpatialTokenGrid, TokenSpatialIdentity
    
    # Simulate a 1x1 grid
    identity = TokenSpatialIdentity(original_index=0, row=0, column=0, bounds=(0,0,8,8))
    grid = SpatialTokenGrid(
        modality="optical",
        batch_size=1,
        grid_height=1,
        grid_width=1,
        num_tokens=1,
        token_dim=768,
        patch_height=8,
        patch_width=8,
        tokens=torch.randn(1, 1, 768),
        spatial_identities=[identity]
    )
    
    layer = TokenRepresentationLayer()
    rep = layer(grid)
    
    # Should not divide by zero, fallback to 0.5
    assert torch.allclose(rep.spatial_embeddings[:, 0, :], torch.tensor([0.5, 0.5]))

def test_24_serialization():
    extractor = CROMATokenExtractor()
    layer = TokenRepresentationLayer()
    bundle = extractor.extract({"optical_encodings": torch.randn(1, 225, 768)})
    rep = layer(bundle.optical)
    
    dump = rep.model_dump()
    assert "original_tokens" in dump
    assert "Tensor shape" in dump["original_tokens"]
    assert "Tensor shape" in dump["spatial_embeddings"]
    
    json.dumps(dump)

def test_25_config_dimensions():
    extractor = CROMATokenExtractor()
    config = ModalitySpatialConfig(modality_dim=32, spatial_dim=2)
    layer = TokenRepresentationLayer(config)
    
    bundle = extractor.extract({"optical_encodings": torch.randn(1, 225, 768)})
    rep = layer(bundle.optical)
    
    assert rep.modality_embeddings.shape == (1, 225, 32)


CROMA_WEIGHTS = os.path.join(os.path.dirname(__file__), "..", "mc4c", "weights", "CROMA_base.pt")
WEIGHTS_AVAILABLE = os.path.exists(CROMA_WEIGHTS)
skip_no_weights = pytest.mark.skipif(not WEIGHTS_AVAILABLE, reason="CROMA weights unavailable")

@skip_no_weights
def test_real_croma_execution():
    from mc4c.croma import PretrainedCROMA
    model = PretrainedCROMA(pretrained_path=CROMA_WEIGHTS, size='base', modality='both')
    model.eval()
    
    opt_input = torch.randn(1, 12, 120, 120)
    sar_input = torch.randn(1, 2, 120, 120)
    
    with torch.no_grad():
        out = model(SAR_images=sar_input, optical_images=opt_input)
        
    extractor = CROMATokenExtractor()
    bundle = extractor.extract(out)
    
    layer = TokenRepresentationLayer()
    opt_rep = layer(bundle.optical)
    sar_rep = layer(bundle.sar)
    
    assert opt_rep.original_tokens.shape == (1, 225, 768)
    assert opt_rep.modality_embeddings.shape == (1, 225, 16)
    assert opt_rep.spatial_embeddings.shape == (1, 225, 2)
    
    subset = opt_rep.select([0, 42, 224])
    
    assert subset.num_tokens == 3
    assert subset.original_tokens.shape == (1, 3, 768)
    assert subset.modality_embeddings.shape == (1, 3, 16)
    assert subset.spatial_embeddings.shape == (1, 3, 2)
    assert subset.spatial_identities[1].row == 2
    assert subset.spatial_identities[1].column == 12
