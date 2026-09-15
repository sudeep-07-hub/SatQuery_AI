import pytest
import torch
import os
import json
from mc4c.token_schema import SpatialTokenGrid, CROMATokenBundle, ModalityType
from mc4c.token_extractor import CROMATokenExtractor

def test_1_token_grid_coordinate_mapping():
    # 15x15 grid
    grid = SpatialTokenGrid(modality="optical", tokens=torch.zeros(1, 225, 768))
    
    # index 0 -> row 0, col 0
    assert grid.index_to_coord(0) == (0, 0)
    assert grid.coord_to_index(0, 0) == 0
    
    # index 14 -> row 0, col 14
    assert grid.index_to_coord(14) == (0, 14)
    assert grid.coord_to_index(0, 14) == 14
    
    # index 15 -> row 1, col 0
    assert grid.index_to_coord(15) == (1, 0)
    assert grid.coord_to_index(1, 0) == 15
    
    # index 224 -> row 14, col 14
    assert grid.index_to_coord(224) == (14, 14)
    assert grid.coord_to_index(14, 14) == 224
    
def test_2_token_grid_coordinate_out_of_bounds():
    grid = SpatialTokenGrid(modality="optical", tokens=torch.zeros(1, 225, 768))
    
    with pytest.raises(ValueError):
        grid.index_to_coord(-1)
    with pytest.raises(ValueError):
        grid.index_to_coord(225)
        
    with pytest.raises(ValueError):
        grid.coord_to_index(-1, 0)
    with pytest.raises(ValueError):
        grid.coord_to_index(15, 0)
    with pytest.raises(ValueError):
        grid.coord_to_index(0, 15)

def test_3_patch_bounds():
    grid = SpatialTokenGrid(modality="optical", tokens=torch.zeros(1, 225, 768))
    # 8x8 patches
    # index 0 (0,0) -> 0, 0, 8, 8
    assert grid.patch_bounds(0) == (0, 0, 8, 8)
    # index 1 (0,1) -> 0, 8, 8, 16
    assert grid.patch_bounds(1) == (0, 8, 8, 16)
    # index 15 (1,0) -> 8, 0, 16, 8
    assert grid.patch_bounds(15) == (8, 0, 16, 8)
    
def test_4_metadata_serialization_omits_tensor():
    grid = SpatialTokenGrid(modality="optical", tokens=torch.zeros(1, 225, 768))
    dumped = grid.model_dump()
    assert dumped["modality"] == "optical"
    assert "Tensor shape" in dumped["tokens"]
    
    # ensure it is JSON serializable
    json_str = json.dumps(dumped)
    assert "Tensor shape" in json_str

def test_5_token_extractor_valid_inputs():
    extractor = CROMATokenExtractor()
    opt = torch.zeros(1, 225, 768)
    sar = torch.ones(1, 225, 768)
    joint = torch.ones(1, 225, 768) * 2
    
    croma_out = {
        "optical_encodings": opt,
        "SAR_encodings": sar,
        "joint_encodings": joint,
        "optical_GAP": torch.zeros(1, 768),  # should be ignored by extractor
    }
    
    bundle = extractor.extract(croma_out)
    
    assert bundle.optical is not None
    assert bundle.optical.modality == "optical"
    assert bundle.optical.num_tokens == 225
    assert torch.equal(bundle.optical.tokens, opt)
    
    assert bundle.sar is not None
    assert bundle.sar.modality == "sar"
    assert torch.equal(bundle.sar.tokens, sar)
    
    assert bundle.joint is not None
    assert bundle.joint.modality == "joint"
    assert torch.equal(bundle.joint.tokens, joint)
    
    # check no mutation
    assert id(bundle.optical.tokens) == id(opt)

def test_6_token_extractor_invalid_rank():
    extractor = CROMATokenExtractor()
    croma_out = {
        "optical_encodings": torch.zeros(225, 768),  # missing batch
    }
    with pytest.raises(ValueError, match="Expected 3D tensor"):
        extractor.extract(croma_out)

def test_7_token_extractor_invalid_token_count():
    extractor = CROMATokenExtractor()
    croma_out = {
        "optical_encodings": torch.zeros(1, 100, 768), 
    }
    with pytest.raises(ValueError, match="Expected 225 tokens"):
        extractor.extract(croma_out)

def test_8_token_extractor_invalid_token_dim():
    extractor = CROMATokenExtractor()
    croma_out = {
        "optical_encodings": torch.zeros(1, 225, 512), 
    }
    with pytest.raises(ValueError, match="Expected token dimension 768"):
        extractor.extract(croma_out)
        
def test_9_token_extractor_batch_greater_than_one():
    extractor = CROMATokenExtractor()
    opt = torch.zeros(4, 225, 768)
    croma_out = {
        "optical_encodings": opt,
    }
    bundle = extractor.extract(croma_out)
    assert bundle.optical.batch_size == 4
    assert bundle.optical.tokens.shape == (4, 225, 768)
    
CROMA_WEIGHTS = os.path.join(os.path.dirname(__file__), "..", "mc4c", "weights", "CROMA_base.pt")
WEIGHTS_AVAILABLE = os.path.exists(CROMA_WEIGHTS)
skip_no_weights = pytest.mark.skipif(not WEIGHTS_AVAILABLE, reason="CROMA weights unavailable")

@skip_no_weights
def test_10_real_croma_execution_extraction():
    from mc4c.croma import PretrainedCROMA
    model = PretrainedCROMA(pretrained_path=CROMA_WEIGHTS, size='base', modality='both')
    model.eval()
    
    opt_input = torch.randn(1, 12, 120, 120)
    sar_input = torch.randn(1, 2, 120, 120)
    
    with torch.no_grad():
        out = model(SAR_images=sar_input, optical_images=opt_input)
        
    extractor = CROMATokenExtractor()
    bundle = extractor.extract(out)
    
    assert bundle.optical is not None
    assert bundle.optical.tokens.shape == (1, 225, 768)
    assert bundle.sar is not None
    assert bundle.sar.tokens.shape == (1, 225, 768)
    assert bundle.joint is not None
    assert bundle.joint.tokens.shape == (1, 225, 768)
    
    # Verify backward compatibility (the original dict still has GAP)
    assert 'optical_GAP' in out
    assert out['optical_GAP'].shape == (1, 768)
