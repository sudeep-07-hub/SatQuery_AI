import torch
from mc4c.optical_encoder import OpticalEncoder
from mc4c.sar_encoder import SAREncoder

def test_optical_encoder():
    print("Testing OpticalEncoder...")
    encoder = OpticalEncoder(croma_weights_path="mc4c/weights/CROMA_base.pt")
    dummy_input = torch.randn(2, 12, 120, 120)
    out = encoder.encode(dummy_input)
    assert 'patch_features' in out
    assert 'global_feature' in out
    
    # 120 / 8 = 15 -> 15x15 = 225 patches
    # base size dim = 768
    assert len(out['global_feature']) == 2
    assert len(out['global_feature'][0]) == 768
    assert len(out['patch_features']) == 2
    assert len(out['patch_features'][0]) == 225
    assert len(out['patch_features'][0][0]) == 768
    print("OpticalEncoder PASSED.")

def test_sar_encoder():
    print("Testing SAREncoder...")
    encoder = SAREncoder(croma_weights_path="mc4c/weights/CROMA_base.pt")
    dummy_input = torch.randn(2, 2, 120, 120)
    out = encoder.encode(dummy_input)
    assert 'patch_features' in out
    assert 'global_feature' in out
    
    assert len(out['global_feature']) == 2
    assert len(out['global_feature'][0]) == 768
    assert len(out['patch_features']) == 2
    assert len(out['patch_features'][0]) == 225
    assert len(out['patch_features'][0][0]) == 768
    print("SAREncoder PASSED.")

if __name__ == "__main__":
    test_optical_encoder()
    test_sar_encoder()
