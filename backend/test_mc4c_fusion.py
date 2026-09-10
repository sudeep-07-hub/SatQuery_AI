import torch
from mc4c.fusion import QueryConditionedFusion

def test_fusion():
    print("Testing QueryConditionedFusion...")
    fusion = QueryConditionedFusion(croma_weights_path="mc4c/weights/CROMA_base.pt")
    
    sar_features = torch.randn(2, 225, 768)
    optical_features = torch.randn(2, 225, 768)
    query = "Find planes on the airstrip"
    
    out = fusion(sar_features, optical_features, query)
    
    assert 'fused_vector' in out
    assert 'joint_encodings' in out
    assert len(out['fused_vector']) == 2
    assert len(out['fused_vector'][0]) == 768
    
    print("QueryConditionedFusion PASSED.")

if __name__ == "__main__":
    test_fusion()
