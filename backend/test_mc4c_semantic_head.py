import torch
from mc4c.semantic_head import SemanticHead

def test_semantic_head():
    print("Testing SemanticHead...")
    head = SemanticHead(k=3)
    
    # BigEarthNet accepts S2 (or S1+S2). The pretrained model we are using `resnet50-all-v0.2.0`
    # expects 14 channels (S1+S2) according to standard BigEarthNet v2.0 'all' models.
    # Wait, does resnet50-all expect 14 channels? Yes, usually 10+2=12 or 10+2+2 = 14. 
    # But let's check what it actually expects in the test. If it crashes, we'll see.
    # We will pass a dummy tensor of shape (B, 14, 120, 120).
    
    try:
        dummy_input = torch.randn(2, 14, 120, 120)
        tags = head.get_tags(dummy_input)
    except Exception as e:
        print(f"Failed with 14 channels: {e}")
        try:
            dummy_input = torch.randn(2, 12, 120, 120)
            tags = head.get_tags(dummy_input)
        except Exception as e:
            print(f"Failed with 12 channels: {e}")
            raise e
            
    assert len(tags) == 2
    assert len(tags[0]) == 3 # top 3
    assert isinstance(tags[0][0], str)
    
    print("SemanticHead PASSED.")
    print("Example tags:", tags[0])

if __name__ == "__main__":
    test_semantic_head()
