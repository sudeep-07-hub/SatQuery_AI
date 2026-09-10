from mc4c.verification import CrossModalVerifier
import torch

def test_verification():
    print("Testing CrossModalVerifier...")
    verifier = CrossModalVerifier()
    
    # Identical tensors should yield MATCH
    opt1 = torch.ones(1, 768).tolist()
    sar1 = torch.ones(1, 768).tolist()
    out1 = verifier.verify(opt1, sar1)
    assert out1['verification_status'] == "MATCH"
    import math
    assert math.isclose(out1['confidence_score'], 1.0, rel_tol=1e-5)
    
    # Orthogonal tensors should yield MISMATCH or PARTIAL depending on thresholds. 
    # Cosine sim of 0 with threshold 0.4 means MISMATCH
    opt2 = torch.tensor([[1.0, 0.0]]).tolist()
    sar2 = torch.tensor([[0.0, 1.0]]).tolist()
    out2 = verifier.verify(opt2, sar2)
    assert out2['verification_status'] == "MISMATCH"
    assert out2['confidence_score'] == 0.5 # (0 + 1) / 2
    
    print("CrossModalVerifier PASSED.")

if __name__ == "__main__":
    test_verification()
