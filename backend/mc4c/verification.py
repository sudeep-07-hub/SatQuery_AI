import torch
import torch.nn.functional as F

class CrossModalVerifier:
    def __init__(self, match_threshold: float = 0.7, partial_threshold: float = 0.4):
        self.match_threshold = match_threshold
        self.partial_threshold = partial_threshold

    def verify(self, optical_global_feature: list, sar_global_feature: list) -> dict:
        """
        Takes the global features (e.g. from GAP) from Optical and SAR encoders.
        Returns a dict containing 'verification_status' and 'confidence_score'.
        """
        opt_tensor = torch.tensor(optical_global_feature)
        sar_tensor = torch.tensor(sar_global_feature)
        
        # Compute cosine similarity
        sim = F.cosine_similarity(opt_tensor, sar_tensor, dim=-1)
        
        # In a real batch scenario, sim would be a tensor of shape (B,). 
        # Here we assume a single item or take the mean.
        score = sim.mean().item()
        
        if score >= self.match_threshold:
            status = "MATCH"
        elif score >= self.partial_threshold:
            status = "PARTIAL"
        else:
            status = "MISMATCH"
            
        # Normalize confidence score between 0 and 1
        confidence = max(0.0, min(1.0, (score + 1) / 2))
        
        return {
            'verification_status': status,
            'confidence_score': confidence
        }
