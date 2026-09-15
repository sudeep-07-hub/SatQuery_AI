import torch
from typing import Dict, Any, Tuple

def validate_pair(visual_data: Dict[str, Any], text_data: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Validates the invariant: visual.sample_id == text.sample_id
    and checks shape integrity for all required features.
    """
    if "sample_id" not in visual_data or "sample_id" not in text_data:
        return False, "Missing sample_id in one or both artifacts"
        
    v_id = str(visual_data["sample_id"])
    t_id = str(text_data["sample_id"])
    
    if v_id != t_id:
        return False, f"ID_mismatch: visual {v_id} != text {t_id}"
        
    # Check text features
    if "pooled_embedding" not in text_data:
        return False, "malformed_feature: Missing pooled_embedding in text data"
        
    t_emb = text_data["pooled_embedding"]
    if t_emb.shape != torch.Size([1, 2560]) and t_emb.shape != torch.Size([2560]):
        return False, f"malformed_feature: Invalid text feature shape {t_emb.shape}"
        
    if not torch.isfinite(t_emb).all():
        return False, "malformed_feature: Text embedding contains NaN/Inf"
        
    # Check visual features
    if "joint_encodings" not in visual_data:
        return False, "malformed_feature: Missing joint_encodings in visual data"
        
    v_emb = visual_data["joint_encodings"]
    if v_emb.shape != torch.Size([1, 225, 768]) and v_emb.shape != torch.Size([225, 768]):
        return False, f"malformed_feature: Invalid visual feature shape {v_emb.shape}"
        
    if not torch.isfinite(v_emb).all():
        return False, "malformed_feature: Visual embedding contains NaN/Inf"
        
    return True, "valid"
