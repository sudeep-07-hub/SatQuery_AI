"""
evidence_normalizer.py — MC4A output → MC5.1 Evidence Objects.
"""

import uuid
from datetime import datetime, timezone
from typing import Dict, List

def to_evidence_object(
    mc4a_output: Dict,
    mc1_profile: Dict,
    mc2_task_spec: Dict,
) -> List[Dict]:
    """
    Convert MC4A output into MC5.1 Evidence Objects.

    Args:
        mc4a_output: Output from PaliGemmaVQASpecialist.run().
        mc1_profile: MC1 Structured Input Profile.
        mc2_task_spec: MC2 Structured Task Specification.

    Returns:
        List of Evidence Objects conforming to MC5.1 schema.
    """
    evidence_objects = []
    
    # Check if inference was blocked
    if "blocked_reason" in mc4a_output:
        return []

    timestamp_source = "system_utc"
    timestamp = datetime.now(timezone.utc).isoformat()
    
    # Determine target image
    target_image_key = "image_1"
    query = mc2_task_spec.get("query", "").lower()
    if mc1_profile.get("image_count") == 2:
        if "second" in query or "image 2" in query:
            target_image_key = "image_2"
            
    img_meta = mc1_profile.get(target_image_key, {})
    if "acquisition_date" in img_meta:
        timestamp = img_meta["acquisition_date"]
        timestamp_source = "metadata"

    modality = img_meta.get("modality", "optical")
    
    # Build claim
    answer = mc4a_output.get("textual_answer") or mc4a_output.get("caption") or "No answer provided."
    claim = f"Q: {mc2_task_spec.get('query', '')} → A: {answer}"
    
    evidence_objects.append({
        "evidence_id": str(uuid.uuid4()),
        "claim": claim,
        "evidence_type": "non_localized_vqa_response",
        "spatial_region": mc4a_output.get("spatial_evidence"),
        "modality": modality,
        "timestamp": timestamp,
        "source_model": "PaliGemma-3B-VQA (google/paligemma-3b-pt-224)",
        "source_input": target_image_key,
        "confidence": mc4a_output.get("model_confidences", {}).get("paligemma_vqa", 0.0),
        "processing_parameters": {
            "prompt_prefix": "answer en",
            "max_new_tokens": 60,
            "dtype": "bfloat16",
            "device": "auto",
            "model_id": "google/paligemma-3b-pt-224",
            "timestamp_source": timestamp_source
        }
    })
    
    return evidence_objects
