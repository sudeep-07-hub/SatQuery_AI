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
    job_id: str,
) -> List[Dict]:
    """
    Convert MC4A output into MC5.1 Evidence Objects.

    Args:
        mc4a_output: Output from PaliGemmaVQASpecialist.run().
        mc1_profile: MC1 Structured Input Profile.
        mc2_task_spec: MC2 Structured Task Specification.
        job_id: The job ID for generating stable evidence IDs.

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
    claim_is_truncated = False
    
    sentences = [s.strip() for s in answer.split('.') if s.strip()]
    if len(sentences) > 1:
        claim = sentences[0] + "."
        claim_is_truncated = True
    else:
        claim = answer

    # Spatial region fallback
    spatial_region = mc4a_output.get("spatial_evidence")
    spatial_region_is_full_image = False
    if not spatial_region:
        spatial_region = mc1_profile.get("footprint", None)
        spatial_region_is_full_image = True
    
    # In MC4A currently we don't have bounding_boxes/segmentation_masks array
    # If we did, we would iterate and yield one per entity. Here we yield one for the whole answer.
    evidence_objects.append({
        "evidence_id": f"{job_id}_mc4a_0",
        "claim": claim,
        "evidence_type": "vqa_grounded_detection",
        "spatial_region": spatial_region,
        "modality": modality,
        "modality_contribution": {modality: 1.0},
        "timestamp": timestamp,
        "source_model": "PALIGEMMA_VQA_TOOL",
        "source_input": img_meta.get("filename", target_image_key),
        "confidence": mc4a_output.get("model_confidences", {}).get("paligemma_vqa", 0.0),
        "processing_parameters": {
            "prompt_prefix": "answer en",
            "max_new_tokens": 60,
            "dtype": "bfloat16",
            "device": "auto",
            "model_id": "google/paligemma-3b-pt-224",
            "timestamp_source": timestamp_source,
            "claim_is_truncated": claim_is_truncated,
            "claim_source": "verbatim",
            "spatial_region_is_full_image": spatial_region_is_full_image
        }
    })
    
    return evidence_objects
