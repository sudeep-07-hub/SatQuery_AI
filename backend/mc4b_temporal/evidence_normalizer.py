"""
evidence_normalizer.py — MC4B output → MC5.1 Evidence Objects.

Converts the raw ChangeMamba output into one or more Evidence Objects
that conform to the MC5.1 schema, with claim, source_model, confidence,
and spatial_region populated from the inference results.
"""

import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional


def normalize_to_evidence(
    mc4b_output: Dict,
    mc1_profile: Dict,
    query: str,
) -> List[Dict]:
    """
    Convert MC4B output into MC5.1 Evidence Objects.

    Each changed region becomes a separate Evidence Object so that
    MC6 can verify/dispute individual spatial claims independently.

    Args:
        mc4b_output: Output from ChangeMambaAdapter.execute().
        mc1_profile: MC1 Structured Input Profile.
        query: Original user query.

    Returns:
        List of Evidence Objects conforming to MC5.1 schema.
    """
    if mc4b_output.get("status") != "SUCCESS":
        return []

    evidence_objects = []
    semantics = mc4b_output.get("semantics", {})
    regions = mc4b_output.get("changed_region_coordinates", [])
    timestamp = datetime.now(timezone.utc).isoformat()

    # Build the claim from semantics
    description = semantics.get("description", "Change detected")
    stats = mc4b_output.get("change_statistics", {})
    claim = (
        f"{description}. "
        f"Changed area: {stats.get('changed_area_m2', 0)} m², "
        f"{stats.get('changed_pixel_pct', 0)}% of scene."
    )

    if regions:
        # One evidence object per spatial region
        for i, region in enumerate(regions):
            evidence_objects.append(
                {
                    "evidence_id": str(uuid.uuid4()),
                    "claim": claim,
                    "evidence_type": "change_detection_mask",
                    "spatial_region": region.get("geometry"),
                    "modality": "optical",
                    "timestamp": timestamp,
                    "source_model": mc4b_output.get("source_model", "CHANGE_MAMBA"),
                    "source_input": {
                        "image_1": mc1_profile.get("image_1", {}).get("filename"),
                        "image_2": mc1_profile.get("image_2", {}).get("filename"),
                        "query": query,
                    },
                    "confidence": mc4b_output.get("model_confidence", 0.0),
                    "processing_parameters": {
                        "backbone": "lightweight_cnn",
                        "threshold": 0.5,
                        "semantic_enabled": True,
                        "change_score": mc4b_output.get("change_score", 0.0),
                    },
                }
            )
    else:
        # No regions found — still produce one evidence object for the claim
        evidence_objects.append(
            {
                "evidence_id": str(uuid.uuid4()),
                "claim": claim,
                "evidence_type": "change_detection_mask",
                "spatial_region": None,
                "modality": "optical",
                "timestamp": timestamp,
                "source_model": mc4b_output.get("source_model", "CHANGE_MAMBA"),
                "source_input": {
                    "image_1": mc1_profile.get("image_1", {}).get("filename"),
                    "image_2": mc1_profile.get("image_2", {}).get("filename"),
                    "query": query,
                },
                "confidence": mc4b_output.get("model_confidence", 0.0),
                "processing_parameters": {
                    "backbone": "lightweight_cnn",
                    "threshold": 0.5,
                    "semantic_enabled": True,
                    "change_score": mc4b_output.get("change_score", 0.0),
                },
            }
        )

    return evidence_objects
