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
    job_id: str,
) -> List[Dict]:
    """
    Convert MC4B output into MC5.1 Evidence Objects.

    Each changed region becomes a separate Evidence Object so that
    MC6 can verify/dispute individual spatial claims independently.

    Args:
        mc4b_output: Output from ChangeMambaAdapter.execute().
        mc1_profile: MC1 Structured Input Profile.
        query: Original user query.
        job_id: The job ID for generating stable evidence IDs.

    Returns:
        List of Evidence Objects conforming to MC5.1 schema.
    """
    if mc4b_output.get("status") != "SUCCESS":
        return []

    evidence_objects = []
    semantics = mc4b_output.get("semantics", {})
    regions = mc4b_output.get("changed_region_coordinates", [])
    
    # Bi-temporal timestamps
    t1_date = mc1_profile.get("image_1", {}).get("acquisition_date", "unknown")
    t2_date = mc1_profile.get("image_2", {}).get("acquisition_date", "unknown")
    timestamp = {"t1": t1_date, "t2": t2_date}

    # Modality
    modality = mc1_profile.get("image_1", {}).get("modality", "optical")

    # Build the claim from semantics (synthesized)
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
                    "evidence_id": f"{job_id}_mc4b_{i}",
                    "claim": claim,
                    "evidence_type": "bitemporal_change_detection",
                    "spatial_region": region.get("geometry"),
                    "modality": modality,
                    "modality_contribution": {modality: 1.0},
                    "timestamp": timestamp,
                    "source_model": mc4b_output.get("source_model", "CHANGE_MAMBA_TOOL"),
                    "source_input": {
                        "image_1": mc1_profile.get("image_1", {}).get("filename"),
                        "image_2": mc1_profile.get("image_2", {}).get("filename"),
                        "query": query,
                    },
                    "confidence": mc4b_output.get("model_confidence", 0.0),
                    "processing_parameters": {
                        "model_identity": "CHANGE_MAMBA",
                        "claim_source": "synthesized",
                        "threshold": 0.5,
                        "semantic_enabled": True,
                        "change_score": mc4b_output.get("change_score", 0.0),
                    },
                }
            )
    else:
        # No regions found — still produce one evidence object for the claim with footprint
        footprint = mc1_profile.get("footprint", None)
        evidence_objects.append(
            {
                "evidence_id": f"{job_id}_mc4b_0",
                "claim": claim,
                "evidence_type": "bitemporal_change_detection",
                "spatial_region": footprint,
                "modality": modality,
                "modality_contribution": {modality: 1.0},
                "timestamp": timestamp,
                "source_model": mc4b_output.get("source_model", "CHANGE_MAMBA_TOOL"),
                "source_input": {
                    "image_1": mc1_profile.get("image_1", {}).get("filename"),
                    "image_2": mc1_profile.get("image_2", {}).get("filename"),
                    "query": query,
                },
                "confidence": mc4b_output.get("model_confidence", 0.0),
                "processing_parameters": {
                    "model_identity": "CHANGE_MAMBA",
                    "claim_source": "synthesized",
                    "threshold": 0.5,
                    "semantic_enabled": True,
                    "change_score": mc4b_output.get("change_score", 0.0),
                },
            }
        )

    return evidence_objects
