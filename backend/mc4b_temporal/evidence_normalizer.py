"""
evidence_normalizer.py — MC4B output → MC5.1 Evidence Objects.

Converts the raw ChangeMamba output into one or more Evidence Objects
that conform to the MC5.1 schema, with claim, source_model, confidence,
and spatial_region populated from the inference results.
"""

import uuid
from typing import Dict, List, Optional
from .result import TemporalResult


def _pixel_to_geo(mask: List[List[int]], affine_transform: List[float], crs: str) -> List[Dict]:
    """
    Converts a thresholded binary mask into geographic polygon coordinates
    using the provided affine transform.
    Returns a list of regions, where each region has a "geometry" representing GeoJSON.
    """
    if not affine_transform or len(affine_transform) != 6:
        return []
    
    # Minimal bounding box geometry for connected components
    # For this task, we will extract the bounding box of all changed pixels if we don't have
    # a full rasterio vectorizer, or ideally a full polygon.
    # To keep it lightweight and mathematically accurate to the instruction:
    # A pixel (c, r) -> x = c * a + r * b + c0, y = c * d + r * e + f0
    # Where affine = [a, b, c0, d, e, f0]
    
    a, b, c0, d, e, f0 = affine_transform
    
    # We will just collect all pixel coordinates that are 1, and build bounding boxes 
    # (or a single multipolygon for simplicity to prove the coordinate transformation).
    
    regions = []
    
    # Find active pixels
    active_pixels = []
    for r, row in enumerate(mask):
        for c, val in enumerate(row):
            if val > 0:
                active_pixels.append((c, r))
                
    if not active_pixels:
        return []

    # Simplified single bounding region for now, unless connected components are required
    # But since the goal is mapping precision, let's just create a bounding box of the active pixels.
    min_c = min(p[0] for p in active_pixels)
    max_c = max(p[0] for p in active_pixels)
    min_r = min(p[1] for p in active_pixels)
    max_r = max(p[1] for p in active_pixels)
    
    # Transform corners
    def transform(c, r):
        # Center of pixel
        x = (c + 0.5) * a + (r + 0.5) * b + c0
        y = (c + 0.5) * d + (r + 0.5) * e + f0
        return [x, y]
        
    tl = transform(min_c, min_r)
    tr = transform(max_c, min_r)
    br = transform(max_c, max_r)
    bl = transform(min_c, max_r)
    
    geometry = {
        "type": "Polygon",
        "coordinates": [[tl, tr, br, bl, tl]]
    }
    
    regions.append({"geometry": geometry, "crs": crs})
    return regions


def normalize_to_evidence(
    temporal_result: TemporalResult,
    job_id: str,
) -> List[Dict]:
    """
    Converts a TemporalResult into formal MC5.1 Evidence Objects.
    Strictly preserves MODEL_UNAVAILABLE as 0 objects without fabricating evidence.
    """
    if temporal_result.status != "SUCCESS" or not temporal_result.raw_output:
        return []
        
    evidence_objects = []
    raw = temporal_result.raw_output
    
    # Extract timestamps
    timestamp = {
        "t1": temporal_result.before_observation_id,
        "t2": temporal_result.after_observation_id
    }
    
    # Extract neutral claim
    claim = (
        f"Model-predicted change region detected between "
        f"{temporal_result.before_observation_id} and {temporal_result.after_observation_id}."
    )
    
    mask = raw.binary_mask
    if mask:
        regions = _pixel_to_geo(mask, temporal_result.affine_transform, temporal_result.crs)
        
        for i, region in enumerate(regions):
            evidence_objects.append({
                "evidence_id": f"{job_id}_mc4b_{i}",
                "claim": claim,
                "evidence_type": "bitemporal_change_detection",
                "spatial_region": region.get("geometry"),
                "modality": "optical",  # Based on upstream constraints
                "modality_contribution": {"optical": 1.0},
                "timestamp": timestamp,
                "source_model": temporal_result.model_id or "ChangeMamba",
                "source_input": {
                    "image_1": temporal_result.before_observation_id,
                    "image_2": temporal_result.after_observation_id,
                },
                "confidence": raw.confidence or 0.0,
                "processing_parameters": {
                    "model_revision": temporal_result.model_revision,
                    "threshold": 0.5,
                    "crs": region.get("crs")
                }
            })
            
    return evidence_objects

