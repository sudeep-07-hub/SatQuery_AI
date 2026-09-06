"""
tool_adapter.py — Wraps MC4B as a MC3.1 Tool Registry-conformant object.

Implements executable precondition validators and a structured refusal path.
This is the only entry-point the MC3 Workflow Planner calls — it never
touches backbone/change_head/localization directly.
"""

import torch
import numpy as np
from typing import Dict, Optional, List

from . import config
from .backbone import get_backbone
from .change_head import ChangeDetector
from .semantics import extract_change_types
from .captioner import generate_change_caption
from .localization import (
    mask_to_regions,
    regions_to_geojson,
    compute_change_statistics,
)


# ── MC3.1 Tool Registry Entry ────────────────────────────────────

CHANGE_MAMBA_TOOL = {
    "name": "CHANGE_MAMBA",
    "task_capabilities": [
        "binary_change_detection",
        "semantic_change_detection",
        "change_localization",
        "change_captioning",
    ],
    "supported_modalities": ["optical"],
    "input_constraints": {
        "image_count": 2,
        "same_modality": True,
        "same_crs": True,
        "min_spatial_overlap": config.MIN_SPATIAL_OVERLAP,
        "min_coregistration_score": config.MIN_COREGISTRATION_SCORE,
    },
    "spatial_resolution_range": {"min_gsd_m": 0.3, "max_gsd_m": 30.0},
    "temporal_requirements": ["bi_temporal_change"],
    "output_types": [
        "binary_change_mask",
        "semantic_change_mask",
        "change_regions",
        "change_statistics",
        "change_caption",
    ],
    "expected_confidence_characteristics": {
        "typical_range": [0.3, 0.95],
        "known_failure_modes": [
            "cloud_occlusion",
            "seasonal_vegetation_false_positives",
            "shadow_artifacts",
            "registration_errors",
        ],
    },
    "computational_cost": {
        "label": config.COMPUTATIONAL_COST_LABEL,
        "flops_estimate": config.COMPUTATIONAL_COST_FLOPS_ESTIMATE,
    },
    "preconditions": [
        "same_crs",
        "sufficient_overlap",
        "spatially_aligned",
    ],
    "postconditions": [
        "output_contains_binary_mask",
        "confidence_in_0_1",
        "regions_georeferenced",
    ],
}


# ── Precondition Validators ──────────────────────────────────────


def validate_preconditions(mc1_profile: Dict) -> Dict:
    """
    Check all MC4B preconditions against the MC1 Structured Input Profile.

    Returns:
        {"passed": True} if all pass, or
        {"passed": False, "status": "PRECONDITION_FAILED",
         "failed": [{"check": ..., "reason": ..., "actual": ...}, ...]}
    """
    failures = []

    # Check image count
    image_count = mc1_profile.get("image_count", 0)
    if image_count != 2:
        failures.append(
            {
                "check": "image_count",
                "reason": f"MC4B requires exactly 2 images, got {image_count}",
                "actual": image_count,
            }
        )
        # If not 2 images, can't check image-level fields
        return {
            "passed": False,
            "status": "PRECONDITION_FAILED",
            "failed": failures,
        }

    img1 = mc1_profile.get("image_1", {})
    img2 = mc1_profile.get("image_2", {})

    # Check same CRS
    crs1 = img1.get("crs")
    crs2 = img2.get("crs")
    if crs1 != crs2 or crs1 is None:
        failures.append(
            {
                "check": "same_crs",
                "reason": f"CRS mismatch: {crs1} vs {crs2}",
                "actual": {"crs_1": crs1, "crs_2": crs2},
            }
        )

    # Check sufficient overlap
    overlap = mc1_profile.get("spatial_overlap")
    if overlap is None or overlap < config.MIN_SPATIAL_OVERLAP:
        failures.append(
            {
                "check": "sufficient_overlap",
                "reason": (
                    f"Spatial overlap {overlap} below threshold "
                    f"{config.MIN_SPATIAL_OVERLAP}"
                ),
                "actual": overlap,
            }
        )

    # Check coregistration
    coreg = mc1_profile.get("coregistration_score")
    if coreg is None or coreg < config.MIN_COREGISTRATION_SCORE:
        failures.append(
            {
                "check": "spatially_aligned",
                "reason": (
                    f"Coregistration score {coreg} below threshold "
                    f"{config.MIN_COREGISTRATION_SCORE}"
                ),
                "actual": coreg,
            }
        )

    if failures:
        return {
            "passed": False,
            "status": "PRECONDITION_FAILED",
            "failed": failures,
        }

    return {"passed": True}


# ── Tool Adapter (main entry point) ─────────────────────────────


class ChangeMambaAdapter:
    """
    Wraps the ChangeMamba/LightweightCNN detector as an MC3-callable tool.

    Usage:
        adapter = ChangeMambaAdapter()
        result = adapter.execute(mc1_profile, t1_tensor, t2_tensor, query)
    """

    def __init__(self):
        backbone = get_backbone(config.BACKBONE)
        self.detector = ChangeDetector(backbone)
        self.detector.eval()

    def execute(
        self,
        mc1_profile: Dict,
        t1: torch.Tensor,
        t2: torch.Tensor,
        query: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> Dict:
        """
        Run the full MC4B pipeline.

        Args:
            mc1_profile: MC1 Structured Input Profile dict.
            t1: (B, C, H, W) image tensor for time 1.
            t2: (B, C, H, W) image tensor for time 2.
            query: User's natural-language query (for captioning).
            seed: Optional random seed for reproducibility.

        Returns:
            MC4B output dict matching the fixed I/O contract, or
            a PRECONDITION_FAILED refusal dict.
        """
        # 1. Validate preconditions — refuse to run on bad input
        precond = validate_preconditions(mc1_profile)
        if not precond["passed"]:
            return precond

        # 2. Set seed for reproducibility
        if seed is not None:
            torch.manual_seed(seed)

        # 3. Run inference
        det_result = self.detector.predict(t1, t2)
        binary_mask = det_result["binary_mask"]
        semantic_logits = det_result["semantic_logits"]
        confidence = det_result["confidence"]

        # 4. Extract change semantics
        semantics = extract_change_types(binary_mask, semantic_logits)
        sem_result = semantics[0] if semantics else {
            "has_change": False,
            "changed_pixel_fraction": 0.0,
            "change_types": [],
            "primary_change": "no_change",
            "description": "No change detected",
        }

        # 5. Generate caption
        caption_result = generate_change_caption(sem_result, query)

        # 6. Localize change regions
        regions = mask_to_regions(binary_mask)

        # 7. Get geo info from MC1 profile
        img1 = mc1_profile.get("image_1", {})
        gsd_m = img1.get("gsd_m", 1.0)
        crs = img1.get("crs", "EPSG:4326")

        # Build a default affine transform from GSD if none provided
        affine = mc1_profile.get("affine_transform")
        if affine is None:
            # Default: origin at (0, 0), GSD as pixel size, no rotation
            affine = [gsd_m, 0, 0, 0, -gsd_m, 0]

        geo_regions = regions_to_geojson(regions, affine, crs, gsd_m)

        # 8. Compute statistics
        stats = compute_change_statistics(binary_mask, gsd_m)

        # 9. Build the fixed I/O contract output
        change_score = sem_result["changed_pixel_fraction"]

        return {
            "status": "SUCCESS",
            "change_map": binary_mask.cpu().numpy().tolist(),
            "binary_change_mask": (binary_mask > 0.5).int().cpu().numpy().tolist(),
            "change_score": round(change_score, 4),
            "changed_region_coordinates": [
                {"geometry": r["geometry"], "crs": r["crs"]}
                for r in geo_regions
            ],
            "change_statistics": {
                "changed_area_m2": stats["changed_area_m2"],
                "changed_pixel_pct": stats["changed_pixel_pct"],
            },
            "model_confidence": round(confidence, 4),
            "semantics": sem_result,
            "caption": caption_result["caption"],
            "source_model": "CHANGE_MAMBA",
        }
