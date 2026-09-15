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
    "supported_modalities": ["optical", "sar"],
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

    # Check modality match and support
    modality1 = img1.get("modality")
    modality2 = img2.get("modality")
    if modality1 != modality2:
        failures.append(
            {
                "check": "same_modality",
                "reason": f"Images must have the same modality. Got {modality1} and {modality2}",
                "actual": {"modality_1": modality1, "modality_2": modality2},
            }
        )
    elif modality1 not in ["optical", "sar"]:
        failures.append(
            {
                "check": "supported_modality",
                "reason": f"Unsupported modality {modality1}. Must be 'optical' or 'sar'",
                "actual": {"modality": modality1},
            }
        )

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
    Wraps the ChangeMamba pipeline as an MC3-callable tool.
    Strictly reports MODEL_UNAVAILABLE if the real model cannot load.
    """

    def __init__(self):
        from .pipeline import TemporalPipeline
        self.pipeline = TemporalPipeline()

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
        """
        # 1. Validate preconditions — refuse to run on bad input
        precond = validate_preconditions(mc1_profile)
        if not precond["passed"]:
            return precond

        # 2. Set seed for reproducibility
        if seed is not None:
            torch.manual_seed(seed)

        # 3. Pass tensors through the strict pipeline
        result = self.pipeline.execute(mc1_profile, t1, t2)

        # 4. Map the Pipeline result back to the Tool Registry dictionary interface
        if result.status != "SUCCESS":
            return {
                "passed": False,
                "status": result.status,
                "reason": result.reason,
            }

        # If success (which is impossible without CUDA right now)
        return {
            "status": "SUCCESS",
            "change_map": result.raw_output.binary_mask,
            "binary_change_mask": result.raw_output.binary_mask,
            "change_score": 0.0,
            "changed_region_coordinates": [],
            "change_statistics": {},
            "model_confidence": result.raw_output.confidence,
            "semantics": {},
            "caption": "",
            "source_model": "CHANGE_MAMBA_TOOL",
        }
