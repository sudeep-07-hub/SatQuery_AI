"""
pipeline.py — The strict Temporal Inference Pipeline boundary.

Responsible for:
- Resolving chronological order using MC1 acquisition metadata.
- Orchestrating safe raster loading and tensor formatting.
- Explicitly detecting structural blockers (like CUDA missing for ChangeMamba).
- Constructing TemporalResult.
"""

from typing import Dict, Any, Tuple, Optional
import torch

from mc1.temporal_analyzer import determine_temporal_relationship
from .result import TemporalResult, RawTemporalOutput
from .backbone import get_backbone


class TemporalPipeline:
    """
    Coordinates the bi-temporal flow strictly preventing fake outputs
    or hidden model stubs if the environment lacks CUDA/mamba-ssm.
    """

    def __init__(self):
        # We explicitly request 'changemamba' and catch the expected NotImplementedError
        # generated if mamba-ssm is unavailable.
        self.model_unavailable = False
        self.unavailable_reason = ""
        self.model = None

        try:
            self.model = get_backbone("changemamba")
        except NotImplementedError as e:
            self.model_unavailable = True
            self.unavailable_reason = str(e)
        except Exception as e:
            self.model_unavailable = True
            self.unavailable_reason = f"Unexpected model load failure: {e}"

    def _determine_chronological_order(self, img1: Dict, img2: Dict) -> Tuple[Optional[Dict], Optional[Dict]]:
        """
        Extract exact chronological order based on `acquisition_date`.
        Returns (earlier, later) or (None, None) if undecidable.
        """
        # If dates exist
        d1 = img1.get("acquisition_date")
        d2 = img2.get("acquisition_date")

        if d1 and d2:
            if d1 < d2:
                return img1, img2
            elif d2 < d1:
                return img2, img1
        
        # Undecidable
        return None, None

    def _build_model_unavailable_result(self) -> TemporalResult:
        return TemporalResult(
            status="MODEL_UNAVAILABLE",
            reason=self.unavailable_reason,
            model_id="ChangeMamba",
            model_revision="official_repo"
        )

    def execute(
        self,
        mc1_profile: Dict,
        t1_tensor: torch.Tensor,
        t2_tensor: torch.Tensor
    ) -> TemporalResult:
        """
        Main pipeline execution entrypoint.
        """
        # 1. Structural Availability Check
        if self.model_unavailable:
            return self._build_model_unavailable_result()

        # 2. Extract specific images and dates
        img1 = mc1_profile.get("image_1", {})
        img2 = mc1_profile.get("image_2", {})
        
        # We need their IDs for the final result mapping.
        # But wait, mc1_profile doesn't inherently give IDs directly in image_1 dict natively 
        # (it gives modality, sensor, gsd, crs, filename). We can use filename as proxy ID.
        filename1 = img1.get("filename", "unknown_1")
        filename2 = img2.get("filename", "unknown_2")

        # In a real scenario we'd sort by date
        # But if the pipeline reaches here, we assume we have a real model
        # 3. Dummy execution if the real model actually loaded (which it won't on MPS)
        # We enforce chronological order
        earlier, later = self._determine_chronological_order(img1, img2)
        if not earlier or not later:
            return TemporalResult(
                status="TEMPORAL_ORDER_UNAVAILABLE",
                reason="Cannot determine explicit chronological ordering from provided metadata."
            )

        # 4. Invoke model ...
        # (This block is unreachable in current env, but establishes the contract)
        with torch.no_grad():
            pass

        return TemporalResult(
            status="INFERENCE_FAILED",
            reason="Placeholder. Should not reach here without a real model."
        )
