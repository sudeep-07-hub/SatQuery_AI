import logging
from typing import Dict, Optional
from PIL import Image

from .paligemma_adapter import PaliGemmaVQAAdapter
from .mock_adapter import MockPaliGemmaVQAAdapter

logger = logging.getLogger(__name__)


class PaliGemmaVQASpecialist:
    """
    MC4A Contract-Compliance Wrapper for the PaliGemma VQA Engine.
    
    Converts inputs from MC1/MC2 into a format the adapter understands,
    applies quality gates and modality penalties, and ensures the output
    strictly adheres to the MC4A contract schema.
    """

    def __init__(self, quality_threshold: float = 0.5, execution_mode: str = "real"):
        self.quality_threshold = quality_threshold
        self.execution_mode = execution_mode
        self.is_mock = False
        
        if execution_mode == "fixture":
            self.adapter = MockPaliGemmaVQAAdapter()
            self.is_mock = True
        else:
            try:
                self.adapter = PaliGemmaVQAAdapter()
            except Exception as e:
                logger.error(f"Failed to load real PaliGemma adapter ({e}).")
                raise RuntimeError(f"MODEL_UNAVAILABLE: {e}")

    def run(
        self,
        image_input,
        query: str,
        mc1_profile: Dict,
        mc2_task_spec: Dict
    ) -> Dict:
        """
        Execute the VQA adapter, conforming strictly to the MC4A output contract.
        
        Args:
            image_input: Path string or PIL.Image.
            query: User's natural-language query.
            mc1_profile: MC1 Structured Input Profile.
            mc2_task_spec: MC2 Structured Task Specification.
            
        Returns:
            Dict conforming to the MC4A output contract.
        """
        # Resolve target image info from mc1_profile (handles ambiguity_flag if set)
        img_meta = mc1_profile.get("image_1", {})
        
        # Initialize output contract with defaults
        output = {
            "textual_answer": None,
            "caption": None,
            "detected_entities": None,
            "bounding_boxes": None,
            "segmentation_masks": None,
            "model_confidences": None,
            # PaliGemma answers about the whole image; the only honest spatial support is the
            # image footprint (WGS84) when the caller could compute one, otherwise none.
            "spatial_evidence": mc1_profile.get("footprint", None),
            "domain_mismatch_flag": False,
        }

        # Quality Gate
        # Assuming quality object has optical or sar keys
        quality_score = 1.0
        if "quality" in mc1_profile:
            modality = img_meta.get("modality", "optical")
            quality_score = mc1_profile["quality"].get(modality, 1.0)
            
        if quality_score < self.quality_threshold:
            output["blocked_reason"] = "input_quality_below_threshold"
            return output

        # Ensure image is valid before inference (handled by adapter now, which supports SAR)

        # Modality Guard
        modality = img_meta.get("modality", "optical")
        if modality != "optical":
            output["domain_mismatch_flag"] = True

        # Check if caption is requested
        primary_task = mc2_task_spec.get("primary_task", "")
        caption_requested = "caption" in primary_task.lower()

        # Run inference
        try:
            adapter_res = self.adapter.predict(image_input, query, task="caption" if caption_requested else "vqa")
        except Exception as e:
            output["blocked_reason"] = f"inference_failed: {e}"
            return output
            
        confidence = adapter_res.get("confidence", 0.0)
        
        # Apply modality penalty if needed
        if output["domain_mismatch_flag"]:
            confidence *= 0.3

        if caption_requested:
            output["caption"] = adapter_res.get("text")
        else:
            output["textual_answer"] = adapter_res.get("text")
            
        output["model_confidences"] = {"paligemma_vqa": confidence}

        return output
