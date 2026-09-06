"""
captioner.py — DeltaVLM-style change VQA/caption head (MC4B optional).

Gated behind config.ENABLE_CAPTIONER. When disabled, all functions
return placeholder text. The rest of the pipeline works without it.

To implement the real DeltaVLM head:
    1. Set config.ENABLE_CAPTIONER = True
    2. Implement DeltaVLMCaptioner with a pre-trained instruction-guided
       language model conditioned on the bi-temporal feature difference.
    3. Requires: a frozen LLM backbone (e.g. LLaMA/Vicuna) + a visual
       projection layer from the backbone feature space.
"""

from typing import Optional, Dict
from . import config


def generate_change_caption(
    change_semantics: Dict,
    query: Optional[str] = None,
) -> Dict:
    """
    Generate a natural-language description of the detected change.

    When ENABLE_CAPTIONER is False, returns a template-based description
    derived from the semantics module output.

    Args:
        change_semantics: Output from semantics.extract_change_types()[0]
        query: Optional user query for VQA-style response.

    Returns:
        dict with keys:
            caption: str  — the generated description
            is_stub: bool — True if using template, False if real model
    """
    if config.ENABLE_CAPTIONER:
        # TODO: implement DeltaVLM inference here
        # This would involve:
        # 1. Projecting backbone features into LLM embedding space
        # 2. Conditioning on the bi-temporal difference features
        # 3. Generating text via instruction-guided decoding
        raise NotImplementedError(
            "DeltaVLM captioner not yet implemented. "
            "Set config.ENABLE_CAPTIONER = False to use template-based fallback."
        )

    # Template-based fallback
    description = change_semantics.get("description", "Change analysis complete")
    fraction = change_semantics.get("changed_pixel_fraction", 0.0)
    types = change_semantics.get("change_types", [])

    if not change_semantics.get("has_change", False):
        caption = "No significant change was detected between the two images."
    else:
        pct = round(fraction * 100, 1)
        type_str = ", ".join(t.replace("_", " ") for t in types)
        caption = (
            f"{description}. Approximately {pct}% of the scene has changed. "
            f"Detected change types: {type_str}."
        )

    # If the user asked a specific question, frame the answer accordingly
    if query:
        caption = f"In response to '{query}': {caption}"

    return {
        "caption": caption,
        "is_stub": True,
    }
