"""
engine_registry.py — Stage 2: Engine Registry and Precondition Layer

Defines ENGINE_REGISTRY: a mapping of task-type names to their handler stubs,
precondition functions, and human-readable rejection messages.

Preconditions operate on an "adapted" profile view (see adapt_profile()) that
derives convenience fields from the raw Stage 1 Structured Input Profile.

Field mapping (Phase 0 findings):
    Raw Stage 1          →  Adapted View
    ─────────────────────────────────────
    image_count          →  num_images
    image_N.modality     →  modalities (set)
    (derived)            →  same_modality (bool)
    spatial_overlap      →  spatial_overlap (unchanged)
"""


# ── Profile Adapter ───────────────────────────────────────────────

def adapt_profile(profile: dict) -> dict:
    """
    Creates a controller-friendly view of the Stage 1 profile.
    Derives convenience fields for precondition evaluation:
      - num_images: int (from image_count)
      - modalities: set of non-'unknown' modality strings
      - same_modality: bool (True if exactly one distinct known modality)
    """
    view = dict(profile)
    view["num_images"] = profile.get("image_count", 0)

    modalities = set()
    for i in range(1, view["num_images"] + 1):
        img = profile.get(f"image_{i}", {})
        mod = img.get("modality", "unknown")
        if mod and mod != "unknown":
            modalities.add(mod)
    view["modalities"] = modalities
    view["same_modality"] = len(modalities) == 1

    return view


# ── Handler Stubs ─────────────────────────────────────────────────
# Each stub returns {"answer": ..., "confidence": ..., "bbox": ...}.
# TODO: wire to Stage 4 engine


def run_vqa_engine(profile: dict) -> dict:
    """Visual Question Answering stub."""
    # TODO: wire to Stage 4 engine
    return {"answer": "VQA stub answer", "confidence": 0.5, "bbox": None}


def run_caption_engine(profile: dict) -> dict:
    """Image captioning stub."""
    # TODO: wire to Stage 4 engine
    return {"answer": "Caption stub answer", "confidence": 0.5, "bbox": None}


def run_grounding_engine(profile: dict) -> dict:
    """Visual grounding stub."""
    # TODO: wire to Stage 4 engine
    return {
        "answer": "Grounding stub answer",
        "confidence": 0.5,
        "bbox": [10, 20, 100, 200],
    }


def run_change_engine(profile: dict) -> dict:
    """Change detection stub."""
    # TODO: wire to Stage 4 engine
    return {"answer": "Change detection stub answer", "confidence": 0.5, "bbox": None}


def run_fusion_engine(profile: dict) -> dict:
    """Cross-modal fusion stub."""
    # TODO: wire to Stage 4 engine
    return {
        "answer": "Cross-modal fusion stub answer",
        "confidence": 0.5,
        "bbox": None,
    }


# ── Engine Registry ───────────────────────────────────────────────
# Keys: task-type strings produced by classify_task (Phase 2).
# Values: dict with handler, precondition (lambda over adapted profile),
#         and precondition_msg (human-readable rejection reason).

ENGINE_REGISTRY = {
    "single_image_vqa": {
        "handler": run_vqa_engine,
        "precondition": lambda p: p["num_images"] >= 1,
        "precondition_msg": "VQA requires at least one image.",
    },
    "captioning": {
        "handler": run_caption_engine,
        "precondition": lambda p: p["num_images"] >= 1,
        "precondition_msg": "Captioning requires at least one image.",
    },
    "grounding": {
        "handler": run_grounding_engine,
        "precondition": lambda p: p["num_images"] == 1,
        "precondition_msg": "Grounding requires exactly one image.",
    },
    "change_detection": {
        "handler": run_change_engine,
        "precondition": lambda p: (
            p["num_images"] == 2
            and p.get("spatial_overlap") is not None
            and p["spatial_overlap"] > 0.6
            and p["same_modality"]
        ),
        "precondition_msg": (
            "Change detection requires exactly 2 images of the same modality "
            "with >60% spatial overlap."
        ),
    },
    "cross_modal_fusion": {
        "handler": run_fusion_engine,
        "precondition": lambda p: (
            p["num_images"] == 2
            and p.get("spatial_overlap") is not None
            and p["spatial_overlap"] > 0.6
            and p["modalities"] == {"optical", "sar"}
        ),
        "precondition_msg": (
            "Cross-modal fusion requires exactly 2 images (one optical, one SAR) "
            "with >60% spatial overlap."
        ),
    },
}
