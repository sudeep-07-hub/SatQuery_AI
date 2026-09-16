from qwen.schemas import ObservationRequirement
from .schemas import ToolSpec
from .registry import AgentToolRegistry

def _create_single_image_vqa() -> ToolSpec:
    return ToolSpec(
        tool_id="single_image_vqa",
        name="Single Image VQA & Captioning",
        description="Analyzes a single optical or SAR image to answer questions, describe scenes, or ground targets.",
        supported_tasks=["single_image_vqa", "captioning", "grounding"],
        required_observations=ObservationRequirement(
            requirement_id="req_single_image_vqa",
            source_subtasks=[],
            minimum_observations=1,
            required_modalities=["unspecified"],
            temporal_relationship="none"
        ),
        output_types=["textual_answer", "spatial_evidence"],
        status="implemented",
        enabled=True
    )

def _create_temporal_change_analysis() -> ToolSpec:
    return ToolSpec(
        tool_id="temporal_change_analysis",
        name="Temporal Change Analysis",
        description="Analyzes exactly two corresponding observations from different dates to detect and describe changes.",
        supported_tasks=["change_detection", "change_vqa"],
        required_observations=ObservationRequirement(
            requirement_id="req_temporal_change",
            source_subtasks=[],
            minimum_observations=2,
            maximum_observations=2,
            temporal_relationship="before_after",
            corresponding_observations_required=True,
            shared_area_required=True
        ),
        output_types=["change_mask", "change_regions", "change_statistics"],
        status="partial",
        enabled=False
    )

def _create_classical_change_detection() -> ToolSpec:
    return ToolSpec(
        tool_id="classical_change_detection",
        name="Classical Change Detection (log-ratio / CVA)",
        description=(
            "Non-learned bi-temporal change detection on two co-registered observations of the same "
            "modality: SAR log-ratio or optical change vector analysis. Localises WHERE change happened "
            "and reports changed area; does not classify WHAT changed."
        ),
        supported_tasks=["change_detection", "change_vqa"],
        required_observations=ObservationRequirement(
            requirement_id="req_classical_change",
            source_subtasks=[],
            minimum_observations=2,
            maximum_observations=2,
            temporal_relationship="before_after",
            corresponding_observations_required=True,
            shared_area_required=True
        ),
        output_types=["change_mask", "change_regions", "change_statistics"],
        status="implemented",
        enabled=True
    )

def _create_optical_sar_fusion() -> ToolSpec:
    return ToolSpec(
        tool_id="optical_sar_fusion",
        name="Optical-SAR Multimodal Fusion",
        description="Fuses exactly one Optical and one SAR co-registered observation to perform rich multimodal reasoning.",
        supported_tasks=["cross_modal_fusion"],
        required_observations=ObservationRequirement(
            requirement_id="req_fusion",
            source_subtasks=[],
            minimum_observations=2,
            maximum_observations=2,
            required_modalities=["optical", "sar"],
            corresponding_observations_required=True,
            co_registration_required=True,
            shared_area_required=True
        ),
        output_types=["multimodal_interpretation", "spatial_evidence"],
        status="disconnected",
        enabled=False
    )

def setup_default_registry() -> AgentToolRegistry:
    """
    Constructs the standard authoritative Phase 3 Tool Registry.
    Does not load any heavy model weights.
    """
    registry = AgentToolRegistry()
    registry.register(_create_single_image_vqa())
    registry.register(_create_temporal_change_analysis())
    registry.register(_create_classical_change_detection())
    registry.register(_create_optical_sar_fusion())
    return registry
