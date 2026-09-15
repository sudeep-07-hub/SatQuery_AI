from pydantic import BaseModel, Field, model_validator
from typing import List, Literal, Optional

class TaskSpec(BaseModel):
    """
    Canonical intermediate representation of the user's analytical intent.
    Describes WHAT is requested, not HOW it is executed.
    """
    query: str = Field(description="The original user query.")
    primary_task: Literal[
        "single_image_vqa", 
        "captioning", 
        "grounding", 
        "change_detection", 
        "change_vqa",
        "cross_modal_fusion", 
        "unknown"
    ]
    target_entities: List[str] = Field(
        default_factory=list, 
        description="Specific objects or features mentioned in the query."
    )
    required_modalities: List[Literal["optical", "sar", "unspecified"]] = Field(
        default_factory=lambda: ["unspecified"]
    )
    temporal_requirement: Optional[Literal["before_after", "none", "unknown"]] = "none"
    spatial_output_required: bool = False
    textual_output_required: bool = True
    ambiguous: bool = Field(
        default=False, 
        description="True if the query intent is unclear, self-contradictory, or lacks necessary details."
    )
    
    @model_validator(mode="after")
    def validate_semantic_consistency(self):
        # Enforce intent-side consistency
        
        # Change detection and change VQA imply temporal before/after
        if self.primary_task in ("change_detection", "change_vqa"):
            if self.temporal_requirement == "none":
                self.ambiguous = True
                
        # Cross modal fusion implies multi-modality or optical+sar intent
        if self.primary_task == "cross_modal_fusion":
            if len(self.required_modalities) == 1 and self.required_modalities[0] != "unspecified":
                self.ambiguous = True
                
        # Grounding mechanically implies spatial output
        if self.primary_task == "grounding":
            self.spatial_output_required = True

        return self


class SubtaskSpec(BaseModel):
    """
    Representation of an atomic analytical step derived from a complex user query.
    """
    subtask_id: str = Field(description="Unique identifier for the subtask, e.g., 'subtask_1'")
    description: str = Field(description="Natural language description of this subtask's objective")
    primary_task: Literal[
        "single_image_vqa", 
        "captioning", 
        "grounding", 
        "change_detection", 
        "change_vqa",
        "cross_modal_fusion", 
        "unknown"
    ]
    target_entities: List[str] = Field(default_factory=list)
    required_modalities: List[Literal["optical", "sar", "unspecified"]] = Field(default_factory=lambda: ["unspecified"])
    temporal_requirement: Optional[Literal["before_after", "none", "unknown"]] = "none"
    spatial_output_required: bool = False
    textual_output_required: bool = True
    depends_on: List[str] = Field(default_factory=list, description="List of subtask_ids this subtask depends on")
    ambiguous: bool = False

    @model_validator(mode="after")
    def validate_semantic_consistency(self):
        # Enforce intent-side consistency identical to TaskSpec
        if self.primary_task in ("change_detection", "change_vqa"):
            if self.temporal_requirement == "none":
                self.ambiguous = True
                
        if self.primary_task == "cross_modal_fusion":
            if len(self.required_modalities) == 1 and self.required_modalities[0] != "unspecified":
                self.ambiguous = True
                
        if self.primary_task == "grounding":
            self.spatial_output_required = True

        return self

class DecompositionResult(BaseModel):
    """
    Result of the query decomposition layer.
    """
    original_query: str
    primary_task_spec: TaskSpec
    is_compound: bool
    subtasks: List[SubtaskSpec]
    ambiguous: bool = False

class ObservationRequirement(BaseModel):
    """
    Explicit, machine-readable requirement describing WHAT observations are needed
    to satisfy a set of analytical subtasks.
    """
    requirement_id: str
    source_subtasks: List[str] = Field(description="Subtask IDs this requirement originates from")
    minimum_observations: int
    maximum_observations: Optional[int] = None
    required_modalities: List[Literal["optical", "sar", "unspecified"]] = Field(default_factory=lambda: ["unspecified"])
    temporal_relationship: Optional[Literal["before_after", "none", "unknown"]] = "none"
    corresponding_observations_required: bool = False
    spatial_relationship: Optional[Literal["co_registered", "shared_area", "none"]] = "none"
    co_registration_required: bool = False
    shared_area_required: bool = False
    ambiguous: bool = False
