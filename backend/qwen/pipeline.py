import re
from typing import List, Optional
from pydantic import BaseModel
from .schemas import TaskSpec, SubtaskSpec, ObservationRequirement, DecompositionResult
from .extractor import TaskSpecExtractor
from .decomposer import QueryDecomposer
from .requirements import ObservationRequirementGenerator
from .inference import Qwen3Inference

class QueryIntelligenceResult(BaseModel):
    """
    Final Output of Phase 2 Query Intelligence Layer.
    Contains the extracted primary intent, semantic decomposition, 
    and the explicit observation requirements necessary to fulfill the query.
    """
    original_query: str
    primary_task_spec: TaskSpec
    is_compound: bool
    subtasks: List[SubtaskSpec]
    observation_requirements: List[ObservationRequirement]
    ambiguous: bool

MODALITY_TERMS = {
    "optical": ("optical", "multispectral", "rgb", "true colour", "true color", "sentinel-2", "sentinel 2", "landsat"),
    "sar": ("sar", "radar", "sentinel-1", "sentinel 1", "backscatter", "polarization", "polarisation"),
}


def ground_modalities(query: str, modalities: List[str]) -> List[str]:
    """
    Keep only modality constraints the user actually expressed. The LLM tends to assume "optical"
    for generic questions ("Describe this image."), which would wrongly reject SAR or unlabelled uploads.
    """
    q = query.lower()
    grounded = [
        m for m in modalities
        if m in MODALITY_TERMS
        and any(re.search(rf"\b{re.escape(term)}\b", q) for term in MODALITY_TERMS[m])
    ]
    return grounded or ["unspecified"]


class QueryIntelligencePipeline:
    """
    End-to-end Phase 2 pipeline.
    Transforms a natural language query into formal analytical subtasks and observation requirements.
    """
    def __init__(self, inference_engine: Qwen3Inference):
        self.extractor = TaskSpecExtractor(inference_engine)
        self.decomposer = QueryDecomposer(inference_engine)
        self.req_generator = ObservationRequirementGenerator()

    def process_query(self, query: str) -> QueryIntelligenceResult:
        # Step 1: Extraction & Classification (Tasks 2.2 & 2.3)
        task_spec = self.extractor.extract(query)
        
        # Step 2: Query Decomposition (Task 2.4)
        decomp_result = self.decomposer.decompose(query, task_spec)

        # Modality constraints must come from the query, not from the model's assumptions.
        # (Fusion keeps optical+SAR: the requirement generator sets both for that task.)
        decomp_result.primary_task_spec.required_modalities = ground_modalities(
            query, decomp_result.primary_task_spec.required_modalities
        )
        for subtask in decomp_result.subtasks:
            if subtask.primary_task != "cross_modal_fusion":
                subtask.required_modalities = ground_modalities(query, subtask.required_modalities)

        # Step 3: Observation Requirements Generation (Task 2.5)
        obs_reqs = self.req_generator.generate(decomp_result)
        
        # Final aggregation
        return QueryIntelligenceResult(
            original_query=query,
            primary_task_spec=decomp_result.primary_task_spec,
            is_compound=decomp_result.is_compound,
            subtasks=decomp_result.subtasks,
            observation_requirements=obs_reqs,
            ambiguous=decomp_result.ambiguous
        )
