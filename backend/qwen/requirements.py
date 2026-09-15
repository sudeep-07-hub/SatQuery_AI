from typing import List, Optional
from .schemas import DecompositionResult, ObservationRequirement, SubtaskSpec

class ObservationRequirementGenerator:
    """
    Translates structured SubtaskSpecs into ObservationRequirements deterministically.
    """
    
    def __init__(self):
        self.req_counter = 0
        
    def _get_next_id(self) -> str:
        self.req_counter += 1
        return f"obs_req_{self.req_counter}"
        
    def _map_single_subtask(self, subtask: SubtaskSpec) -> ObservationRequirement:
        req = ObservationRequirement(
            requirement_id=self._get_next_id(),
            source_subtasks=[subtask.subtask_id],
            minimum_observations=1,
            ambiguous=subtask.ambiguous
        )
        
        # Determine temporal requirements
        if subtask.temporal_requirement == "before_after":
            req.minimum_observations = max(req.minimum_observations, 2)
            req.temporal_relationship = "before_after"
            req.corresponding_observations_required = True
            req.shared_area_required = True
            
        # Specific semantic overrides
        if subtask.primary_task in ("change_detection", "change_vqa"):
            req.minimum_observations = max(req.minimum_observations, 2)
            req.temporal_relationship = "before_after"
            req.corresponding_observations_required = True
            req.shared_area_required = True
            
        elif subtask.primary_task == "cross_modal_fusion":
            req.minimum_observations = max(req.minimum_observations, 2)
            req.required_modalities = ["optical", "sar"]
            req.spatial_relationship = "co_registered"
            req.co_registration_required = True
            req.corresponding_observations_required = True
            req.shared_area_required = True
            
        else:
            # Propagate modalities for standard tasks
            if subtask.required_modalities:
                req.required_modalities = subtask.required_modalities
                
        return req

    def generate(self, decomposition: DecompositionResult) -> List[ObservationRequirement]:
        if not decomposition.subtasks:
            return []
            
        # Pass 1: Map each subtask individually
        individual_reqs = [self._map_single_subtask(st) for st in decomposition.subtasks]
        
        if len(individual_reqs) <= 1:
            return individual_reqs
            
        # Pass 2: Merge compatible requirements (simplified grouping)
        # 1. Group by temporal context (all subtasks sharing before_after should share the same 2-image obs)
        # 2. Group by modality conflicts.
        
        # For our prototype, we will attempt to merge all requirements into a single unified requirement
        # UNLESS they have strictly conflicting required modalities (e.g., optical vs sar).
        
        unified = ObservationRequirement(
            requirement_id=self._get_next_id(),
            source_subtasks=[],
            minimum_observations=1,
            ambiguous=decomposition.ambiguous
        )
        
        modalities = set()
        
        for req in individual_reqs:
            unified.source_subtasks.extend(req.source_subtasks)
            unified.minimum_observations = max(unified.minimum_observations, req.minimum_observations)
            if req.maximum_observations:
                unified.maximum_observations = req.maximum_observations
            
            # Combine modalities
            for mod in req.required_modalities:
                if mod != "unspecified":
                    modalities.add(mod)
            
            # Combine temporal
            if req.temporal_relationship == "before_after":
                unified.temporal_relationship = "before_after"
                
            # Combine spatial
            if req.spatial_relationship == "co_registered":
                unified.spatial_relationship = "co_registered"
            elif req.spatial_relationship == "shared_area" and unified.spatial_relationship != "co_registered":
                unified.spatial_relationship = "shared_area"
                
            unified.corresponding_observations_required = unified.corresponding_observations_required or req.corresponding_observations_required
            unified.co_registration_required = unified.co_registration_required or req.co_registration_required
            unified.shared_area_required = unified.shared_area_required or req.shared_area_required
            unified.ambiguous = unified.ambiguous or req.ambiguous

        # If they explicitly requested different modalities across different subtasks but did NOT request fusion, 
        # it is a conflict and we should return the individual reqs instead of merging.
        # But if the decomposition actually requested cross_modal_fusion, it will naturally have both optical and sar.
        # We can check if any individual task had cross_modal_fusion.
        has_fusion_intent = any(st.primary_task == "cross_modal_fusion" for st in decomposition.subtasks)
        
        if "optical" in modalities and "sar" in modalities and not has_fusion_intent:
            # We have an explicit conflict that was not fusion. (e.g. caption optical AND caption SAR).
            # We preserve them as separate independent requirements.
            return individual_reqs
            
        if modalities:
            unified.required_modalities = sorted(list(modalities))
        else:
            unified.required_modalities = ["unspecified"]

        return [unified]
