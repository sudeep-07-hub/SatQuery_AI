import copy
from typing import List, Dict, Tuple
from agent.schemas import ToolCall, BindingResult, InputBinding
from qwen.schemas import ObservationRequirement
from mc1.schemas import RequestObservationProfile, ObservationProfile

class ObservationBinder:
    """
    Evaluates whether actual MC1 observations satisfy Phase 2 ObservationRequirements.
    Produces a Bound ToolCall if hard constraints and semantic roles are satisfied.
    """
    
    def __init__(self, assume_upload_order: bool = False):
        # When True and acquisition dates are missing, the first uploaded observation is treated as
        # "before" and the second as "after". The assumption is always reported as a binding warning.
        self.assume_upload_order = assume_upload_order

    def bind(self, call: ToolCall, requirement: ObservationRequirement, profile: RequestObservationProfile) -> BindingResult:
        warnings = list(profile.warnings)
        if profile.compatibility:
            warnings.extend(profile.compatibility.warnings)
            
        failed_constraints = []
        if profile.compatibility and not profile.compatibility.hard_compatible:
            failures = profile.compatibility.hard_failures
            if not failures:
                failures = ["Hard compatibility check failed with no specific reason."]
            failed_constraints.extend(failures)
            
        candidate_ids = [obs.observation_id for obs in profile.observations]

        # 1. Hard Incompatibility check
        if profile.compatibility and not profile.compatibility.hard_compatible:
            return BindingResult(
                status="INCOMPATIBLE",
                requirement_id=requirement.requirement_id,
                failed_constraints=failed_constraints,
                warnings=warnings,
                candidate_observation_ids=candidate_ids
            )
            
        obs_count = len(profile.observations)
        
        # 2. Minimum Observation Count
        if obs_count < requirement.minimum_observations:
            failed_constraints.append(f"Requires at least {requirement.minimum_observations} observations, found {obs_count}")
            return BindingResult(
                status="INSUFFICIENT",
                requirement_id=requirement.requirement_id,
                failed_constraints=failed_constraints,
                warnings=warnings,
                candidate_observation_ids=candidate_ids
            )
            
        # 3. Maximum Observation Count
        if requirement.maximum_observations is not None and obs_count > requirement.maximum_observations:
            # We have extra observations. We need a deterministic rule to narrow down.
            # For this task, if we have more than max and no clear semantic discriminator, it's ambiguous.
            pass
            
        # 4. Filter by Required Modalities
        # A single observation can have a specific modality.
        valid_observations = []
        req_mods = set(requirement.required_modalities)
        if "unspecified" in req_mods:
            req_mods.remove("unspecified")
            
        if not req_mods:
            valid_observations = list(profile.observations)
        else:
            # If multiple modalities are required (e.g. optical, sar) we must have both represented
            found_mods = {obs.sensor.modality for obs in profile.observations}
            missing_mods = req_mods - found_mods
            if missing_mods:
                failed_constraints.append(f"Missing required modalities: {missing_mods}")
                return BindingResult(
                    status="INSUFFICIENT",
                    requirement_id=requirement.requirement_id,
                    failed_constraints=failed_constraints,
                    warnings=warnings,
                    candidate_observation_ids=candidate_ids
                )
            # Filter to only observations that match the required modalities or are unknown if strictly needed, but let's assume they are valid if they fulfill req.
            valid_observations = [obs for obs in profile.observations if obs.sensor.modality in req_mods or not req_mods]

        if not valid_observations:
            failed_constraints.append("No valid observations matching modality constraints")
            return BindingResult(
                status="INSUFFICIENT",
                requirement_id=requirement.requirement_id,
                failed_constraints=failed_constraints,
                warnings=warnings,
                candidate_observation_ids=candidate_ids
            )

        # 5. Spatial Relationship (Shared area / Co-registration)
        alignment = profile.compatibility.factors.get("alignment") if profile.compatibility else None
        if requirement.shared_area_required:
            if profile.spatial_overlap is None and alignment == "assumed_pixel_aligned":
                warnings.append("Shared area assumed: images are not georeferenced but have identical dimensions.")
            elif profile.spatial_overlap is None or profile.spatial_overlap <= 0.0:
                failed_constraints.append("Shared spatial area required but no overlap exists")
                return BindingResult(
                    status="INCOMPATIBLE",
                    requirement_id=requirement.requirement_id,
                    failed_constraints=failed_constraints,
                    warnings=warnings,
                    candidate_observation_ids=candidate_ids
                )
        
        if requirement.co_registration_required:
            if profile.coregistration_score is None or profile.coregistration_score < 0.8:
                failed_constraints.append("Co-registration required but score is insufficient")
                return BindingResult(
                    status="INCOMPATIBLE",
                    requirement_id=requirement.requirement_id,
                    failed_constraints=failed_constraints,
                    warnings=warnings,
                    candidate_observation_ids=candidate_ids
                )

        # 6. Assign Semantic Roles and Bind
        bound_call = call.model_copy(deep=True)
        # Clear existing bindings in the bound call (we will re-populate them with physical IDs)
        bound_call.input_bindings = {}
        
        unresolved_roles = []
        
        if len(req_mods) == 2 and "optical" in req_mods and "sar" in req_mods:
            # Cross-modal fusion
            opt_obs = [obs for obs in valid_observations if obs.sensor.modality == "optical"]
            sar_obs = [obs for obs in valid_observations if obs.sensor.modality == "sar"]
            
            if len(opt_obs) > 1 or len(sar_obs) > 1:
                return BindingResult(
                    status="AMBIGUOUS",
                    requirement_id=requirement.requirement_id,
                    warnings=warnings,
                    candidate_observation_ids=[o.observation_id for o in valid_observations],
                    unresolved_roles=["optical", "sar"]
                )
                
            bound_call.input_bindings["optical"] = InputBinding(
                requirement_id=requirement.requirement_id,
                semantic_role="optical",
                observation_id=opt_obs[0].observation_id
            )
            bound_call.input_bindings["sar"] = InputBinding(
                requirement_id=requirement.requirement_id,
                semantic_role="sar",
                observation_id=sar_obs[0].observation_id
            )
            
        elif requirement.temporal_relationship == "before_after":
            # Change detection / Temporal VQA
            # Needs two observations, and temporal metadata to sort them.
            if len(valid_observations) < 2:
                failed_constraints.append("Temporal relationship 'before_after' requires 2 observations")
                return BindingResult(
                    status="INSUFFICIENT",
                    requirement_id=requirement.requirement_id,
                    failed_constraints=failed_constraints,
                    warnings=warnings,
                    candidate_observation_ids=[o.observation_id for o in valid_observations]
                )
            if len(valid_observations) > 2:
                return BindingResult(
                    status="AMBIGUOUS",
                    requirement_id=requirement.requirement_id,
                    warnings=warnings,
                    candidate_observation_ids=[o.observation_id for o in valid_observations],
                    unresolved_roles=["before", "after"]
                )
                
            obs1 = valid_observations[0]
            obs2 = valid_observations[1]
            
            t1 = obs1.temporal.get("timestamp") if obs1.temporal else None
            t2 = obs2.temporal.get("timestamp") if obs2.temporal else None
            
            if (not t1 or not t2) and self.assume_upload_order:
                before_obs, after_obs = obs1, obs2
                warnings.append(
                    f"Acquisition dates missing; ASSUMED upload order: {obs1.observation_id} = before, "
                    f"{obs2.observation_id} = after."
                )
            elif not t1 or not t2:
                unresolved_roles = ["before", "after"]
                return BindingResult(
                    status="AMBIGUOUS",
                    requirement_id=requirement.requirement_id,
                    warnings=warnings + ["Missing temporal metadata for one or more observations, cannot order before/after."],
                    candidate_observation_ids=[o.observation_id for o in valid_observations],
                    unresolved_roles=unresolved_roles
                )
                
            elif t1 < t2:
                before_obs, after_obs = obs1, obs2
            elif t1 > t2:
                before_obs, after_obs = obs2, obs1
            else:
                unresolved_roles = ["before", "after"]
                return BindingResult(
                    status="AMBIGUOUS",
                    requirement_id=requirement.requirement_id,
                    warnings=warnings + ["Timestamps are identical, cannot determine before/after."],
                    candidate_observation_ids=[o.observation_id for o in valid_observations],
                    unresolved_roles=unresolved_roles
                )
                
            bound_call.input_bindings["before"] = InputBinding(
                requirement_id=requirement.requirement_id,
                semantic_role="before",
                observation_id=before_obs.observation_id
            )
            bound_call.input_bindings["after"] = InputBinding(
                requirement_id=requirement.requirement_id,
                semantic_role="after",
                observation_id=after_obs.observation_id
            )
            
        else:
            # Single observation
            if len(valid_observations) > 1:
                # If we need 1 and have >1, and no deterministic discriminator exists, it's ambiguous
                if requirement.maximum_observations == 1:
                    return BindingResult(
                        status="AMBIGUOUS",
                        requirement_id=requirement.requirement_id,
                        warnings=warnings,
                        candidate_observation_ids=[o.observation_id for o in valid_observations],
                        unresolved_roles=["primary_observation"]
                    )
                # If maximum_observations > 1 is allowed but it's not a specific temporal/cross-modal setup,
                # we just bind them all sequentially or we say ambiguous if we only have one role slot.
                # For this task, let's keep it simple: if single observation expected but multiple exist -> AMBIGUOUS
                return BindingResult(
                    status="AMBIGUOUS",
                    requirement_id=requirement.requirement_id,
                    warnings=warnings,
                    candidate_observation_ids=[o.observation_id for o in valid_observations],
                    unresolved_roles=["primary_observation"]
                )
                
            bound_call.input_bindings["primary"] = InputBinding(
                requirement_id=requirement.requirement_id,
                semantic_role="primary_observation",
                observation_id=valid_observations[0].observation_id
            )
            
        return BindingResult(
            status="SUFFICIENT",
            requirement_id=requirement.requirement_id,
            bound_call=bound_call,
            warnings=warnings,
            candidate_observation_ids=candidate_ids
        )
