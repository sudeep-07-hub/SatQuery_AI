from typing import List
from .schemas import ObservationProfile, CompatibilityProfile

def assess_compatibility(
    observations: List[ObservationProfile], 
    spatial_overlap: float, 
    coreg_score: float, 
    relationship: str
) -> CompatibilityProfile:
    """
    Evaluates Hard Compatibility and Soft Suitability for a request profile.
    Returns a CompatibilityProfile containing the results.
    """
    comp = CompatibilityProfile()
    
    # --- 1. Hard Compatibility Checks ---
    # Invalid formats
    for idx, obs in enumerate(observations):
        if obs.file_format == "unknown":
            comp.hard_compatible = False
            comp.hard_failures.append(f"Observation {idx+1} ({obs.file_source}) has an unknown or invalid file format.")
            
    # Spatial Overlap
    if len(observations) == 2 and spatial_overlap is not None:
        if spatial_overlap <= 0.0:
            comp.hard_compatible = False
            comp.hard_failures.append("No spatial overlap between observation footprints. Cannot process as a spatial pair.")

    # --- 2. Soft Suitability Checks ---
    score = 1.0
    
    # Partial Overlap
    if len(observations) == 2 and spatial_overlap is not None:
        if 0.0 < spatial_overlap < 0.95:
            comp.warnings.append(f"Partial spatial overlap ({spatial_overlap*100:.1f}%).")
            
    # Coregistration
    if len(observations) == 2 and coreg_score is not None:
        if coreg_score < 0.8:
            score -= 0.1
            comp.warnings.append(f"Weak coregistration proxy score ({coreg_score}).")

    # Observation specific soft factors
    for idx, obs in enumerate(observations):
        # Quality
        if obs.quality.score < 0.8:
            score -= (0.8 - obs.quality.score) * 0.5
            comp.warnings.append(f"Observation {idx+1} quality is degraded ({obs.quality.score}).")
            
        # Conditions
        if obs.conditions:
            # NoData
            if obs.conditions.nodata_fraction > 0.05:
                score -= 0.1
                comp.warnings.append(f"Observation {idx+1} has significant NoData ({obs.conditions.nodata_fraction*100:.1f}%).")
            # Cloud
            if obs.conditions.cloud_status == "present":
                score -= 0.2
                comp.warnings.append(f"Observation {idx+1} is cloud contaminated.")
            elif obs.conditions.cloud_status == "unknown" and obs.sensor.modality == "optical":
                score -= 0.05
                comp.warnings.append(f"Observation {idx+1} cloud status is unknown.")
                
    comp.soft_suitability_score = max(0.0, min(1.0, round(score, 2)))
    
    # --- 3. Provenance ---
    comp.factors = {
        "spatial_overlap": spatial_overlap,
        "coregistration": coreg_score,
        "relationship": relationship
    }
    
    return comp
