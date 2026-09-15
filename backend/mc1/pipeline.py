from typing import List
from fastapi import UploadFile
from .format_validator import validate_format
from .metadata_extractor import extract_metadata
from .classifier import EvidenceBasedClassifier, assess_quality, assess_conditions
from .spatial_analyzer import verify_crs, get_footprint, calculate_overlap, calculate_coregistration
from .temporal_analyzer import determine_temporal_relationship
from .schemas import (
    ObservationProfile,
    SpatialProfile,
    SensorProfile,
    QualityProfile,
    ConditionProfile,
    RequestObservationProfile,
)

async def run_mc1_pipeline(files: List[UploadFile], query: str) -> dict:
    """
    Executes the MC1 validation pipeline.
    Constructs a strongly typed ObservationProfile and returns its legacy dictionary representation.
    """
    warnings = []
    image_count = len(files)
    
    classifier = EvidenceBasedClassifier()
    metas = []
    footprints = []
    modalities = []
    observations = []
    
    for idx, file in enumerate(files):
        img_key = f"image_{idx+1}"
        file_bytes = await file.read()
        await file.seek(0)
        
        # 1 & 2: Format Validation & Magic Bytes
        is_valid, err_reason, detected_fmt = validate_format(file.filename, file_bytes)
        if not is_valid:
            warnings.append(f"{file.filename}: {err_reason}")
            obs = ObservationProfile(
                observation_id=f"obs_{idx+1}",
                file_source=file.filename,
                file_format="unknown",
                spatial=SpatialProfile(),
                sensor=SensorProfile(modality="unknown", sensor="unknown"),
                quality=QualityProfile(score=0.0),
                conditions=ConditionProfile()
            )
            observations.append(obs)
            metas.append({})
            footprints.append(None)
            modalities.append("unknown")
            continue
            
        # 3: Metadata Extraction
        meta = extract_metadata(file_bytes, detected_fmt)
        metas.append(meta)
        
        # 4: Classifier
        modality, sensor, conf, source = classifier.classify(meta, file.filename)
        modalities.append(modality)
        
        # 11: Quality & Conditions
        quality_prof = assess_quality(meta, modality)
        cond_prof = assess_conditions(meta, modality)
        
        # Build spatial profile
        footprint = get_footprint(meta)
        footprints.append(footprint)
        
        spatial_prof = SpatialProfile(
            crs=meta.get("crs"),
            bounds=meta.get("bounds"),
            footprint=footprint,
            width=meta.get("width"),
            height=meta.get("height"),
            gsd_m=round(meta.get("gsd_m"), 2) if meta.get("gsd_m") else None,
            transform=meta.get("transform")
        )
        
        sensor_prof = SensorProfile(
            modality=modality,
            sensor=sensor,
            bands=None,  # Can be mapped if needed later
            identification_confidence=conf,
            identification_source=source
        )
        
        temporal_prof = None
        if meta.get("acquisition_date"):
            temporal_prof = {"timestamp": str(meta.get("acquisition_date"))}
        
        obs = ObservationProfile(
            observation_id=f"obs_{idx+1}",
            file_source=file.filename,
            file_format=detected_fmt,
            spatial=spatial_prof,
            temporal=temporal_prof,
            sensor=sensor_prof,
            quality=quality_prof,
            conditions=cond_prof
        )
        observations.append(obs)
        
        if not meta.get("crs"):
            warnings.append(f"{file.filename}: Missing CRS/georeferencing.")
            
        if not meta.get("acquisition_date"):
            warnings.append(f"{file.filename}: Missing acquisition date.")

    # Multi-image specifics
    spatial_overlap = None
    coregistration_score = None
    relationship = "unknown"

    if image_count == 2:
        # 5: CRS Verification
        crs_match, crs_warn = verify_crs(metas[0], metas[1])
        if not crs_match:
            warnings.append(crs_warn)
            
        # 8 & 9: Overlap & Co-registration
        if crs_match and footprints[0] and footprints[1]:
            overlap = calculate_overlap(footprints[0], footprints[1])
            coreg = calculate_coregistration(footprints[0], footprints[1])
            spatial_overlap = round(overlap, 2) if overlap is not None else None
            coregistration_score = coreg
            
            if spatial_overlap is not None and spatial_overlap <= 0.0:
                warnings.append("Images have no spatial overlap. Cannot process as a pair.")
        else:
            warnings.append("Could not compute spatial overlap/co-registration due to missing CRS or footprints.")
            spatial_overlap = 1.0
            coregistration_score = 1.0
            
        # 10: Temporal Relationship
        rel = determine_temporal_relationship(metas[0], metas[1], modalities[0], modalities[1])
        relationship = rel
    elif image_count == 1:
        relationship = "single_image"

    # Calculate Compatibility
    from .compatibility import assess_compatibility
    comp_prof = assess_compatibility(
        observations=observations,
        spatial_overlap=spatial_overlap,
        coreg_score=coregistration_score,
        relationship=relationship
    )
    
    # Merge warnings
    warnings.extend(comp_prof.warnings)
    warnings.extend(comp_prof.hard_failures)
    
    # task_executable becomes an alias for hard_compatible
    task_executable = comp_prof.hard_compatible
    
    request_profile = RequestObservationProfile(
        observations=observations,
        spatial_overlap=spatial_overlap,
        coregistration_score=coregistration_score,
        relationship=relationship,
        task_executable=task_executable,
        warnings=warnings,
        compatibility=comp_prof
    )
    
    return request_profile.to_legacy_dict(query)
