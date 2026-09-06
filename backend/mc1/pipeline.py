from typing import List
from fastapi import UploadFile
from .format_validator import validate_format
from .metadata_extractor import extract_metadata
from .classifier import HeuristicClassifier, assess_quality
from .spatial_analyzer import verify_crs, get_footprint, calculate_overlap, calculate_coregistration
from .temporal_analyzer import determine_temporal_relationship

async def run_mc1_pipeline(files: List[UploadFile], query: str) -> dict:
    """
    Executes the 12-step MC1 validation pipeline.
    """
    warnings = []
    image_count = len(files)
    
    # Structure setup
    profile = {
        "image_count": image_count,
        "query": query,
        "spatial_overlap": None,
        "coregistration_score": None,
        "relationship": "unknown",
        "quality": {},
        "warnings": [],
        "task_executable": False
    }
    
    classifier = HeuristicClassifier()
    metas = []
    footprints = []
    modalities = []
    
    for idx, file in enumerate(files):
        img_key = f"image_{idx+1}"
        file_bytes = await file.read()
        await file.seek(0) # reset for potential later use
        
        # 1 & 2: Format Validation & Magic Bytes
        is_valid, err_reason, detected_fmt = validate_format(file.filename, file_bytes)
        if not is_valid:
            warnings.append(f"{file.filename}: {err_reason}")
            profile[img_key] = {
                "filename": file.filename,
                "modality": "unknown",
                "sensor": "unknown",
                "gsd_m": None,
                "crs": None
            }
            metas.append({})
            footprints.append(None)
            modalities.append("unknown")
            continue
            
        # 3: Metadata Extraction
        meta = extract_metadata(file_bytes, detected_fmt)
        metas.append(meta)
        
        # 4: Classifier
        modality, sensor = classifier.classify(meta, file.filename)
        modalities.append(modality)
        
        # 11: Quality
        q_score = assess_quality(meta, modality)
        profile["quality"][modality if modality != "unknown" else img_key] = q_score
        
        # Build image object
        profile[img_key] = {
            "filename": file.filename,
            "modality": modality,
            "sensor": sensor,
            "gsd_m": round(meta.get("gsd_m"), 2) if meta.get("gsd_m") else None,
            "crs": meta.get("crs"),
            "acquisition_date": meta.get("acquisition_date")
        }
        
        if not meta.get("crs"):
            warnings.append(f"{file.filename}: Missing CRS/georeferencing.")
            
        if not meta.get("acquisition_date"):
            warnings.append(f"{file.filename}: Missing acquisition date.")
            
        footprints.append(get_footprint(meta))

    # Multi-image specifics
    if image_count == 2:
        # 5: CRS Verification
        crs_match, crs_warn = verify_crs(metas[0], metas[1])
        if not crs_match:
            warnings.append(crs_warn)
            
        # 8 & 9: Overlap & Co-registration
        if crs_match and footprints[0] and footprints[1]:
            overlap = calculate_overlap(footprints[0], footprints[1])
            coreg = calculate_coregistration(footprints[0], footprints[1])
            profile["spatial_overlap"] = round(overlap, 2) if overlap is not None else None
            profile["coregistration_score"] = coreg
        else:
            warnings.append("Could not compute spatial overlap/co-registration due to missing CRS or footprints.")
            
        # 10: Temporal Relationship
        rel = determine_temporal_relationship(metas[0], metas[1], modalities[0], modalities[1])
        profile["relationship"] = rel
    elif image_count == 1:
        profile["relationship"] = "single_image"

    profile["warnings"] = warnings
    
    # task_executable should be false if there are fatal warnings.
    # Missing acquisition date is a non-blocking warning.
    fatal_warnings = [w for w in warnings if "Missing acquisition date" not in w]
    profile["task_executable"] = len(fatal_warnings) == 0
    
    return profile
