import random

class SensorClassifier:
    def classify(self, meta: dict, filename: str) -> tuple[str, str, str, str]:
        """Returns (modality, sensor, confidence, source)"""
        raise NotImplementedError

class EvidenceBasedClassifier(SensorClassifier):
    def classify(self, meta: dict, filename: str) -> tuple[str, str, str, str]:
        modality = "unknown"
        sensor = "unknown"
        confidence = "unknown"
        source = "unknown"
        
        tags = meta.get("tags", {})
        descriptions = meta.get("descriptions", [])
        colorinterp = meta.get("colorinterp", [])
        bands = meta.get("band_count")
        
        # 1. Explicit Metadata (High Confidence)
        if tags:
            tag_str = str(tags).lower()
            if "sensor" in tag_str or "model" in tag_str or "platform" in tag_str:
                if "sentinel-1" in tag_str or "s1" in tag_str:
                    return "sar", "Sentinel-1", "high", "metadata"
                if "sentinel-2" in tag_str or "s2" in tag_str:
                    return "optical", "Sentinel-2", "high", "metadata"
                if "landsat" in tag_str or "lc08" in tag_str:
                    return "optical", "Landsat", "high", "metadata"
                # If explicit metadata exists but we don't recognize the specific string, 
                # we still know it has metadata, but we'll fall back to lower priority checks.

        # 2. Raster Properties (High to Medium Confidence)
        is_sar_desc = False
        for desc in descriptions:
            if desc and desc.lower() in ["vv", "vh", "hh", "hv"]:
                is_sar_desc = True
                break
                
        is_opt_color = False
        for color in colorinterp:
            if color and color.lower() in ["red", "green", "blue", "nir"]:
                is_opt_color = True
                break

        if is_sar_desc:
            return "sar", "unknown", "high", "raster_properties"
        if is_opt_color:
            return "optical", "unknown", "high", "raster_properties"
            
        # 3. Raster Characteristics (Low Confidence)
        if bands:
            if bands >= 3:
                return "optical", "unknown", "low", "raster_properties"
            elif bands == 1 or bands == 2:
                dtypes = meta.get("dtypes", [])
                if dtypes and any("float" in d.lower() for d in dtypes):
                    # Usually SAR imagery is distributed as float, but this is a weak heuristic
                    return "sar", "unknown", "low", "raster_properties"

        # 4. Filename Heuristic (Low Confidence)
        fname_lower = filename.lower()
        if fname_lower.startswith("s1a_") or fname_lower.startswith("s1b_"):
            return "sar", "Sentinel-1", "low", "filename_heuristic"
        if fname_lower.startswith("s2a_") or fname_lower.startswith("s2b_"):
            return "optical", "Sentinel-2", "low", "filename_heuristic"
        if fname_lower.startswith("lc08_"):
            return "optical", "Landsat 8", "low", "filename_heuristic"
            
        # 5. Unknown
        return modality, sensor, confidence, source


from .schemas import QualityProfile, ConditionProfile

def assess_conditions(meta: dict, modality: str) -> ConditionProfile:
    """
    Evaluates NoData and Cloud conditions deterministically.
    Returns a ConditionProfile.
    """
    stats = meta.get("image_statistics", {})
    cond = ConditionProfile()
    
    if stats:
        valid_fraction = stats.get("valid_fraction", 1.0)
        cond.nodata_fraction = 1.0 - valid_fraction
        if cond.nodata_fraction > 0.01:
            cond.nodata_status = "present"
        else:
            cond.nodata_status = "clear"
        cond.nodata_source = "dataset_mask"
        
    if modality == "optical":
        cloud_frac = meta.get("cloud_fraction")
        if cloud_frac is not None:
            cond.cloud_fraction = cloud_frac
            cond.cloud_status = "present" if cloud_frac > 0.05 else "clear"
            cond.cloud_source = "product_metadata"
        else:
            cond.cloud_status = "unknown"
            cond.cloud_source = "unknown"
    elif modality == "sar":
        cond.cloud_status = "not_applicable"
        cond.cloud_source = "not_applicable"
        
    return cond


def assess_quality(meta: dict, modality: str) -> QualityProfile:
    """
    Deterministic image quality assessment.
    Uses image_statistics computed during metadata extraction.
    """
    stats = meta.get("image_statistics", {})
    
    if not stats:
        return QualityProfile(
            score=0.0,
            status="unknown",
            source="current_mc1",
            reasons=["Could not compute image statistics (missing or unreadable raster)"]
        )
        
    std = stats.get("std", 0.0)
    valid_fraction = stats.get("valid_fraction", 0.0)
    
    reasons = []
    
    if valid_fraction < 0.10:
        reasons.append("Very few valid pixels (< 10%)")
        return QualityProfile(score=0.2, status="poor", source="current_mc1", metrics=stats, reasons=reasons)
        
    if std <= 1e-3:
        reasons.append("Low information content (constant image)")
        return QualityProfile(score=0.1, status="poor", source="current_mc1", metrics=stats, reasons=reasons)
        
    if std < 2.0:
        reasons.append("Low dynamic range / variance")
        return QualityProfile(score=0.6, status="degraded", source="current_mc1", metrics=stats, reasons=reasons)
        
    reasons.append("Nominal variance and readability")
    return QualityProfile(score=0.9, status="good", source="current_mc1", metrics=stats, reasons=reasons)

