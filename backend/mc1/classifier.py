import random

class SensorClassifier:
    def classify(self, meta: dict, filename: str) -> tuple[str, str]:
        raise NotImplementedError

class HeuristicClassifier(SensorClassifier):
    def classify(self, meta: dict, filename: str) -> tuple[str, str]:
        modality = "unknown"
        sensor = "unknown"
        
        # Simple heuristic based on band count and descriptions
        bands = meta.get("band_count")
        descriptions = meta.get("descriptions", [])
        if bands:
            if bands >= 3:
                modality = "optical"
                sensor = "Generic Optical"
            elif bands == 1 or bands == 2:
                # Check for SAR specific polarization descriptions
                is_sar = False
                for desc in descriptions:
                    if desc and desc.lower() in ["vv", "vh", "hh", "hv"]:
                        is_sar = True
                        break
                        
                if is_sar:
                    modality = "sar"
                    sensor = f"SAR ({bands}-Pol)"
                else:
                    dtypes = meta.get("dtypes", [])
                    if dtypes and any("float" in d.lower() for d in dtypes):
                        modality = "sar"
                        sensor = "Generic SAR"
                    else:
                        modality = "optical"
                        sensor = "Panchromatic" if bands == 1 else "Unknown"
                
        return modality, sensor

def assess_quality(meta: dict, modality: str) -> float:
    """
    Placeholder for image quality assessment.
    Returns a float between 0 and 1.
    """
    # For now, return a deterministic-ish heuristic value based on bands or just random fallback.
    # In reality, this would evaluate cloud cover, contrast, noise, etc.
    return round(random.uniform(0.7, 0.95), 2)
