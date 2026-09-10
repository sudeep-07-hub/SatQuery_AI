from datetime import datetime

def determine_temporal_relationship(meta1: dict, meta2: dict, mod1: str, mod2: str) -> str:
    """
    Step 10: Determine temporal relationship.
    Returns "same_date", "multi_temporal", "cross_modal", or "unknown".
    """
    if mod1 != "unknown" and mod2 != "unknown" and mod1 != mod2:
        return "cross_modal"
        
    date1_str = meta1.get("acquisition_date")
    date2_str = meta2.get("acquisition_date")
    
    if not date1_str or not date2_str:
        # Fallback for S1-AAD which lacks TIFF dates: if same modality, assume bi-temporal
        if mod1 != "unknown" and mod1 == mod2:
            return "bi_temporal"
        return "unknown"
        
    try:
        # Extract just the date part (YYYY-MM-DD) for simple comparison
        # Format might be 'YYYY:MM:DD HH:MM:SS' (EXIF) or 'YYYY-MM-DD HH:MM:SS'
        d1_clean = str(date1_str).replace(":", "-").split(" ")[0][:10]
        d2_clean = str(date2_str).replace(":", "-").split(" ")[0][:10]
        
        if d1_clean == d2_clean:
            return "same_date"
        else:
            return "bi_temporal"
    except Exception:
        if mod1 != "unknown" and mod1 == mod2:
            return "bi_temporal"
        return "unknown"
