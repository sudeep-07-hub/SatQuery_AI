from shapely.geometry import box
import math

def get_footprint(meta: dict):
    """
    Creates a shapely geometry polygon from rasterio bounds.
    """
    bounds = meta.get("bounds")
    if bounds:
        return box(bounds.left, bounds.bottom, bounds.right, bounds.top)
    return None

def verify_crs(meta1: dict, meta2: dict) -> tuple[bool, str]:
    """
    Step 5: Verify CRS of two images. 
    Returns (match, warning_msg).
    """
    crs1 = meta1.get("crs")
    crs2 = meta2.get("crs")
    
    if not crs1 or not crs2:
        return False, "One or both images missing CRS."
    
    if crs1 != crs2:
        # In a full implementation, we'd use pyproj or rasterio.warp to reproject footprints here.
        # For MC1, we'll note the mismatch and proceed assuming they are close enough or overlapping is uncomputable.
        return False, f"CRS mismatch: {crs1} vs {crs2}."
        
    return True, ""

def calculate_overlap(foot1, foot2) -> float:
    """
    Step 8: Compute IoU overlap.
    """
    if not foot1 or not foot2:
        return None
        
    try:
        intersection = foot1.intersection(foot2).area
        union = foot1.union(foot2).area
        if union == 0:
            return 0.0
        return intersection / union
    except Exception:
        return None

def calculate_coregistration(foot1, foot2) -> float:
    """
    Step 9: Co-registration proxy using footprint centroid distance vs size.
    Returns 0-1 score where 1 is perfect alignment.
    """
    if not foot1 or not foot2:
        return None
        
    try:
        c1 = foot1.centroid
        c2 = foot2.centroid
        
        dist = c1.distance(c2)
        # Max distance proxy based on diagonal of union
        union = foot1.union(foot2)
        minx, miny, maxx, maxy = union.bounds
        diag = math.sqrt((maxx-minx)**2 + (maxy-miny)**2)
        
        if diag == 0:
            return 1.0
            
        score = 1.0 - min(1.0, dist / (diag / 2)) # simplistic proxy
        return round(max(0.0, score), 2)
    except Exception:
        return None
