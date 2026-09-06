"""
controller.py — Stage 2: Agentic Orchestration & Decomposition Controller

Phase 2: Task Classifier
"""

def classify_task(query: str, profile: dict) -> str:
    """
    Classifies a natural-language query into a known task type.
    Must run in isolation (no imports from engine_registry).
    
    Precedence order for classification:
    1. change_detection
    2. cross_modal_fusion
    3. grounding
    4. captioning
    5. single_image_vqa (default fallback)
    """
    if not query or not query.strip():
        return "single_image_vqa"

    q_lower = query.lower()

    # 1. change_detection
    change_keywords = ["changed", "increase", "decrease", "before and after", "between these"]
    if any(k in q_lower for k in change_keywords):
        return "change_detection"

    # 2. cross_modal_fusion
    # Check keywords first
    if "sar" in q_lower and ("optical" in q_lower or "together" in q_lower):
        return "cross_modal_fusion"
    
    # Input-shape fallback for fusion: if profile literally has optical+SAR
    image_count = profile.get("image_count", 0)
    modalities = set()
    for i in range(1, image_count + 1):
        img = profile.get(f"image_{i}", {})
        mod = img.get("modality", "unknown")
        if mod and mod != "unknown":
            modalities.add(mod)
            
    if modalities == {"optical", "sar"}:
        return "cross_modal_fusion"

    # 3. grounding
    grounding_keywords = ["highlight", "locate", "where is", "point to"]
    if any(k in q_lower for k in grounding_keywords):
        return "grounding"

    # 4. captioning
    captioning_keywords = ["describe", "caption", "scene"]
    if any(k in q_lower for k in captioning_keywords):
        return "captioning"

    # 5. single_image_vqa (default fallback)
    return "single_image_vqa"

from .engine_registry import ENGINE_REGISTRY, adapt_profile

def route_query(query: str, profile: dict) -> dict:
    """
    Dispatcher. Wires Phase 1 registry to Phase 2 classifier.
    """
    try:
        task_type = classify_task(query, profile)
        entry = ENGINE_REGISTRY.get(task_type)
        
        # This shouldn't happen unless registry gets out of sync with classifier
        if not entry:
            return {
                "status": "error",
                "error": f"Unknown task type: {task_type}"
            }
            
        adapted = adapt_profile(profile)
        
        if not entry["precondition"](adapted):
            return {
                "status": "rejected",
                "reason": entry["precondition_msg"],
                "answer": None
            }
            
        handler = entry["handler"]
        result = handler(adapted)
        
        return {
            "status": "ok",
            "task_type": task_type,
            "engine": handler.__name__,
            "confidence": result.get("confidence"),
            "answer": result.get("answer"),
            "evidence_region": result.get("bbox")
        }
        
    except Exception as e:
        # Never let raw tracebacks propagate
        return {
            "status": "error",
            "error": "Internal engine error occurred."
        }
