"""
schemas.py — MC5.1 Evidence Object schema & validation.
"""

EVIDENCE_OBJECT_REQUIRED_KEYS = {
    "evidence_id",
    "claim",
    "evidence_type",
    "spatial_region",
    "modality",
    "timestamp",
    "source_model",
    "source_input",
    "confidence",
    "processing_parameters",
}


def validate_evidence_object(obj: dict) -> bool:
    """Return True if obj has all required MC5.1 keys."""
    return EVIDENCE_OBJECT_REQUIRED_KEYS.issubset(set(obj.keys()))
