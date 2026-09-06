"""
conflict_detector.py — MC6 Conflicting evidence detection.

Detects disagreement between evidence objects from different specialists
(e.g., MC4B says "increase", MC4C says "no change") and produces
conflict records for the verification loop.
"""

from typing import Dict, List


def detect_conflicts(evidence_objects: List[Dict]) -> List[Dict]:
    """
    Detect conflicts between evidence objects from different source models.

    Returns:
        List of conflict dicts:
        [{"evidence_a": id, "evidence_b": id, "conflict_type": str, "description": str}]
    """
    conflicts = []

    for i, ev_a in enumerate(evidence_objects):
        for ev_b in evidence_objects[i + 1 :]:
            # Only flag conflicts between different source models
            if ev_a.get("source_model") == ev_b.get("source_model"):
                continue

            claim_a = ev_a.get("claim", "").lower()
            claim_b = ev_b.get("claim", "").lower()

            # Check for directional contradictions
            a_has_increase = "increased" in claim_a
            b_has_increase = "increased" in claim_b
            a_no_change = "no change" in claim_a or "no significant change" in claim_a
            b_no_change = "no change" in claim_b or "no significant change" in claim_b

            if (a_has_increase and b_no_change) or (b_has_increase and a_no_change):
                conflicts.append({
                    "evidence_a": ev_a.get("evidence_id"),
                    "evidence_b": ev_b.get("evidence_id"),
                    "conflict_type": "directional_contradiction",
                    "description": (
                        f"{ev_a.get('source_model')} claims '{ev_a.get('claim')}' "
                        f"but {ev_b.get('source_model')} claims '{ev_b.get('claim')}'"
                    ),
                })

            if ("decreased" in claim_a and "increased" in claim_b) or \
               ("increased" in claim_a and "decreased" in claim_b):
                conflicts.append({
                    "evidence_a": ev_a.get("evidence_id"),
                    "evidence_b": ev_b.get("evidence_id"),
                    "conflict_type": "directional_contradiction",
                    "description": (
                        f"{ev_a.get('source_model')} and {ev_b.get('source_model')} "
                        f"disagree on direction of change"
                    ),
                })

    return conflicts
