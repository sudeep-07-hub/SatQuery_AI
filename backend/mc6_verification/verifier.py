"""
verifier.py — MC6.1 Evidence Reconciliation & Verification Loop.

Trigger conditions relevant to temporal specialist (MC4B):
- Conflicting masks between specialists
- Low confidence (below threshold)
- Insufficient spatial coverage
- Image-quality problems from MC1

Output states: VERIFIED / RE_PLAN_REQUIRED / INSUFFICIENT_EVIDENCE
"""

from typing import Dict, List, Optional, Callable
from mc4b_temporal import config


class Verifier:
    """
    Checks evidence objects and decides whether to verify, re-plan, or halt.
    """

    def __init__(
        self,
        confidence_threshold: float = None,
        max_replan_attempts: int = None,
    ):
        self.confidence_threshold = (
            confidence_threshold or config.LOW_CONFIDENCE_THRESHOLD
        )
        self.max_replan_attempts = (
            max_replan_attempts or config.MAX_REPLAN_ATTEMPTS
        )
        self._replan_count = 0

    def verify(
        self,
        evidence_objects: List[Dict],
        mc1_profile: Optional[Dict] = None,
    ) -> Dict:
        """
        Run verification triggers on a set of evidence objects.

        Returns:
            {
                "status": "VERIFIED" | "RE_PLAN_REQUIRED" | "INSUFFICIENT_EVIDENCE",
                "triggers_fired": [...],
                "replan_count": int,
            }
        """
        triggers = []

        if not evidence_objects:
            return {
                "status": "INSUFFICIENT_EVIDENCE",
                "triggers_fired": ["no_evidence_produced"],
                "replan_count": self._replan_count,
            }

        # Check confidence
        for ev in evidence_objects:
            conf = ev.get("confidence", 0.0)
            if conf < self.confidence_threshold:
                triggers.append({
                    "trigger": "low_confidence",
                    "evidence_id": ev.get("evidence_id"),
                    "confidence": conf,
                    "threshold": self.confidence_threshold,
                })

        # Check image quality from MC1
        if mc1_profile:
            quality = mc1_profile.get("quality", {})
            for key, score in quality.items():
                if isinstance(score, (int, float)) and score < 0.5:
                    triggers.append({
                        "trigger": "image_quality_problem",
                        "image": key,
                        "quality_score": score,
                    })

            # Check spatial coverage
            overlap = mc1_profile.get("spatial_overlap")
            if overlap is not None and overlap < config.MIN_SPATIAL_OVERLAP:
                triggers.append({
                    "trigger": "insufficient_spatial_coverage",
                    "overlap": overlap,
                    "threshold": config.MIN_SPATIAL_OVERLAP,
                })

        # Check for conflicting evidence (masks disagree)
        if len(evidence_objects) > 1:
            claims = [ev.get("claim", "") for ev in evidence_objects]
            # Simple conflict detection: if claims contain contradictory keywords
            has_increase = any("increased" in c.lower() for c in claims)
            has_decrease = any("decreased" in c.lower() for c in claims)
            has_no_change = any("no change" in c.lower() or "no significant" in c.lower() for c in claims)

            if (has_increase and has_no_change) or (has_decrease and has_no_change):
                triggers.append({
                    "trigger": "conflicting_masks",
                    "claims": claims,
                })

        # Decision
        if not triggers:
            return {
                "status": "VERIFIED",
                "triggers_fired": [],
                "replan_count": self._replan_count,
            }

        # Should we re-plan or give up?
        if self._replan_count >= self.max_replan_attempts:
            return {
                "status": "INSUFFICIENT_EVIDENCE",
                "triggers_fired": triggers,
                "replan_count": self._replan_count,
            }

        self._replan_count += 1
        return {
            "status": "RE_PLAN_REQUIRED",
            "triggers_fired": triggers,
            "replan_count": self._replan_count,
        }

    def reset(self):
        """Reset re-plan counter for a new query."""
        self._replan_count = 0
