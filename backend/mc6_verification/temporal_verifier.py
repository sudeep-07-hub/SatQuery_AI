"""
temporal_verifier.py — MC6.1 Temporal Claim Verifier.

Independently verifies Qwen3 temporal semantic claims against the structured
MC5 Evidence Objects to detect hallucination, missing evidence, temporal
reversals, and model availability blockage.
"""

from typing import Dict, List, Optional
from pydantic import BaseModel

class VerificationTrace(BaseModel):
    verification_status: str
    checks: List[str]
    reasons: List[str]


class TemporalClaimVerifier:
    def verify(
        self,
        temporal_answer: Dict,
        evidence_objects: List[Dict],
        pipeline_status: str
    ) -> VerificationTrace:
        """
        Verify the TemporalAnswer against the Evidence Objects.
        
        Args:
            temporal_answer: Serialized TemporalAnswer from Qwen3.
            evidence_objects: The MC5 evidence used to generate the answer.
            pipeline_status: The status of the temporal model pipeline.
        """
        checks = []
        reasons = []
        status = "VERIFIED"
        
        # 1. Model Availability Check
        if pipeline_status == "MODEL_UNAVAILABLE":
            return VerificationTrace(
                verification_status="TEMPORAL_EVIDENCE_UNAVAILABLE",
                checks=["model_status_check"],
                reasons=["The temporal model was unavailable. Cannot verify claims."]
            )
            
        # Extract fields
        answer_status = temporal_answer.get("status")
        claimed_evidence_ids = temporal_answer.get("evidence_ids", [])
        answer_text = temporal_answer.get("answer", "").lower()
        
        # 2. Qwen Error / Abstention Check
        if answer_status in ["INSUFFICIENT_EVIDENCE", "GENERATION_FAILED", "INVALID_EVIDENCE"]:
            return VerificationTrace(
                verification_status="ABSTAIN",
                checks=["qwen_status_check"],
                reasons=[f"Generator abstained or failed with status: {answer_status}."]
            )
            
        # Map existing evidence
        available_evidence = {e.get("evidence_id"): e for e in evidence_objects}
        
        # 3. Evidence Reference Validation Check
        missing_ids = [eid for eid in claimed_evidence_ids if eid not in available_evidence]
        if missing_ids:
            return VerificationTrace(
                verification_status="INVALID_EVIDENCE_REFERENCE",
                checks=["evidence_reference_check"],
                reasons=[f"Claim references non-existent evidence IDs: {missing_ids}."]
            )
            
        # 4. Zero-change success check
        if pipeline_status == "SUCCESS" and not evidence_objects:
            # If Qwen claims no change, that is VERIFIED.
            if "no change" in answer_text or "not detect" in answer_text:
                return VerificationTrace(
                    verification_status="VERIFIED",
                    checks=["zero_change_check"],
                    reasons=["Correctly identified successful zero-change result."]
                )
            else:
                return VerificationTrace(
                    verification_status="DISPUTED",
                    checks=["zero_change_check"],
                    reasons=["Claimed change but evidence contains 0 change regions."]
                )
                
        # 5. Semantic Support Check
        # Ensure hallucinated semantics like "building", "road", "forest" are not in the answer
        # unless explicitly supported by the evidence claims.
        hallucination_risks = ["building", "road", "forest", "urban", "water"]
        for risk in hallucination_risks:
            if risk in answer_text:
                supported = False
                for eid in claimed_evidence_ids:
                    ev_claim = available_evidence[eid].get("claim", "").lower()
                    if risk in ev_claim:
                        supported = True
                        break
                if not supported:
                    status = "UNSUPPORTED"
                    reasons.append(f"Answer contains unsupported semantic claim: '{risk}'.")
                    checks.append("semantic_support_check")
                    
        # 6. Temporal Ordering Consistency
        # Check if Qwen3 reversed the chronological relationship.
        # e.g., if answer states "from 2024 to 2020" but t1 is 2020.
        for eid in claimed_evidence_ids:
            ev = available_evidence[eid]
            timestamps = ev.get("timestamp", {})
            t1 = timestamps.get("t1")
            t2 = timestamps.get("t2")
            if t1 and t2:
                # Basic string heuristic to detect explicit reversal
                reversed_pattern = f"from {t2} to {t1}".lower()
                if reversed_pattern in answer_text:
                    status = "DISPUTED"
                    reasons.append("Answer reverses the explicit temporal ordering (t1, t2).")
                    checks.append("temporal_consistency_check")
                    
        # 7. Area/Quantitative Check
        # Check if the answer hallucinates quantitative area measurements (e.g. "hectares", "m²", "%")
        quant_keywords = ["hectare", "m²", "m2", "%", "percent"]
        if any(q in answer_text for q in quant_keywords):
            supported = False
            for eid in claimed_evidence_ids:
                ev_claim = available_evidence[eid].get("claim", "").lower()
                if any(q in ev_claim for q in quant_keywords):
                    supported = True
                    break
            if not supported:
                if status == "VERIFIED":
                    status = "UNSUPPORTED"
                reasons.append("Answer hallucinates quantitative area/percentage without evidence support.")
                checks.append("quantitative_support_check")

        if not checks:
            checks.append("all_checks_passed")
            reasons.append("All deterministic checks passed.")
            
        return VerificationTrace(
            verification_status=status,
            checks=checks,
            reasons=reasons
        )
