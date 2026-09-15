"""
semantic_interpreter.py — Qwen3 Temporal Evidence Reasoning Layer

Provides the boundary layer for Qwen3 to interpret MC5 Evidence Objects.
Enforces structural blockades against hallucination and raw pixel assumption.
"""

from typing import List, Dict, Optional, Any
from pydantic import BaseModel
from qwen.inference import Qwen3Inference
from .result import TemporalResult


class TemporalEvidenceContext(BaseModel):
    query: str
    task_type: str  # e.g., 'change_description', 'change_vqa'
    before_observation: str
    after_observation: str
    evidence_objects: List[Dict]
    pipeline_status: str


class TemporalAnswer(BaseModel):
    status: str
    answer: str
    evidence_ids: List[str]
    supported_claims: List[str]
    uncertainty: str


class TemporalSemanticInterpreter:
    def __init__(self, inference_engine: Optional[Qwen3Inference] = None):
        self.qwen = inference_engine or Qwen3Inference()

    def generate_answer(self, context: TemporalEvidenceContext) -> TemporalAnswer:
        """
        Executes the explicit semantic interpretation against Qwen3.
        """
        # Fast paths for non-success pipeline states
        if context.pipeline_status == "MODEL_UNAVAILABLE":
            return TemporalAnswer(
                status="TEMPORAL_EVIDENCE_UNAVAILABLE",
                answer="The temporal evidence model is currently unavailable; cannot answer.",
                evidence_ids=[],
                supported_claims=[],
                uncertainty="Complete."
            )
            
        if context.pipeline_status == "SUCCESS" and not context.evidence_objects:
            return TemporalAnswer(
                status="ANSWERED",
                answer="No change detected between the specified observations.",
                evidence_ids=[],
                supported_claims=["no_change_detected"],
                uncertainty="None."
            )
            
        if context.pipeline_status != "SUCCESS":
            return TemporalAnswer(
                status="INVALID_EVIDENCE",
                answer="The provided temporal evidence is invalid or incomplete.",
                evidence_ids=[],
                supported_claims=[],
                uncertainty="High."
            )
            
        # Ensure we don't ask unanswerable questions
        if "building" in context.query.lower() or "forest" in context.query.lower():
            # If evidence does not contain semantic tags, fail safely
            has_semantics = any("building" in e.get("claim", "").lower() for e in context.evidence_objects)
            if not has_semantics:
                return TemporalAnswer(
                    status="INSUFFICIENT_EVIDENCE",
                    answer="The available evidence indicates spatial change, but it does not establish specific semantic land-cover types like buildings or forests.",
                    evidence_ids=[],
                    supported_claims=[],
                    uncertainty="High (Ungrounded Semantic Query)."
                )

        # Build formal prompt for Qwen
        evidence_text = ""
        evidence_ids = []
        for e in context.evidence_objects:
            eid = e.get("evidence_id")
            claim = e.get("claim")
            timestamp = e.get("timestamp", {})
            evidence_ids.append(eid)
            evidence_text += f"- Evidence ID: {eid}\n  Claim: {claim}\n  Time Context: {timestamp.get('t1')} -> {timestamp.get('t2')}\n"
            
        prompt = (
            f"You are a strict temporal reasoning assistant. Answer only from the supplied temporal evidence.\n"
            f"Do not claim visual observations that are not represented in the evidence.\n"
            f"If the evidence is insufficient to answer the question, explicitly state that the evidence is insufficient.\n\n"
            f"Task: {context.task_type}\n"
            f"Query: {context.query}\n"
            f"Temporal Context: {context.before_observation} -> {context.after_observation}\n\n"
            f"Evidence Objects:\n{evidence_text}\n"
            f"Answer:"
        )

        # Execute
        self.qwen.load()
        response = self.qwen.generate(prompt, temperature=0.1, max_new_tokens=150)
        
        if response["status"] != "ok":
            return TemporalAnswer(
                status="GENERATION_FAILED",
                answer="Failed to generate an interpretation.",
                evidence_ids=[],
                supported_claims=[],
                uncertainty="High."
            )
            
        return TemporalAnswer(
            status="ANSWERED",
            answer=response["response"],
            evidence_ids=evidence_ids,
            supported_claims=["model_predicted_change"],
            uncertainty="Dependent on upstream model score."
        )
