import pytest
from mc6_verification.temporal_verifier import TemporalClaimVerifier

def test_model_unavailable_blocked():
    verifier = TemporalClaimVerifier()
    trace = verifier.verify(
        temporal_answer={"status": "ANSWERED", "answer": "No change.", "evidence_ids": []},
        evidence_objects=[],
        pipeline_status="MODEL_UNAVAILABLE"
    )
    assert trace.verification_status == "TEMPORAL_EVIDENCE_UNAVAILABLE"
    assert "model_status_check" in trace.checks

def test_valid_verified_claim():
    verifier = TemporalClaimVerifier()
    trace = verifier.verify(
        temporal_answer={"status": "ANSWERED", "answer": "Change detected.", "evidence_ids": ["e1"]},
        evidence_objects=[{"evidence_id": "e1", "claim": "Spatial change."}],
        pipeline_status="SUCCESS"
    )
    assert trace.verification_status == "VERIFIED"

def test_invalid_evidence_id():
    verifier = TemporalClaimVerifier()
    trace = verifier.verify(
        temporal_answer={"status": "ANSWERED", "answer": "Change detected.", "evidence_ids": ["e99"]},
        evidence_objects=[{"evidence_id": "e1", "claim": "Spatial change."}],
        pipeline_status="SUCCESS"
    )
    assert trace.verification_status == "INVALID_EVIDENCE_REFERENCE"

def test_unsupported_semantic_hallucination():
    verifier = TemporalClaimVerifier()
    trace = verifier.verify(
        temporal_answer={"status": "ANSWERED", "answer": "New building constructed.", "evidence_ids": ["e1"]},
        evidence_objects=[{"evidence_id": "e1", "claim": "Generic spatial change."}],
        pipeline_status="SUCCESS"
    )
    assert trace.verification_status == "UNSUPPORTED"
    assert "semantic_support_check" in trace.checks

def test_reversed_temporal_ordering():
    verifier = TemporalClaimVerifier()
    trace = verifier.verify(
        temporal_answer={"status": "ANSWERED", "answer": "Change from 2024 to 2020.", "evidence_ids": ["e1"]},
        evidence_objects=[{"evidence_id": "e1", "timestamp": {"t1": "2020", "t2": "2024"}}],
        pipeline_status="SUCCESS"
    )
    assert trace.verification_status == "DISPUTED"
    assert "temporal_consistency_check" in trace.checks

def test_zero_change_success():
    verifier = TemporalClaimVerifier()
    trace = verifier.verify(
        temporal_answer={"status": "ANSWERED", "answer": "I did not detect any change.", "evidence_ids": []},
        evidence_objects=[],
        pipeline_status="SUCCESS"
    )
    assert trace.verification_status == "VERIFIED"

def test_unsupported_area_claim():
    verifier = TemporalClaimVerifier()
    trace = verifier.verify(
        temporal_answer={"status": "ANSWERED", "answer": "20 hectares changed.", "evidence_ids": ["e1"]},
        evidence_objects=[{"evidence_id": "e1", "claim": "Spatial change."}],
        pipeline_status="SUCCESS"
    )
    assert trace.verification_status == "UNSUPPORTED"
    assert "quantitative_support_check" in trace.checks
