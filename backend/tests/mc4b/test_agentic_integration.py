"""
test_agentic_integration.py — Agentic pipeline integration tests.

F. Planner selects CHANGE_MAMBA correctly (and excludes it for wrong inputs).
   Re-plan loop terminates.
   End-to-end trace through MC2→MC3→MC4B→MC5→MC6.
"""

import pytest
import torch
from mc3_planner.tool_registry import ToolRegistry
from mc3_planner.workflow_planner import build_workflow_plan
from mc3_planner.dispatcher import execute_plan
from mc4b_temporal.tool_adapter import CHANGE_MAMBA_TOOL
from mc4b_temporal.evidence_normalizer import normalize_to_evidence
from mc5_evidence.evidence_graph import EvidenceGraph
from mc5_evidence.schemas import validate_evidence_object
from mc6_verification.verifier import Verifier
from mc6_verification.conflict_detector import detect_conflicts


@pytest.fixture
def registry():
    reg = ToolRegistry()
    reg.register(CHANGE_MAMBA_TOOL)
    return reg


class TestPlannerSelection:
    """Planner selects CHANGE_MAMBA only for valid temporal requests."""

    def test_selects_for_bitemporal_change(
        self, registry, valid_mc1_profile, sample_task_spec
    ):
        plan = build_workflow_plan(sample_task_spec, valid_mc1_profile, registry)
        assert plan["status"] == "PLAN_READY"
        assert "CHANGE_MAMBA" in plan["selected_tools"]

    def test_excludes_for_single_image(
        self, registry, single_image_profile, sample_task_spec
    ):
        plan = build_workflow_plan(sample_task_spec, single_image_profile, registry)
        # Should not find a viable tool (image_count mismatch)
        assert plan["status"] == "CONSTRAINTS_NOT_MET"

    def test_excludes_for_low_overlap(
        self, registry, low_overlap_profile, sample_task_spec
    ):
        plan = build_workflow_plan(sample_task_spec, low_overlap_profile, registry)
        assert plan["status"] == "CONSTRAINTS_NOT_MET"

    def test_excludes_for_non_temporal_query(self, registry, valid_mc1_profile):
        non_temporal_spec = {
            "primary_task": "object_detection",
            "temporal_requirement": "",
            "required_modalities": ["optical"],
        }
        plan = build_workflow_plan(non_temporal_spec, valid_mc1_profile, registry)
        assert plan["status"] in ("NO_TOOL_FOUND", "CONSTRAINTS_NOT_MET")


class TestEndToEndTrace:
    """Full pipeline: task spec → plan → execute → evidence → verify."""

    def test_clean_passthrough(
        self, registry, valid_mc1_profile, sample_task_spec, sample_tensors
    ):
        # 1. Build plan
        plan = build_workflow_plan(sample_task_spec, valid_mc1_profile, registry)
        assert plan["status"] == "PLAN_READY"

        # 2. Execute plan
        exec_result = execute_plan(
            plan, valid_mc1_profile, sample_tensors, "Has built-up area increased?",
            seed=42,
        )
        assert exec_result["status"] == "SUCCESS"

        # 3. Check evidence objects
        evidence = exec_result["evidence_objects"]
        assert len(evidence) >= 1
        for ev in evidence:
            assert validate_evidence_object(ev)

        # 4. Insert into evidence graph
        graph = EvidenceGraph()
        graph.add_node("query_1", "query", {"text": "Has built-up area increased?"})
        for ev in evidence:
            graph.insert_evidence(ev, "query_1")
        assert graph.node_count >= 3  # query + evidence + claim + model

        # 5. Verify
        verifier = Verifier()
        verification = verifier.verify(evidence, valid_mc1_profile)
        # With random weights, confidence may be low — that's OK for this test.
        # We just check the verifier returns a valid status.
        assert verification["status"] in (
            "VERIFIED",
            "RE_PLAN_REQUIRED",
            "INSUFFICIENT_EVIDENCE",
        )

    def test_forced_conflict_replan(self, valid_mc1_profile, sample_tensors):
        """
        Force a conflict scenario: two specialists disagree.
        Verify the loop re-plans once then terminates.
        """
        # Simulate: MC4B says "increased", a mock SAR specialist says "no change"
        mc4b_evidence = {
            "evidence_id": "ev_mc4b_001",
            "claim": "Built-up area increased. Changed area: 500 m², 2.5% of scene.",
            "evidence_type": "change_detection_mask",
            "spatial_region": None,
            "modality": "optical",
            "timestamp": "2024-01-01T00:00:00Z",
            "source_model": "CHANGE_MAMBA",
            "source_input": {"image_1": "t1.tif", "image_2": "t2.tif"},
            "confidence": 0.65,
            "processing_parameters": {},
        }
        sar_evidence = {
            "evidence_id": "ev_sar_001",
            "claim": "No significant change detected in SAR analysis.",
            "evidence_type": "change_detection_mask",
            "spatial_region": None,
            "modality": "sar",
            "timestamp": "2024-01-01T00:00:00Z",
            "source_model": "SAR_COHERENCE",
            "source_input": {"image_1": "t1_sar.tif", "image_2": "t2_sar.tif"},
            "confidence": 0.72,
            "processing_parameters": {},
        }

        # Detect conflict
        conflicts = detect_conflicts([mc4b_evidence, sar_evidence])
        assert len(conflicts) >= 1
        assert conflicts[0]["conflict_type"] == "directional_contradiction"

        # Verify → should trigger RE_PLAN
        verifier = Verifier(max_replan_attempts=1)
        v1 = verifier.verify([mc4b_evidence, sar_evidence], valid_mc1_profile)
        assert v1["status"] == "RE_PLAN_REQUIRED"
        assert v1["replan_count"] == 1

        # Second attempt → should return INSUFFICIENT_EVIDENCE (max reached)
        v2 = verifier.verify([mc4b_evidence, sar_evidence], valid_mc1_profile)
        assert v2["status"] == "INSUFFICIENT_EVIDENCE"
        assert v2["replan_count"] == 1  # didn't increment again

    def test_audit_trail_evidence_shape(
        self, registry, valid_mc1_profile, sample_task_spec, sample_tensors
    ):
        """
        Confirm the audit-trail receives well-formed trace objects
        for both success and rejection cases.
        """
        # Success case
        plan = build_workflow_plan(sample_task_spec, valid_mc1_profile, registry)
        exec_result = execute_plan(
            plan, valid_mc1_profile, sample_tensors, "test",
            seed=42,
        )
        assert exec_result["status"] == "SUCCESS"
        for ev in exec_result["evidence_objects"]:
            assert "evidence_id" in ev
            assert "claim" in ev
            assert "source_model" in ev

        # Rejection case (single image)
        single_prof = {
            "image_count": 1,
            "image_1": {
                "modality": "optical", "crs": "EPSG:32643", "gsd_m": 1.0,
            },
            "spatial_overlap": None,
            "coregistration_score": None,
        }
        plan_bad = build_workflow_plan(sample_task_spec, single_prof, registry)
        assert plan_bad["status"] == "CONSTRAINTS_NOT_MET"
