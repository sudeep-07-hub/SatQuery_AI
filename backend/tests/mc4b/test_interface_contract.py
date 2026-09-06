"""
test_interface_contract.py — Phase 2 & 3 tests.

A. Interface-contract: output object keys, types, confidence range.
B. Precondition / fail-fast: CRS mismatch, low overlap, low coreg, single image, >2 images.
"""

import pytest
import torch
from mc4b_temporal.tool_adapter import (
    ChangeMambaAdapter,
    CHANGE_MAMBA_TOOL,
    validate_preconditions,
)
from mc4b_temporal import config
from mc3_planner.tool_registry import REQUIRED_TOOL_FIELDS
from mc5_evidence.schemas import validate_evidence_object, EVIDENCE_OBJECT_REQUIRED_KEYS
from mc4b_temporal.evidence_normalizer import normalize_to_evidence


# ══════════════════════════════════════════════════════════════════
# A. Interface-Contract Tests
# ══════════════════════════════════════════════════════════════════

class TestOutputContract:
    """MC4B output must contain every required key with correct types."""

    @pytest.fixture(autouse=True)
    def _setup(self, valid_mc1_profile, sample_tensors):
        adapter = ChangeMambaAdapter()
        self.result = adapter.execute(
            valid_mc1_profile,
            sample_tensors["t1"],
            sample_tensors["t2"],
            query="Has built-up area increased?",
            seed=42,
        )

    def test_status_is_success(self):
        assert self.result["status"] == "SUCCESS"

    def test_has_change_map(self):
        assert "change_map" in self.result
        assert self.result["change_map"] is not None

    def test_has_binary_change_mask(self):
        assert "binary_change_mask" in self.result

    def test_has_change_score(self):
        assert "change_score" in self.result
        assert isinstance(self.result["change_score"], float)

    def test_has_changed_region_coordinates(self):
        assert "changed_region_coordinates" in self.result
        assert isinstance(self.result["changed_region_coordinates"], list)

    def test_changed_regions_have_geometry_and_crs(self):
        for region in self.result["changed_region_coordinates"]:
            assert "geometry" in region
            assert "crs" in region

    def test_has_change_statistics(self):
        stats = self.result["change_statistics"]
        assert "changed_area_m2" in stats
        assert "changed_pixel_pct" in stats

    def test_has_model_confidence(self):
        assert "model_confidence" in self.result

    def test_confidence_in_0_1(self):
        conf = self.result["model_confidence"]
        assert isinstance(conf, float)
        assert 0.0 <= conf <= 1.0

    def test_has_source_model(self):
        assert self.result["source_model"] == "CHANGE_MAMBA"

    def test_has_semantics(self):
        sem = self.result["semantics"]
        assert "has_change" in sem
        assert "primary_change" in sem
        assert "description" in sem

    def test_has_caption(self):
        assert "caption" in self.result
        assert isinstance(self.result["caption"], str)


class TestToolRegistrySchema:
    """CHANGE_MAMBA_TOOL validates against MC3.1 schema."""

    def test_all_required_fields_present(self):
        missing = REQUIRED_TOOL_FIELDS - set(CHANGE_MAMBA_TOOL.keys())
        assert not missing, f"Missing: {missing}"

    def test_name_is_string(self):
        assert isinstance(CHANGE_MAMBA_TOOL["name"], str)

    def test_capabilities_is_list(self):
        assert isinstance(CHANGE_MAMBA_TOOL["task_capabilities"], list)
        assert len(CHANGE_MAMBA_TOOL["task_capabilities"]) >= 1

    def test_baseline_capability_included(self):
        assert "binary_change_detection" in CHANGE_MAMBA_TOOL["task_capabilities"]


class TestEvidenceObjectSchema:
    """Evidence objects from MC4B must conform to MC5.1 schema."""

    def test_evidence_objects_valid(self, valid_mc1_profile, sample_tensors):
        adapter = ChangeMambaAdapter()
        result = adapter.execute(
            valid_mc1_profile,
            sample_tensors["t1"],
            sample_tensors["t2"],
            seed=42,
        )
        evidence = normalize_to_evidence(result, valid_mc1_profile, "test query")
        assert len(evidence) >= 1
        for ev in evidence:
            assert validate_evidence_object(ev), (
                f"Evidence missing keys: "
                f"{EVIDENCE_OBJECT_REQUIRED_KEYS - set(ev.keys())}"
            )

    def test_confidence_field_in_evidence(self, valid_mc1_profile, sample_tensors):
        adapter = ChangeMambaAdapter()
        result = adapter.execute(
            valid_mc1_profile,
            sample_tensors["t1"],
            sample_tensors["t2"],
            seed=42,
        )
        evidence = normalize_to_evidence(result, valid_mc1_profile, "test")
        for ev in evidence:
            assert 0.0 <= ev["confidence"] <= 1.0


# ══════════════════════════════════════════════════════════════════
# B. Precondition / Fail-Fast Tests
# ══════════════════════════════════════════════════════════════════

class TestPreconditions:
    """Precondition validators must reject bad inputs before inference."""

    def test_mismatched_crs_rejected(self, mismatched_crs_profile):
        result = validate_preconditions(mismatched_crs_profile)
        assert not result["passed"]
        assert result["status"] == "PRECONDITION_FAILED"
        checks = [f["check"] for f in result["failed"]]
        assert "same_crs" in checks

    def test_low_overlap_rejected(self, low_overlap_profile):
        result = validate_preconditions(low_overlap_profile)
        assert not result["passed"]
        checks = [f["check"] for f in result["failed"]]
        assert "sufficient_overlap" in checks

    def test_low_coreg_rejected(self, low_coreg_profile):
        result = validate_preconditions(low_coreg_profile)
        assert not result["passed"]
        checks = [f["check"] for f in result["failed"]]
        assert "spatially_aligned" in checks

    def test_single_image_rejected(self, single_image_profile):
        result = validate_preconditions(single_image_profile)
        assert not result["passed"]
        checks = [f["check"] for f in result["failed"]]
        assert "image_count" in checks

    def test_three_images_rejected(self):
        profile = {
            "image_count": 3,
            "image_1": {"crs": "EPSG:32643"},
            "image_2": {"crs": "EPSG:32643"},
            "image_3": {"crs": "EPSG:32643"},
        }
        result = validate_preconditions(profile)
        assert not result["passed"]

    def test_valid_profile_passes(self, valid_mc1_profile):
        result = validate_preconditions(valid_mc1_profile)
        assert result["passed"]

    def test_adapter_refuses_on_bad_input(self, mismatched_crs_profile, sample_tensors):
        """Tool adapter returns structured refusal, NOT a crash."""
        adapter = ChangeMambaAdapter()
        result = adapter.execute(
            mismatched_crs_profile,
            sample_tensors["t1"],
            sample_tensors["t2"],
        )
        assert result["status"] == "PRECONDITION_FAILED"
        assert "failed" in result
