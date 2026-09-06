"""
test_registry.py — Phase 1 tests for ENGINE_REGISTRY and adapt_profile.

Tests:
  1. Structural: every entry has callable handler, callable precondition, non-empty msg.
  2. Precondition pass/fail: for each of 5 task types, one passing and one failing profile.
  3. Handler stubs: each returns a dict with answer, confidence, bbox keys.
"""

import pytest
from controller.engine_registry import ENGINE_REGISTRY, adapt_profile


# ── Helper profiles ───────────────────────────────────────────────

def _make_profile(**overrides):
    """Build a minimal raw Stage 1 profile, applying overrides."""
    base = {
        "image_count": 1,
        "query": "test query",
        "spatial_overlap": None,
        "coregistration_score": None,
        "relationship": "single_image",
        "quality": {"optical": 0.9},
        "warnings": [],
        "task_executable": True,
        "image_1": {
            "filename": "a.tif",
            "modality": "optical",
            "sensor": "Generic Optical",
            "gsd_m": 10.0,
            "crs": "EPSG:32643",
            "acquisition_date": "2024-01-01 10:00:00",
        },
    }
    base.update(overrides)
    return base


def _single_optical():
    return adapt_profile(_make_profile())


def _two_optical_high_overlap():
    return adapt_profile(_make_profile(
        image_count=2,
        spatial_overlap=0.8,
        coregistration_score=0.7,
        relationship="multi_temporal",
        image_2={
            "filename": "b.tif",
            "modality": "optical",
            "sensor": "Generic Optical",
            "gsd_m": 10.0,
            "crs": "EPSG:32643",
            "acquisition_date": "2024-02-01 10:00:00",
        },
    ))


def _optical_sar_high_overlap():
    return adapt_profile(_make_profile(
        image_count=2,
        spatial_overlap=0.75,
        coregistration_score=0.6,
        relationship="cross_modal",
        quality={"optical": 0.9, "sar": 0.8},
        image_2={
            "filename": "b.tif",
            "modality": "sar",
            "sensor": "Generic SAR",
            "gsd_m": 10.0,
            "crs": "EPSG:32643",
            "acquisition_date": "2024-01-02 10:00:00",
        },
    ))


def _empty_profile():
    """No images at all."""
    return adapt_profile({
        "image_count": 0,
        "query": "test",
        "spatial_overlap": None,
        "coregistration_score": None,
        "relationship": "unknown",
        "quality": {},
        "warnings": ["No images"],
        "task_executable": False,
    })


def _two_optical_low_overlap():
    return adapt_profile(_make_profile(
        image_count=2,
        spatial_overlap=0.3,
        coregistration_score=0.4,
        relationship="multi_temporal",
        image_2={
            "filename": "b.tif",
            "modality": "optical",
            "sensor": "Generic Optical",
            "gsd_m": 10.0,
            "crs": "EPSG:32643",
            "acquisition_date": "2024-02-01 10:00:00",
        },
    ))


# ══════════════════════════════════════════════════════════════════
# Test 1: Structural — every entry has the required keys & types
# ══════════════════════════════════════════════════════════════════

class TestRegistryStructure:
    EXPECTED_TASKS = {
        "single_image_vqa",
        "captioning",
        "grounding",
        "change_detection",
        "cross_modal_fusion",
    }

    def test_all_five_tasks_present(self):
        assert set(ENGINE_REGISTRY.keys()) == self.EXPECTED_TASKS

    @pytest.mark.parametrize("task_type", EXPECTED_TASKS)
    def test_entry_has_callable_handler(self, task_type):
        assert callable(ENGINE_REGISTRY[task_type]["handler"])

    @pytest.mark.parametrize("task_type", EXPECTED_TASKS)
    def test_entry_has_callable_precondition(self, task_type):
        assert callable(ENGINE_REGISTRY[task_type]["precondition"])

    @pytest.mark.parametrize("task_type", EXPECTED_TASKS)
    def test_entry_has_nonempty_precondition_msg(self, task_type):
        msg = ENGINE_REGISTRY[task_type]["precondition_msg"]
        assert isinstance(msg, str) and len(msg) > 0


# ══════════════════════════════════════════════════════════════════
# Test 2: Preconditions — one pass and one fail per task type
# ══════════════════════════════════════════════════════════════════

class TestSingleImageVqaPrecondition:
    def test_passes_with_one_image(self):
        assert ENGINE_REGISTRY["single_image_vqa"]["precondition"](_single_optical())

    def test_fails_with_no_images(self):
        assert not ENGINE_REGISTRY["single_image_vqa"]["precondition"](_empty_profile())


class TestCaptioningPrecondition:
    def test_passes_with_one_image(self):
        assert ENGINE_REGISTRY["captioning"]["precondition"](_single_optical())

    def test_fails_with_no_images(self):
        assert not ENGINE_REGISTRY["captioning"]["precondition"](_empty_profile())


class TestGroundingPrecondition:
    def test_passes_with_one_image(self):
        assert ENGINE_REGISTRY["grounding"]["precondition"](_single_optical())

    def test_fails_with_two_images(self):
        assert not ENGINE_REGISTRY["grounding"]["precondition"](_two_optical_high_overlap())


class TestChangeDetectionPrecondition:
    def test_passes_with_two_same_modality_high_overlap(self):
        assert ENGINE_REGISTRY["change_detection"]["precondition"](
            _two_optical_high_overlap()
        )

    def test_fails_with_one_image(self):
        assert not ENGINE_REGISTRY["change_detection"]["precondition"](
            _single_optical()
        )

    def test_fails_with_low_overlap(self):
        assert not ENGINE_REGISTRY["change_detection"]["precondition"](
            _two_optical_low_overlap()
        )

    def test_fails_with_different_modalities(self):
        assert not ENGINE_REGISTRY["change_detection"]["precondition"](
            _optical_sar_high_overlap()
        )


class TestCrossModalFusionPrecondition:
    def test_passes_with_optical_sar_high_overlap(self):
        assert ENGINE_REGISTRY["cross_modal_fusion"]["precondition"](
            _optical_sar_high_overlap()
        )

    def test_fails_with_same_modality(self):
        assert not ENGINE_REGISTRY["cross_modal_fusion"]["precondition"](
            _two_optical_high_overlap()
        )

    def test_fails_with_one_image(self):
        assert not ENGINE_REGISTRY["cross_modal_fusion"]["precondition"](
            _single_optical()
        )


# ══════════════════════════════════════════════════════════════════
# Test 3: Handler stubs return the required dict shape
# ══════════════════════════════════════════════════════════════════

class TestHandlerStubs:
    @pytest.mark.parametrize("task_type", list(ENGINE_REGISTRY.keys()))
    def test_handler_returns_required_keys(self, task_type):
        handler = ENGINE_REGISTRY[task_type]["handler"]
        result = handler({})
        assert isinstance(result, dict)
        assert "answer" in result
        assert "confidence" in result
        assert "bbox" in result
