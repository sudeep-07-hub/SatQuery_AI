"""
conftest.py — Shared test fixtures for MC4B tests.

Provides realistic MC1 profiles and synthetic image tensors.
"""

import pytest
import torch
import numpy as np


@pytest.fixture
def valid_mc1_profile():
    """A valid bi-temporal optical MC1 profile that passes all preconditions."""
    return {
        "image_count": 2,
        "image_1": {
            "filename": "img_t1.tif",
            "modality": "optical",
            "sensor": "Cartosat-2S",
            "gsd_m": 1.0,
            "crs": "EPSG:32643",
            "acquisition_date": "2024-01-15 10:00:00",
        },
        "image_2": {
            "filename": "img_t2.tif",
            "modality": "optical",
            "sensor": "Cartosat-2S",
            "gsd_m": 1.0,
            "crs": "EPSG:32643",
            "acquisition_date": "2024-06-15 10:00:00",
        },
        "spatial_overlap": 0.94,
        "coregistration_score": 0.91,
        "relationship": "bi_temporal",
        "quality": {"image_1": 0.89, "image_2": 0.90},
        "affine_transform": [1.0, 0, 300000.0, 0, -1.0, 4000000.0],
    }


@pytest.fixture
def mismatched_crs_profile(valid_mc1_profile):
    """MC1 profile with mismatched CRS between images."""
    profile = dict(valid_mc1_profile)
    profile["image_2"] = dict(profile["image_2"])
    profile["image_2"]["crs"] = "EPSG:32644"
    return profile


@pytest.fixture
def low_overlap_profile(valid_mc1_profile):
    """MC1 profile with insufficient spatial overlap."""
    profile = dict(valid_mc1_profile)
    profile["spatial_overlap"] = 0.3
    return profile


@pytest.fixture
def low_coreg_profile(valid_mc1_profile):
    """MC1 profile with low coregistration score."""
    profile = dict(valid_mc1_profile)
    profile["coregistration_score"] = 0.2
    return profile


@pytest.fixture
def single_image_profile():
    """MC1 profile with only one image."""
    return {
        "image_count": 1,
        "image_1": {
            "filename": "img_t1.tif",
            "modality": "optical",
            "sensor": "Cartosat-2S",
            "gsd_m": 1.0,
            "crs": "EPSG:32643",
            "acquisition_date": "2024-01-15 10:00:00",
        },
        "spatial_overlap": None,
        "coregistration_score": None,
        "relationship": "single_image",
        "quality": {"image_1": 0.89},
    }


@pytest.fixture
def sample_tensors():
    """A pair of synthetic 3-channel 256×256 image tensors."""
    torch.manual_seed(42)
    t1 = torch.rand(1, 3, 256, 256)
    t2 = torch.rand(1, 3, 256, 256)
    return {"t1": t1, "t2": t2}


@pytest.fixture
def identical_tensors():
    """Identical T1 and T2 (no change scenario)."""
    torch.manual_seed(42)
    t = torch.rand(1, 3, 256, 256)
    return {"t1": t.clone(), "t2": t.clone()}


@pytest.fixture
def sample_task_spec():
    """MC2 Structured Task Specification for change detection."""
    return {
        "primary_task": "change_detection",
        "target_entities": ["built-up area"],
        "required_operations": [
            "temporal_analysis",
            "area_computation",
            "spatial_localization",
        ],
        "required_modalities": ["optical"],
        "temporal_requirement": "bi_temporal_change",
        "spatial_output_required": True,
        "textual_output_required": True,
        "query": "Has built-up area increased?",
    }
