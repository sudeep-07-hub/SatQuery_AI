"""
demo_fixtures.py - Pre-configured deterministic configurations for the 4 final SIH scenarios.
"""

from typing import Dict, Any

def get_demo_fixture(demo_id: str) -> Dict[str, Any]:
    """
    Returns the deterministic configuration for a specific demo scenario.
    """
    fixtures = {
        "demo_d1_single_image_vqa": {
            "query": "What is in this image?",
            "expected_primary_task": "single_image_vqa",
            "expected_tools": ["single_image_vqa"],
            "execution_mode": "real", # Use real PaliGemma and Qwen3
            "mc1_profile": {
                "task_executable": True,
                "spatial_overlap": 1.0,
                "coregistration_score": 1.0,
                "image_1": {
                    "filename": "d1_optical.tif", 
                    "modality": "optical",
                    "crs": "EPSG:32643", 
                    "gsd_m": 1.0,
                    "acquisition_date": "2024-01-01T00:00:00Z"
                },
                "image_count": 1
            }
        },
        "demo_d2_multitool": {
            "query": "Describe the scene in optical and determine if there are changes between the two dates.",
            "expected_primary_task": "change_detection", # Compound task with subtasks
            "expected_tools": ["single_image_vqa", "temporal_change_analysis"],
            "execution_mode": "fixture", # Since ChangeMamba is unavailable, use fixture mode to show orchestration
            "mc1_profile": {
                "task_executable": True,
                "spatial_overlap": 0.95,
                "coregistration_score": 0.92,
                "image_1": {
                    "filename": "d2_optical_t1.tif", 
                    "modality": "optical",
                    "crs": "EPSG:32643", 
                    "gsd_m": 1.0,
                    "acquisition_date": "2023-01-01T00:00:00Z"
                },
                "image_2": {
                    "filename": "d2_optical_t2.tif", 
                    "modality": "optical",
                    "crs": "EPSG:32643", 
                    "gsd_m": 1.0,
                    "acquisition_date": "2024-01-01T00:00:00Z"
                },
                "image_count": 2
            }
        },
        "demo_d3_failure_recovery": {
            "query": "Analyze changes between these two dates.",
            "expected_primary_task": "change_detection",
            "expected_tools": ["temporal_change_analysis"],
            "execution_mode": "real", # Real mode will naturally fail on ChangeMamba (no CUDA), triggering recovery/MODEL_UNAVAILABLE
            "mc1_profile": {
                "task_executable": True,
                "spatial_overlap": 0.95,
                "coregistration_score": 0.92,
                "image_1": {
                    "filename": "d3_optical_t1.tif", 
                    "modality": "optical",
                    "crs": "EPSG:32643", 
                    "gsd_m": 1.0,
                    "acquisition_date": "2023-01-01T00:00:00Z"
                },
                "image_2": {
                    "filename": "d3_optical_t2.tif", 
                    "modality": "optical",
                    "crs": "EPSG:32643", 
                    "gsd_m": 1.0,
                    "acquisition_date": "2024-01-01T00:00:00Z"
                },
                "image_count": 2
            }
        },
        "demo_d4_input_rejection": {
            "query": "Analyze changes.",
            "expected_primary_task": "change_detection",
            "expected_tools": [],
            "execution_mode": "real",
            "mc1_profile": {
                "task_executable": False,
                "warnings": ["Insufficient overlap between images for change detection."],
                "spatial_overlap": 0.1,
                "coregistration_score": 0.1,
                "image_1": {
                    "filename": "d4_optical_t1.tif", 
                    "modality": "optical",
                    "crs": "EPSG:32643", 
                    "gsd_m": 1.0
                },
                "image_2": {
                    "filename": "d4_optical_t2.tif", 
                    "modality": "optical",
                    "crs": "EPSG:32643", 
                    "gsd_m": 1.0
                },
                "image_count": 2
            }
        }
    }
    return fixtures.get(demo_id)
