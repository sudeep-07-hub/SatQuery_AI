"""
Tool Registry Entry for PaliGemma VQA (MC4A)
"""

PALIGEMMA_VQA_TOOL = {
    "name": "PALIGEMMA_VQA",
    "task_capabilities": [
        "visual_question_answering",
        "attribute_query",
        "object_presence_check",
        "counting",
        "single_image_vqa"
    ],
    "supported_modalities": ["optical"], # SAR permitted but flagged with domain mismatch
    "input_constraints": {
        "image_count": 1,
        "quality": {"optical": 0.5} # Minimum quality score
    },
    "spatial_resolution_range": {
        "min_gsd_m": 0.0, 
        "max_gsd_m": 1000.0 # No strict limits since no localization is done
    },
    "temporal_requirements": ["none"], # Single timestamp only
    "output_types": [
        "textual_answer",
        "model_confidences",
        "spatial_evidence" # Full-image footprint only
    ],
    "expected_confidence_characteristics": {
        "typical_range": [0.1, 0.99],
        "known_failure_modes": [
            "domain_shift_on_sar",
            "hallucination_on_reverse_prompt",
            "poor_counting_above_5"
        ]
    },
    "computational_cost": {
        "label": "high",
        "flops_estimate": 3e9 # 3B parameters
    },
    "preconditions": [
        "image_count_1_or_resolvable",
        "modality_quality_gate"
    ],
    "postconditions": [
        "output_contains_textual_answer",
        "spatial_evidence_is_full_image"
    ]
}
