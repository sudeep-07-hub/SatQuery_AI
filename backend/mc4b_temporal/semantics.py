"""
semantics.py — Change-type classification for MC4B.

Converts raw binary/semantic change masks + class logits into
human-readable change-type label sets (e.g. "built-up area increased").
"""

import torch
import numpy as np
from typing import List, Dict, Optional

from . import config


def extract_change_types(
    binary_mask: torch.Tensor,
    semantic_logits: Optional[torch.Tensor] = None,
    threshold: float = 0.5,
) -> List[Dict]:
    """
    Extract human-readable change-type labels from model outputs.

    Args:
        binary_mask: (B, 1, H, W) binary change probabilities.
        semantic_logits: (B, num_classes, H, W) per-class logits, or None.
        threshold: Probability threshold for binary change.

    Returns:
        List of dicts per batch item:
        [
            {
                "has_change": bool,
                "changed_pixel_fraction": float,
                "change_types": ["built_up_gain", ...],
                "primary_change": "built_up_gain" or "no_change",
                "description": "built-up area increased"
            },
            ...
        ]
    """
    results = []
    batch_size = binary_mask.shape[0]

    for b in range(batch_size):
        mask_b = binary_mask[b, 0]  # (H, W)
        changed_pixels = (mask_b > threshold).float()
        changed_fraction = changed_pixels.mean().item()
        has_change = changed_fraction > 0.01  # at least 1% changed

        change_types = []
        primary_change = "no_change"

        if has_change and semantic_logits is not None:
            sem_b = semantic_logits[b]  # (num_classes, H, W)
            # Only look at pixels flagged as changed
            change_mask_bool = changed_pixels.bool()

            if change_mask_bool.any():
                # Get class predictions in changed region
                class_preds = sem_b.argmax(dim=0)  # (H, W)
                changed_classes = class_preds[change_mask_bool]

                # Count pixels per class
                for cls_idx, cls_name in enumerate(config.CHANGE_CLASSES):
                    if cls_name == "no_change":
                        continue
                    count = (changed_classes == cls_idx).sum().item()
                    if count > 0:
                        change_types.append(cls_name)

                if change_types:
                    # Primary = most frequent class
                    class_counts = {}
                    for cls_idx, cls_name in enumerate(config.CHANGE_CLASSES):
                        if cls_name != "no_change":
                            class_counts[cls_name] = (
                                (changed_classes == cls_idx).sum().item()
                            )
                    primary_change = max(class_counts, key=class_counts.get)
                else:
                    primary_change = "unclassified_change"
                    change_types = ["unclassified_change"]
        elif has_change:
            # No semantic head — binary change only
            change_types = ["binary_change"]
            primary_change = "binary_change"

        description = _label_to_description(primary_change)

        results.append(
            {
                "has_change": has_change,
                "changed_pixel_fraction": round(changed_fraction, 4),
                "change_types": change_types,
                "primary_change": primary_change,
                "description": description,
            }
        )

    return results


def _label_to_description(label: str) -> str:
    """Convert a machine label to a human-readable description."""
    mapping = {
        "no_change": "No significant change detected",
        "built_up_gain": "Built-up area increased",
        "built_up_loss": "Built-up area decreased",
        "vegetation_loss": "Vegetation cover decreased",
        "vegetation_gain": "Vegetation cover increased",
        "binary_change": "Change detected (unclassified)",
        "unclassified_change": "Change detected (unclassified)",
    }
    return mapping.get(label, f"Change detected: {label}")
