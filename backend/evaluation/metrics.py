import re
from typing import Any

class Metric:
    def __init__(self, name: str):
        self.name = name

    def compute(self, prediction: Any, ground_truth: Any) -> float:
        raise NotImplementedError

class ExactMatchMetric(Metric):
    def __init__(self):
        super().__init__("exact_match")

    def compute(self, prediction: Any, ground_truth: Any) -> float:
        if prediction is None or ground_truth is None:
            return 0.0
            
        pred_str = str(prediction).lower().strip()
        
        # Strip punctuation
        pred_str = re.sub(r'[^\w\s]', '', pred_str)
        gt_str = str(ground_truth).lower().strip()
        gt_str = re.sub(r'[^\w\s]', '', gt_str)
        
        # Handle simple boolean cases in VQA
        if gt_str in ["yes", "no"]:
            if gt_str in pred_str.split():
                return 1.0
                
        return 1.0 if pred_str == gt_str else 0.0

class FallbackTextMetric(Metric):
    def __init__(self):
        super().__init__("text_similarity_placeholder")

    def compute(self, prediction: Any, ground_truth: Any) -> float:
        # A placeholder for BLEU/ROUGE/CIDEr until required libraries are integrated
        if prediction is None or ground_truth is None:
            return 0.0
        return -1.0 # indicating pending implementation

class SemanticGroundingMetric(Metric):
    def __init__(self):
        super().__init__("iou_semantic_grounding")

    def compute(self, prediction: Any, ground_truth: Any) -> float:
        # The current system only supports full-image footprints, not true semantic bounding boxes.
        # We must not generate fake metrics. 
        raise NotImplementedError("Semantic grounding evaluation is UNSUPPORTED because the core system currently relies on full-image footprints instead of object-level boxes.")

