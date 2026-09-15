import os
import json
from datetime import datetime
from typing import List, Dict, Any

class DatasetManifest:
    def __init__(self, cache_dir: str, metadata_dir: str):
        self.cache_dir = cache_dir
        self.metadata_dir = metadata_dir
        self.paired_samples = []
        self.rejected_samples = []
        self.stats = {
            "candidate_samples": 0,
            "valid_pairs": 0,
            "rejected_samples": 0,
            "duplicate_samples": 0,
            "missing_text": 0,
            "missing_S1": 0,
            "missing_S2": 0,
            "CROMA_failures": 0,
            "Qwen3_failures": 0,
            "successful_CROMA_extractions": 0,
            "successful_Qwen3_extractions": 0
        }
        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(self.metadata_dir, exist_ok=True)
        
    def add_valid_pair(self, sample_id: str, visual_path: str, text_path: str):
        # Prevent duplicates
        if any(p["sample_id"] == sample_id for p in self.paired_samples):
            self.add_rejection(sample_id, "duplicate_ID", "pairing")
            self.stats["duplicate_samples"] += 1
            return
            
        self.paired_samples.append({
            "sample_id": sample_id,
            "visual_feature_path": visual_path,
            "text_feature_path": text_path,
            "status": "valid"
        })
        self.stats["valid_pairs"] += 1
        
    def add_rejection(self, sample_id: str, reason: str, stage: str):
        self.rejected_samples.append({
            "sample_id": sample_id,
            "rejection_reason": reason,
            "stage": stage,
            "timestamp": datetime.utcnow().isoformat()
        })
        self.stats["rejected_samples"] += 1
        if "Qwen3" in reason:
            self.stats["Qwen3_failures"] += 1
        elif "CROMA" in reason:
            self.stats["CROMA_failures"] += 1
        elif "text" in reason:
            self.stats["missing_text"] += 1
            
    def save(self):
        with open(os.path.join(self.cache_dir, "paired_manifest.json"), 'w') as f:
            json.dump(self.paired_samples, f, indent=2)
            
        with open(os.path.join(self.metadata_dir, "rejected_samples.json"), 'w') as f:
            json.dump(self.rejected_samples, f, indent=2)

    def report_statistics(self, metrics_dir: str):
        os.makedirs(metrics_dir, exist_ok=True)
        with open(os.path.join(metrics_dir, "dataset_expansion_report.json"), 'w') as f:
            json.dump(self.stats, f, indent=2)
            
        with open(os.path.join(metrics_dir, "dataset_expansion_report.md"), 'w') as f:
            f.write("# Dataset Expansion Statistics\n\n")
            for k, v in self.stats.items():
                f.write(f"- **{k}**: {v}\n")
