import os
import torch
from torch.utils.data import Dataset
from typing import List, Dict, Any, Tuple

class BigEarthNetAdaptationDataset(Dataset):
    """
    Dataset that loads and pairs real BigEarthNet visual and Qwen3 text features.
    """
    def __init__(self, visual_dir: str, text_dir: str, required_sample_ids: List[str] = None):
        self.visual_dir = visual_dir
        self.text_dir = text_dir
        self.samples = []
        
        self.stats = {
            "text_records": 0,
            "visual_records": 0,
            "paired_records": 0,
            "missing_visual": 0,
            "missing_text": 0,
            "invalid_samples": 0
        }
        
        self._load_samples(required_sample_ids)
        
    def _load_samples(self, required_sample_ids: List[str] = None):
        if not os.path.exists(self.visual_dir) or not os.path.exists(self.text_dir):
            return
            
        visual_files = set(f for f in os.listdir(self.visual_dir) if f.startswith("sample_") and f.endswith(".pt"))
        text_files = set(f for f in os.listdir(self.text_dir) if f.startswith("qwen3_text_") and f.endswith(".pt"))
        
        self.stats["visual_records"] = len(visual_files)
        self.stats["text_records"] = len(text_files)
        
        # We index by sample_id
        text_by_sample_id = {}
        for f in text_files:
            sample_id = f.replace("qwen3_text_", "").replace(".pt", "")
            # Assuming format: qwen3_text_{sample_id}_{idx}.pt or qwen3_text_{sample_id}.pt
            # In our case it was saved as qwen3_text_{caption_id}.pt
            # Let's actually load the metadata to be robust if the filename isn't strictly the sample_id.
            # But loading all files might be slow for a huge directory.
            # For this task, we will load and check.
            try:
                data = torch.load(os.path.join(self.text_dir, f), map_location='cpu', weights_only=False)
                sid = str(data['sample_id'])
                if required_sample_ids and sid not in required_sample_ids:
                    continue
                if sid not in text_by_sample_id:
                    text_by_sample_id[sid] = []
                text_by_sample_id[sid].append(data)
            except Exception:
                self.stats["invalid_samples"] += 1
                
        visual_by_sample_id = {}
        for f in visual_files:
            try:
                data = torch.load(os.path.join(self.visual_dir, f), map_location='cpu', weights_only=False)
                sid = str(data['sample_id'])
                if required_sample_ids and sid not in required_sample_ids:
                    continue
                visual_by_sample_id[sid] = data
            except Exception:
                self.stats["invalid_samples"] += 1
                
        # Now pair them
        all_sids = set(text_by_sample_id.keys()).union(set(visual_by_sample_id.keys()))
        
        for sid in all_sids:
            if sid in text_by_sample_id and sid in visual_by_sample_id:
                # We can have multiple text captions for one image
                for text_data in text_by_sample_id[sid]:
                    self.samples.append({
                        "sample_id": sid,
                        "visual": visual_by_sample_id[sid],
                        "text": text_data
                    })
                    self.stats["paired_records"] += 1
            elif sid in text_by_sample_id:
                self.stats["missing_visual"] += 1
            elif sid in visual_by_sample_id:
                self.stats["missing_text"] += 1

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.samples[idx]
        
        # Extract the tensors we need for adaptation
        # Text pooled embedding is [2560]
        text_emb = item["text"]["pooled_embedding"]
        if text_emb.dim() == 2 and text_emb.size(0) == 1:
            text_emb = text_emb.squeeze(0)
            
        # Visual spatial tokens: [225, 768] (joint_encodings)
        # Note: the artifact might store it as [1, 225, 768]
        vis_emb = item["visual"]["joint_encodings"]
        if vis_emb.dim() == 3 and vis_emb.size(0) == 1:
            vis_emb = vis_emb.squeeze(0)
            
        return {
            "sample_id": item["sample_id"],
            "visual_features": vis_emb,
            "text_features": text_emb,
            "text_metadata": item["text"]  # to keep track of provenance
        }
