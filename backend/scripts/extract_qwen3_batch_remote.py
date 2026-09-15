import os
import json
import torch
import traceback
import pandas as pd
from typing import List, Dict, Any
from huggingface_hub import hf_hub_download

from backend.adaptation.qwen3_feature_extractor import Qwen3FeatureExtractor, InsufficientMemoryError
from backend.datasets.bigearthnet.manifest import DatasetManifest
from backend.datasets.bigearthnet.pairing import validate_pair

def batch_extract_qwen3(batch_size: int = 1):
    cache_dir = "/Users/sukesh/Desktop/satquery/backend/data/adaptation/cache"
    metadata_dir = "/Users/sukesh/Desktop/satquery/backend/data/bigearthnet"
    metrics_dir = "/Users/sukesh/Desktop/satquery/backend/data/adaptation/metrics"
    features_dir = "/Users/sukesh/Desktop/satquery/backend/data/features"
    text_features_dir = "/Users/sukesh/Desktop/satquery/backend/data/text_features"
    
    os.makedirs(text_features_dir, exist_ok=True)
    
    manifest = DatasetManifest(cache_dir, metadata_dir)
    
    print("Loading selected samples...")
    with open(os.path.join(metadata_dir, "selected_samples.json"), "r") as f:
        selected_manifest = json.load(f)
        
    selected_ids = selected_manifest.get("selected_ids", [])
    if not selected_ids:
        print("No samples selected.")
        return
        
    print(f"Selected {len(selected_ids)} samples for batch extraction.")
    
    print("Loading BigEarthNet authoritative metadata...")
    official_meta_path = hf_hub_download(repo_id='BIFOLD-BigEarthNetv2-0/BigEarthNet.txt', filename='BigEarthNet.txt.parquet', repo_type='dataset')
    df = pd.read_parquet(official_meta_path)
    
    # Filter the dataframe for the selected samples
    df_selected = df[df['ID'].isin(selected_ids)].drop_duplicates(subset=['ID'])
    
    # We create the extractor. If memory is too constrained, this will predictably fail.
    print("Initializing Qwen3FeatureExtractor (Requires Remote GPU Environment or >= 8GB RAM)...")
    try:
        extractor = Qwen3FeatureExtractor(required_ram_gb=8.0, max_length=512)
        extractor.load_model()
        model_loaded = True
    except InsufficientMemoryError as e:
        print(f"\n[BLOCKED] Environment lacks sufficient resources to load the Qwen3 model:\n{e}")
        model_loaded = False
    except Exception as e:
        print(f"\n[ERROR] Unknown error loading Qwen3: {e}")
        model_loaded = False

    # Process all selected samples
    for idx, row in df_selected.iterrows():
        sample_id = row['ID']
        caption = row['output']
        
        text_path = os.path.join(text_features_dir, f"qwen3_text_{sample_id}.pt")
        visual_path = os.path.join(features_dir, f"sample_{sample_id}.pt")
        
        # Assume visual path exists because CROMA extraction succeeded in 5.9R
        if not os.path.exists(visual_path):
            manifest.add_rejection(sample_id, "Visual CROMA artifact missing", "croma_missing")
            continue
            
        if os.path.exists(text_path):
            print(f"Sample {sample_id}: Artifact already exists, skipping generation.")
            # Artifact is assumed valid; it will be checked below in pairing.
        else:
            if not model_loaded:
                manifest.add_rejection(sample_id, "Qwen3_failure: Insufficient physical memory for remote extraction", "qwen_oom")
            else:
                try:
                    # Execute extraction
                    results = extractor.extract([caption])
                    result = results[0]
                    
                    artifact = {
                        "sample_id": sample_id,
                        "caption_id": sample_id,  # In this dataset, sample ID is typically identical or maps 1:1
                        "caption": caption,
                        "model_id": result["model_id"],
                        "model_revision": result["model_revision"],
                        "layer": result["layer"],
                        "hidden_dim": result["hidden_dim"],
                        "pooled_embedding": result["pooled_embedding"],
                        "normalized_embedding": result["normalized_embedding"],
                        "token_count": result["token_count"],
                        "truncated": result["truncated"],
                        "pooling": result["pooling"],
                        "normalization": result["normalization"],
                        "device": result["device"],
                        "dtype": result["dtype"],
                        "source": "BigEarthNet.txt"
                    }
                    
                    torch.save(artifact, text_path)
                    print(f"Successfully saved feature artifact to {text_path}")
                except Exception as e:
                    manifest.add_rejection(sample_id, f"Qwen3 extraction failed: {str(e)}", "inference_failure")

        # Now, if both exist, run pairing validation
        if os.path.exists(visual_path) and os.path.exists(text_path):
            v_data = torch.load(visual_path, map_location='cpu', weights_only=False)
            t_data = torch.load(text_path, map_location='cpu', weights_only=False)
            
            is_valid, reason = validate_pair(v_data, t_data)
            if is_valid:
                manifest.add_valid_pair(sample_id, visual_path, text_path)
            else:
                manifest.add_rejection(sample_id, reason, "pairing_validation")

    manifest.save()
    
    # Compute readiness statistics
    paired_count = len(manifest.paired_samples)
    print(f"Total selected samples: {len(selected_ids)}")
    print(f"Successfully paired samples: {paired_count}")
    
    if paired_count >= 10:
        print("\nTask 5.10 READY (PASS)")
    else:
        print("\nTask 5.10 NOT READY (BLOCKED)")

if __name__ == "__main__":
    batch_extract_qwen3(batch_size=1)
