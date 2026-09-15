import os
import argparse
import pandas as pd
import zipfile
import rasterio
import numpy as np
import torch
import traceback
from huggingface_hub import hf_hub_download

from backend.mc4c.croma import PretrainedCROMA
from backend.datasets.bigearthnet.manifest import DatasetManifest
from backend.datasets.bigearthnet.pairing import validate_pair
from backend.adaptation.qwen3_feature_extractor import Qwen3FeatureExtractor, InsufficientMemoryError

def preprocess_s1(data):
    tensor = torch.from_numpy(data.astype(np.float32))
    return tensor.unsqueeze(0)

def preprocess_s2(data):
    tensor = torch.from_numpy(data.astype(np.float32))
    if tensor.shape[0] == 10:
        b01 = torch.zeros((1, tensor.shape[1], tensor.shape[2]), dtype=torch.float32)
        b09 = torch.zeros((1, tensor.shape[1], tensor.shape[2]), dtype=torch.float32)
        padded = torch.cat([b01, tensor[0:8], b09, tensor[8:10]], dim=0)
        tensor = padded
    if tensor.shape[1] != 120 or tensor.shape[2] != 120:
        import torch.nn.functional as F
        tensor = F.interpolate(tensor.unsqueeze(0), size=(120, 120), mode='bilinear', align_corners=False)
    else:
        tensor = tensor.unsqueeze(0)
    return tensor

def acquire_samples(target_samples: int = 10):
    cache_dir = "/Users/sukesh/Desktop/satquery/backend/data/adaptation/cache"
    metadata_dir = "/Users/sukesh/Desktop/satquery/backend/data/bigearthnet"
    metrics_dir = "/Users/sukesh/Desktop/satquery/backend/data/adaptation/metrics"
    features_dir = "/Users/sukesh/Desktop/satquery/backend/data/features"
    text_features_dir = "/Users/sukesh/Desktop/satquery/backend/data/text_features"
    
    os.makedirs(features_dir, exist_ok=True)
    os.makedirs(text_features_dir, exist_ok=True)
    os.makedirs("/tmp/ben_test", exist_ok=True)
    
    manifest = DatasetManifest(cache_dir, metadata_dir)
    
    # Load authoritative text
    official_meta_path = hf_hub_download(repo_id='BIFOLD-BigEarthNetv2-0/BigEarthNet.txt', filename='BigEarthNet.txt.parquet', repo_type='dataset')
    official_df = pd.read_parquet(official_meta_path)
    
    zip_path = hf_hub_download(repo_id='ranjeetgupta/Cross-Modal_Retrieval_BigEarthNet_14K_S1_and_S2', filename='BigEarthNet_14K.zip', repo_type='dataset')
    
    print("Loading CROMA...")
    checkpoint_path = hf_hub_download(repo_id='antofuller/CROMA', filename='CROMA_base.pt')
    croma = PretrainedCROMA(pretrained_path=checkpoint_path, size='base', modality='both', image_resolution=120)
    croma.eval()
    
    try:
        qwen3_extractor = Qwen3FeatureExtractor()
    except InsufficientMemoryError:
        print("Warning: Qwen3 extractor cannot load due to memory constraints. Will fail gracefully on extraction attempts.")
        qwen3_extractor = None
        
    count = 0
    with zipfile.ZipFile(zip_path, 'r') as z:
        names = z.namelist()
        s1_files = {os.path.basename(n).replace('.tif', ''): n for n in names if '/BigEarthNet-S1/' in n and n.endswith('.tif')}
        s2_files = {os.path.basename(n).replace('.tif', ''): n for n in names if '/BigEarthNet-S2/' in n and n.endswith('.tif')}
        candidate_count = 0
        for _, row in official_df.iterrows():
            if count >= target_samples or candidate_count >= 20:
                break
            candidate_count += 1
                
            sample_id = row['ID']
            s1 = row['s1_name']
            s2 = row['patch_id']
            caption = row['output']
            
            # The exact 771130 sample check
            visual_path = os.path.join(features_dir, f"sample_{sample_id}.pt")
            text_path = os.path.join(text_features_dir, f"qwen3_text_{sample_id}.pt")
            
            manifest.stats["candidate_samples"] += 1
            
            # 1. Acquire Visual Features (CROMA)
            if not os.path.exists(visual_path):
                if s1 in s1_files and s2 in s2_files:
                    try:
                        z.extract(s1_files[s1], path='/tmp/ben_test')
                        z.extract(s2_files[s2], path='/tmp/ben_test')
                        with rasterio.open(os.path.join('/tmp/ben_test', s1_files[s1])) as src:
                            s1_data = src.read()
                        with rasterio.open(os.path.join('/tmp/ben_test', s2_files[s2])) as src:
                            s2_data = src.read()
                            
                        s1_tensor = preprocess_s1(s1_data)
                        s2_tensor = preprocess_s2(s2_data)
                        
                        with torch.no_grad():
                            out = croma(SAR_images=s1_tensor, optical_images=s2_tensor)
                            
                        torch.save({
                            'sample_id': sample_id,
                            'optical_encodings': out['optical_encodings'],
                            'SAR_encodings': out['SAR_encodings'],
                            'joint_encodings': out['joint_encodings'],
                            's1_name': s1,
                            's2_name': s2,
                            'preprocessing': 'engineering compatibility padding for missing bands'
                        }, visual_path)
                        manifest.stats["successful_CROMA_extractions"] += 1
                        
                    except Exception as e:
                        manifest.add_rejection(sample_id, f"CROMA_failure: {str(e)}", "visual_extraction")
                        continue
                else:
                    manifest.add_rejection(sample_id, "missing_S1_or_S2", "imagery_validation")
                    continue
            else:
                manifest.stats["successful_CROMA_extractions"] += 1
                
            # 2. Acquire Text Features (Qwen3)
            if not os.path.exists(text_path):
                if qwen3_extractor is None:
                    manifest.add_rejection(sample_id, "Qwen3_failure: Insufficient memory", "text_extraction")
                    continue
                else:
                    try:
                        qwen3_extractor.extract_features(sample_id, caption, text_path)
                        manifest.stats["successful_Qwen3_extractions"] += 1
                    except Exception as e:
                        manifest.add_rejection(sample_id, f"Qwen3_failure: {str(e)}", "text_extraction")
                        continue
            else:
                manifest.stats["successful_Qwen3_extractions"] += 1
                
            # 3. Final validation
            v_data = torch.load(visual_path, map_location='cpu', weights_only=False)
            t_data = torch.load(text_path, map_location='cpu', weights_only=False)
            
            is_valid, reason = validate_pair(v_data, t_data)
            if is_valid:
                manifest.add_valid_pair(sample_id, visual_path, text_path)
                count += 1
            else:
                manifest.add_rejection(sample_id, reason, "pairing_validation")

    manifest.save()
    manifest.report_statistics(metrics_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_samples", type=int, default=10)
    args = parser.parse_args()
    acquire_samples(args.target_samples)
