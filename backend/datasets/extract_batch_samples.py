import os
import torch
import pandas as pd
import zipfile
import rasterio
import numpy as np
import time
import sys
from huggingface_hub import hf_hub_download

# We'll extract 100 samples
NUM_SAMPLES_TO_EXTRACT = 100
CACHE_DIR = "/Users/sukesh/Desktop/satquery/backend/data/features"
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs("/tmp/ben_test", exist_ok=True)

sys.path.append('/Users/sukesh/Desktop/satquery')
from backend.mc4c.croma import PretrainedCROMA

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

def run_extraction():
    print("Loading official BigEarthNet.txt metadata...")
    official_meta_path = hf_hub_download(repo_id='BIFOLD-BigEarthNetv2-0/BigEarthNet.txt', filename='BigEarthNet.txt.parquet', repo_type='dataset')
    official_df = pd.read_parquet(official_meta_path)
    
    zip_path = hf_hub_download(repo_id='ranjeetgupta/Cross-Modal_Retrieval_BigEarthNet_14K_S1_and_S2', filename='BigEarthNet_14K.zip', repo_type='dataset')
    print("Opening ZIP...")
    
    # Load CROMA once
    print("Loading CROMA...")
    checkpoint_path = hf_hub_download(repo_id='antofuller/CROMA', filename='CROMA_base.pt')
    model = PretrainedCROMA(pretrained_path=checkpoint_path, size='base', modality='both', image_resolution=120)
    model.eval()
    
    count = 0
    t0 = time.time()
    
    with zipfile.ZipFile(zip_path, 'r') as z:
        names = z.namelist()
        s1_files = {os.path.basename(n).replace('.tif', ''): n for n in names if '/BigEarthNet-S1/' in n and n.endswith('.tif')}
        s2_files = {os.path.basename(n).replace('.tif', ''): n for n in names if '/BigEarthNet-S2/' in n and n.endswith('.tif')}
        
        for _, row in official_df.iterrows():
            if count >= NUM_SAMPLES_TO_EXTRACT:
                break
                
            s1 = row['s1_name']
            s2 = row['patch_id']
            sample_id = row['ID']
            
            cache_path = os.path.join(CACHE_DIR, f"sample_{sample_id}.pt")
            if os.path.exists(cache_path):
                # Skip already cached
                continue
            
            if s1 in s1_files and s2 in s2_files:
                z.extract(s1_files[s1], path='/tmp/ben_test')
                z.extract(s2_files[s2], path='/tmp/ben_test')
                
                extracted_s1 = os.path.join('/tmp/ben_test', s1_files[s1])
                extracted_s2 = os.path.join('/tmp/ben_test', s2_files[s2])
                
                with rasterio.open(extracted_s1) as src:
                    s1_data = src.read()
                with rasterio.open(extracted_s2) as src:
                    s2_data = src.read()
                    
                s1_tensor = preprocess_s1(s1_data)
                s2_tensor = preprocess_s2(s2_data)
                
                with torch.no_grad():
                    out = model(SAR_images=s1_tensor, optical_images=s2_tensor)
                    
                torch.save({
                    'sample_id': sample_id,
                    'optical_encodings': out['optical_encodings'],
                    'SAR_encodings': out['SAR_encodings'],
                    'joint_encodings': out['joint_encodings'],
                    's1_name': s1,
                    's2_name': s2
                }, cache_path)
                
                # Cleanup
                os.remove(extracted_s1)
                os.remove(extracted_s2)
                
                count += 1
                if count % 10 == 0:
                    print(f"Extracted {count}/{NUM_SAMPLES_TO_EXTRACT} samples...")
                    
    print(f"Extraction completed in {time.time() - t0:.2f}s")

if __name__ == "__main__":
    run_extraction()
