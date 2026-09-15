import os
import torch
import pandas as pd
import zipfile
from huggingface_hub import hf_hub_download
import rasterio
import numpy as np
import time

print("Downloading subset metadata...")
t0 = time.time()
meta_path = hf_hub_download(repo_id='ranjeetgupta/Cross-Modal_Retrieval_BigEarthNet_14K_S1_and_S2', filename='metadata.parquet', repo_type='dataset')
df = pd.read_parquet(meta_path)
print(f"Downloaded metadata in {time.time()-t0:.2f}s")

print("Downloading 3.1GB zip (this will take ~1-2 mins)...")
t0 = time.time()
zip_path = hf_hub_download(repo_id='ranjeetgupta/Cross-Modal_Retrieval_BigEarthNet_14K_S1_and_S2', filename='BigEarthNet_14K.zip', repo_type='dataset')
print(f"Downloaded zip in {time.time()-t0:.2f}s")

with zipfile.ZipFile(zip_path, 'r') as z:
    names = z.namelist()
    s1_files = [n for n in names if '/BigEarthNet-S1/' in n and n.endswith('.tif')]
    s2_files = [n for n in names if '/BigEarthNet-S2/' in n and n.endswith('.tif')]
    
    print(f"Found {len(s1_files)} S1 files and {len(s2_files)} S2 files.")
    
    official_meta_path = hf_hub_download(repo_id='BIFOLD-BigEarthNetv2-0/BigEarthNet.txt', filename='BigEarthNet.txt.parquet', repo_type='dataset')
    official_df = pd.read_parquet(official_meta_path)
    
    print("Finding a paired match...")
    s1_basenames = {os.path.basename(f).replace('.tif', ''): f for f in s1_files}
    s2_basenames = {os.path.basename(f).replace('.tif', ''): f for f in s2_files}
    
    paired_sample = None
    for _, row in official_df.iterrows():
        s1 = row['s1_name']
        s2 = row['patch_id'] 
        
        if s1 in s1_basenames and s2 in s2_basenames:
            paired_sample = {
                's1_name': s1,
                's2_name': s2,
                's1_path': s1_basenames[s1],
                's2_path': s2_basenames[s2],
                'id': row['ID']
            }
            break
            
    if not paired_sample:
        print("CRITICAL FAILURE: Could not establish S1<->S2 correspondence.")
        exit(1)
        
    print("Selected Sample:", paired_sample)
    
    t0 = time.time()
    z.extract(paired_sample['s1_path'], path='/tmp/ben_test')
    z.extract(paired_sample['s2_path'], path='/tmp/ben_test')
    print(f"Files extracted in {time.time()-t0:.2f}s")
    
    extracted_s1 = os.path.join('/tmp/ben_test', paired_sample['s1_path'])
    extracted_s2 = os.path.join('/tmp/ben_test', paired_sample['s2_path'])

def inspect_tiff(path, modality):
    with rasterio.open(path) as src:
        data = src.read()
        print(f"--- {modality} ---")
        print("Shape:", data.shape)
        print("Dtype:", data.dtype)
        print("CRS:", src.crs)
        return data

s1_data = inspect_tiff(extracted_s1, 'Sentinel-1')
s2_data = inspect_tiff(extracted_s2, 'Sentinel-2')

def preprocess_s1(data):
    tensor = torch.from_numpy(data.astype(np.float32))
    return tensor.unsqueeze(0)

def preprocess_s2(data):
    tensor = torch.from_numpy(data.astype(np.float32))
    if data.shape[1] != 120 or data.shape[2] != 120:
        import torch.nn.functional as F
        tensor = F.interpolate(tensor.unsqueeze(0), size=(120, 120), mode='bilinear', align_corners=False)
    else:
        tensor = tensor.unsqueeze(0)
    return tensor

s1_tensor = preprocess_s1(s1_data)
s2_tensor = preprocess_s2(s2_data)
print(f"S1 Tensor Shape: {s1_tensor.shape}")
print(f"S2 Tensor Shape: {s2_tensor.shape}")

import sys
sys.path.append('/Users/sukesh/Desktop/satquery')
from backend.mc1.format_validator import validate_format

with open(extracted_s1, 'rb') as f:
    v, r, fmt = validate_format(extracted_s1, f.read())
    print("MC1 S1 Valid:", v, fmt)
with open(extracted_s2, 'rb') as f:
    v, r, fmt = validate_format(extracted_s2, f.read())
    print("MC1 S2 Valid:", v, fmt)

print("Loading CROMA...")
from backend.mc4c.croma import PretrainedCROMA

checkpoint_path = hf_hub_download(repo_id='antofuller/CROMA', filename='CROMA_base.pt')

model = PretrainedCROMA(pretrained_path=checkpoint_path, size='base', modality='both', image_resolution=120)
model.eval()

t0 = time.time()
with torch.no_grad():
    out = model(SAR_images=s1_tensor, optical_images=s2_tensor)
print(f"CROMA Inference in {time.time()-t0:.2f}s")

print("CROMA Optical Shape:", out['optical_encodings'].shape)
print("CROMA SAR Shape:", out['SAR_encodings'].shape)
print("CROMA Joint Shape:", out['joint_encodings'].shape)
print("Optical NaN/Inf?", torch.isnan(out['optical_encodings']).any().item(), torch.isinf(out['optical_encodings']).any().item())

os.makedirs('/Users/sukesh/Desktop/satquery/backend/data/features', exist_ok=True)
cache_path = f"/Users/sukesh/Desktop/satquery/backend/data/features/sample_{paired_sample['id']}.pt"
t0 = time.time()
torch.save({
    'sample_id': paired_sample['id'],
    'optical_encodings': out['optical_encodings'],
    'SAR_encodings': out['SAR_encodings'],
    'joint_encodings': out['joint_encodings'],
    's1_name': paired_sample['s1_name'],
    's2_name': paired_sample['s2_name']
}, cache_path)
print(f"Features cached in {time.time()-t0:.2f}s. Size: {os.path.getsize(cache_path)} bytes")

reloaded = torch.load(cache_path)
print("Token 42 spatial identity preserved:", torch.allclose(out['optical_encodings'][0, 42], reloaded['optical_encodings'][0, 42]))

print("SUCCESS")
