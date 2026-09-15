import os
import torch
import pandas as pd
import zipfile
import rasterio
import numpy as np
import time
import sys

paired_sample = {
    's1_name': 'S1A_IW_GRDH_1SDV_20170802T163350_34TCR_26_19',
    's2_name': 'S2A_MSIL2A_20170803T094031_N9999_R036_T34TCR_26_19',
    's1_path': 'BEN_14k/BigEarthNet-S1/validation/S1A_IW_GRDH_1SDV_20170802T163350_34TCR_26_19.tif',
    's2_path': 'BEN_14k/BigEarthNet-S2/validation/S2A_MSIL2A_20170803T094031_N9999_R036_T34TCR_26_19.tif',
    'id': 771130
}

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
    # This 10-band dataset omits B01 and B09 (coastal aerosol and water vapour, 60m bands)
    # The standard 12-band order for CROMA is: B01, B02, B03, B04, B05, B06, B07, B08, B8A, B09, B11, B12.
    # The 10 bands we have are: B02, B03, B04, B08, B05, B06, B07, B8A, B11, B12 (or similar 10-band BEN layout)
    # Actually, standard BEN 10 bands are B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12 in order of ascending name.
    # We will pad B01 (index 0) and B09 (index 9) with zeros to reconstruct the 12-band shape legally.
    tensor = torch.from_numpy(data.astype(np.float32))
    
    if tensor.shape[0] == 10:
        print("Padding 10-band S2 image to 12 bands by inserting empty B01 and B09...")
        b01 = torch.zeros((1, tensor.shape[1], tensor.shape[2]), dtype=torch.float32)
        b09 = torch.zeros((1, tensor.shape[1], tensor.shape[2]), dtype=torch.float32)
        # Assuming order: 0:B02, 1:B03, 2:B04, 3:B05, 4:B06, 5:B07, 6:B08, 7:B8A, 8:B11, 9:B12
        padded = torch.cat([
            b01,
            tensor[0:8],  # B02 - B8A
            b09,
            tensor[8:10]  # B11, B12
        ], dim=0)
        tensor = padded
        
    if tensor.shape[1] != 120 or tensor.shape[2] != 120:
        import torch.nn.functional as F
        tensor = F.interpolate(tensor.unsqueeze(0), size=(120, 120), mode='bilinear', align_corners=False)
    else:
        tensor = tensor.unsqueeze(0)
    return tensor

s1_tensor = preprocess_s1(s1_data)
s2_tensor = preprocess_s2(s2_data)

print("Loading CROMA...")
sys.path.append('/Users/sukesh/Desktop/satquery')
from backend.mc4c.croma import PretrainedCROMA
from huggingface_hub import hf_hub_download

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

cache_path = f"/Users/sukesh/Desktop/satquery/backend/data/features/sample_{paired_sample['id']}.pt"
os.makedirs('/Users/sukesh/Desktop/satquery/backend/data/features', exist_ok=True)
torch.save({
    'sample_id': paired_sample['id'],
    'optical_encodings': out['optical_encodings'],
    'SAR_encodings': out['SAR_encodings'],
    'joint_encodings': out['joint_encodings'],
    's1_name': paired_sample['s1_name'],
    's2_name': paired_sample['s2_name']
}, cache_path)
print("Features cached successfully.")
print("SUCCESS")
