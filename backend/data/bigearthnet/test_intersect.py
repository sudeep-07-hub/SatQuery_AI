import zipfile
import pandas as pd
from huggingface_hub import hf_hub_download

print("Downloading BigEarthNet.txt...")
df_path = hf_hub_download(repo_id='BIFOLD-BigEarthNetv2-0/BigEarthNet.txt', filename='BigEarthNet.txt.parquet', repo_type='dataset')
df = pd.read_parquet(df_path)

print("Opening ZIP...")
zip_path = hf_hub_download(repo_id='ranjeetgupta/Cross-Modal_Retrieval_BigEarthNet_14K_S1_and_S2', filename='BigEarthNet_14K.zip', repo_type='dataset')
with zipfile.ZipFile(zip_path, 'r') as z:
    names = z.namelist()
    
s1_names = {n.split('/')[-1].replace('.tif', '') for n in names if '/BigEarthNet-S1/' in n and n.endswith('.tif')}
s2_names = {n.split('/')[-1].replace('.tif', '') for n in names if '/BigEarthNet-S2/' in n and n.endswith('.tif')}

print(f"Physical S1 count: {len(s1_names)}")
print(f"Physical S2 count: {len(s2_names)}")

# The problem says: "Compute: common_physical_ids = S1_ids ∩ S2_ids"
# But s1_names and s2_names have completely different formats!
# Let's intersect via the DF mapping.
s1_in_df = df[df['s1_name'].isin(s1_names)]
s2_in_df = df[df['patch_id'].isin(s2_names)]

print(f"S1 names found in DF: {len(s1_in_df)}")
print(f"S2 names found in DF: {len(s2_in_df)}")

# Common IDs
common_ids = set(s1_in_df['ID']) & set(s2_in_df['ID'])
print(f"Common valid BigEarthNet IDs: {len(common_ids)}")

import json
with open('intersection.json', 'w') as f:
    json.dump(list(common_ids)[:20], f)
