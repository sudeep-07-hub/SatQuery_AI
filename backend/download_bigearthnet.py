import os
import hashlib
from huggingface_hub import hf_hub_download

print("Downloading BigEarthNet v2.0 weights (model.safetensors)...")

path = hf_hub_download(repo_id="BIFOLD-BigEarthNetv2-0/resnet50-all-v0.2.0", filename="model.safetensors", local_dir="mc4c/weights")
size = os.path.getsize(path)

with open(path, "rb") as f:
    md5 = hashlib.md5(f.read()).hexdigest()

print(f"Downloaded to: {path}")
print(f"Size: {size} bytes")
print(f"MD5: {md5}")

from mc4c.reben_publication.BENv2_utils import NEW_LABELS
print(f"Resolved Class Count: {len(NEW_LABELS)}")
print(f"Classes: {NEW_LABELS}")
