import os
import hashlib
from huggingface_hub import hf_hub_download

def download_croma():
    print("Downloading CROMA_base.pt...")
    path = hf_hub_download(repo_id="antofuller/CROMA", filename="CROMA_base.pt", local_dir="mc4c/weights")
    
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        md5 = hashlib.md5(f.read()).hexdigest()
        
    print(f"Downloaded to: {path}")
    print(f"Size: {size} bytes")
    print(f"MD5: {md5}")

if __name__ == "__main__":
    download_croma()
