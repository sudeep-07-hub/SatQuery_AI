import os
import argparse
import torch
import psutil
from backend.datasets.bigearthnet_text import BigEarthNetTextDatasetBuilder
from backend.adaptation.qwen3_feature_extractor import Qwen3FeatureExtractor, InsufficientMemoryError

def main():
    parser = argparse.ArgumentParser(description="Extract Qwen3 text features for BigEarthNet samples.")
    parser.add_argument("--limit", type=int, default=1, help="Maximum number of samples to process.")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for feature extraction.")
    parser.add_argument("--output-dir", type=str, default="/Users/sukesh/Desktop/satquery/backend/data/text_features", help="Directory to save the extracted features.")
    parser.add_argument("--max-length", type=int, default=512, help="Maximum sequence length.")
    parser.add_argument("--start-sample-id", type=int, default=771130, help="The first sample ID to process.")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Check if the start sample feature already exists (for resume logic/skipping)
    # But for testing we might want to overwrite. Let's just create it.
    
    print("Initializing BigEarthNetTextDatasetBuilder...")
    dataset_builder = BigEarthNetTextDatasetBuilder()
    
    print("Loading metadata...")
    try:
        dataset_builder.load_metadata()
    except Exception as e:
        print(f"Failed to load metadata: {e}")
        return
        
    print(f"Searching for sample ID {args.start_sample_id}...")
    samples = dataset_builder.prepare_dataset_for_sample(args.start_sample_id)
    
    if not samples:
        print(f"Could not prepare dataset for sample {args.start_sample_id}.")
        return
        
    print(f"Found {len(samples)} valid captions for sample {args.start_sample_id}.")
    
    print("Initializing Qwen3FeatureExtractor...")
    extractor = Qwen3FeatureExtractor(max_length=args.max_length)
    
    try:
        print("Loading Qwen3 model...")
        extractor.load_model()
    except InsufficientMemoryError as e:
        print(f"\n[BLOCKED] Environment lacks sufficient resources to load the Qwen3 model:\n{e}")
        return
    except Exception as e:
        print(f"\n[ERROR] Failed to load model: {e}")
        return
        
    # We will process up to args.limit samples (for now we just have 1 sample's captions)
    processed_count = 0
    for sample in samples:
        if processed_count >= args.limit:
            break
            
        print(f"Extracting features for annotation {sample.annotation_id} (Sample {sample.sample_id})...")
        
        try:
            results = extractor.extract([sample.text])
            result = results[0]
            
            output_path = os.path.join(args.output_dir, f"qwen3_text_{sample.annotation_id}.pt")
            
            artifact = {
                "sample_id": sample.sample_id,
                "caption_id": sample.annotation_id,
                "caption": sample.text,
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
            
            torch.save(artifact, output_path)
            print(f"Successfully saved feature artifact to {output_path}")
            processed_count += 1
            
        except Exception as e:
            print(f"Failed to extract features for {sample.annotation_id}: {e}")
            
    print(f"Extraction complete. Processed {processed_count} samples.")

if __name__ == "__main__":
    main()
