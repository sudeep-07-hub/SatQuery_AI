import os
import json
import pandas as pd
from typing import List, Dict
from huggingface_hub import hf_hub_download
from backend.adaptation.schemas import BigEarthNetTextSample
from backend.adaptation.text_representation import FrozenQwenTextEncoder, masked_mean_pooling

class BigEarthNetTextDatasetBuilder:
    def __init__(self, visual_cache_dir: str = "/Users/sukesh/Desktop/satquery/backend/data/features"):
        self.visual_cache_dir = visual_cache_dir
        self.encoder = FrozenQwenTextEncoder(model_id="Qwen/Qwen3-4B-Instruct-2507")
        self.metadata_path = None
        self.metadata_df = None
        
    def load_metadata(self):
        print("Loading official BigEarthNet.txt metadata...")
        try:
            self.metadata_path = hf_hub_download(
                repo_id='BIFOLD-BigEarthNetv2-0/BigEarthNet.txt', 
                filename='BigEarthNet.txt.parquet', 
                repo_type='dataset'
            )
            self.metadata_df = pd.read_parquet(self.metadata_path)
            print(f"Loaded {len(self.metadata_df)} official records.")
        except Exception as e:
            print(f"Failed to load metadata: {e}")
            raise
            
    def prepare_dataset_for_sample(self, sample_id: int) -> List[BigEarthNetTextSample]:
        """
        Creates dataset records for a single sample ID based on available captions.
        """
        if self.metadata_df is None:
            self.load_metadata()
            
        # Extract rows for this sample ID
        rows = self.metadata_df[self.metadata_df['ID'] == sample_id]
        
        if len(rows) == 0:
            print(f"Sample {sample_id} not found in metadata.")
            return []
            
        # Determine visual cache path
        visual_path = os.path.join(self.visual_cache_dir, f"sample_{sample_id}.pt")
        if not os.path.exists(visual_path):
            print(f"Visual feature cache not found at {visual_path}. Cannot join.")
            return []
            
        samples = []
        # In BigEarthNet.txt, there might be multiple rows or a 'caption' column.
        for idx, row in rows.iterrows():
            text = row.get('output', None)
            if not text:
                continue
                
            sample = BigEarthNetTextSample(
                sample_id=str(sample_id),
                annotation_id=f"{sample_id}_{idx}",
                text=text,
                annotation_type=row.get('type', 'captioning'),
                split=row.get('split', 'validation'),
                visual_feature_path=visual_path,
                s1_name=row.get('s1_name', 'unknown'),
                s2_name=row.get('patch_id', 'unknown'),
                instruction=row.get('input', None),
                latitude=float(row.get('latitude')) if row.get('latitude') else None,
                longitude=float(row.get('longitude')) if row.get('longitude') else None,
                country=row.get('country', None),
                season=row.get('season', None),
                climate_zone=row.get('climate_zone', None)
            )
            samples.append(sample)
            
        return samples

    def process_and_cache(self, sample: BigEarthNetTextSample, output_dir: str = "/Users/sukesh/Desktop/satquery/backend/data/text"):
        """
        Computes the pooled text representation and caches it.
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Get hidden states (mocked if offline)
        hidden, mask = self.encoder.get_hidden_states(sample.text)
        
        # 2. Pool
        pooled = masked_mean_pooling(hidden, mask)
        
        # 3. Cache
        cache_file = os.path.join(output_dir, f"text_{sample.annotation_id}.json")
        record = sample.model_dump()
        record['pooled_dim'] = list(pooled.shape)
        # Note: We do not store the massive hidden states permanently, only the schema and shape info.
        
        with open(cache_file, 'w') as f:
            json.dump(record, f, indent=2)
            
        return record, pooled
