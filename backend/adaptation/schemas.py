from pydantic import BaseModel, Field
from typing import Optional

class BigEarthNetTextSample(BaseModel):
    sample_id: str = Field(..., description="BigEarthNet sample identity")
    annotation_id: str = Field(..., description="Unique annotation identity")
    text: str = Field(..., description="Original text annotation")
    annotation_type: str = Field(..., description="Type of annotation (e.g. captioning)")
    split: str = Field(..., description="Original dataset split")
    visual_feature_path: str = Field(..., description="Path to the frozen spatial feature cache")
    s1_name: str = Field(..., description="Sentinel-1 patch name")
    s2_name: str = Field(..., description="Sentinel-2 patch name")
    source_dataset: str = Field(default="BIFOLD-BigEarthNetv2-0/BigEarthNet.txt", description="Source of the metadata")
    text_model: str = Field(default="Qwen/Qwen3-4B-Instruct-2507", description="Text encoder model identifier")
    tokenizer: str = Field(default="Qwen/Qwen3-4B-Instruct-2507", description="Tokenizer identifier")
    pooling_method: str = Field(default="masked_mean_pooling", description="Method used to pool text hidden states")
    text_dimension: int = Field(default=2560, description="Dimension of text representation")
    visual_feature_shape: list = Field(default=[225, 768], description="Shape of the visual spatial feature")
    
    # Optional metadata
    instruction: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    country: Optional[str] = None
    season: Optional[str] = None
    climate_zone: Optional[str] = None
