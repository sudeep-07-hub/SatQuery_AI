"""
result.py — MC4B Temporal Inference Result schemas.

Defines the explicit boundary for RawTemporalOutput and TemporalResult,
avoiding semantic fabrication before Task 6.4/6.5.
"""

from typing import Dict, Any, Optional, List, Union
from pydantic import BaseModel, Field

class RawTemporalOutput(BaseModel):
    """
    Direct structural wrapper around the genuine ChangeMamba raw tensor outputs.
    """
    binary_mask: Optional[List] = None  # nested list or flat depending on serialization
    semantic_logits: Optional[List] = None
    confidence: Optional[float] = None
    output_shape: List[int]
    dtype: str
    device: str
    runtime_ms: Optional[float] = None


class TemporalResult(BaseModel):
    """
    Structured outcome of the Temporal Pipeline, encapsulating raw output
    alongside preserved geospatial metadata.
    """
    status: str
    reason: Optional[str] = None
    task_type: str = "change_detection"
    
    before_observation_id: Optional[str] = None
    after_observation_id: Optional[str] = None
    
    model_id: Optional[str] = None
    model_revision: Optional[str] = None
    
    raw_output: Optional[RawTemporalOutput] = None
    
    # Preserved spatial metadata to reconstruct geographic regions
    crs: Optional[str] = None
    affine_transform: Optional[List[float]] = None
    gsd_m: Optional[float] = None
    raster_width: Optional[int] = None
    raster_height: Optional[int] = None

    def dict_contract(self) -> dict:
        """
        Converts the formal Pydantic schema into the dictionary format
        expected by the broader MC3 -> MC5 architecture.
        """
        return self.model_dump(exclude_none=True)
