from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field

class OpticalEvidence(BaseModel):
    source_id: str
    feature_vector: List[float] = Field(..., description="1D embedding from CROMA")
    spatial_features: Optional[List[List[List[float]]]] = Field(None, description="Optional patch-level features")

class SAREvidence(BaseModel):
    source_id: str
    feature_vector: List[float] = Field(..., description="1D embedding from CROMA")
    spatial_features: Optional[List[List[List[float]]]] = Field(None, description="Optional patch-level features")

class JointEvidence(BaseModel):
    fused_vector: List[float] = Field(..., description="Query-conditioned cross-modal embedding")
    attention_weights: Optional[Dict[str, Any]] = None

class SemanticMetadata(BaseModel):
    optical_tags: List[str] = Field(..., description="Tags from BigEarthNet v2.0 classifier")
    sar_tags: List[str] = Field(..., description="Tags from BigEarthNet v2.0 classifier")
    joint_tags: List[str] = Field(..., description="Consensus or highest confidence tags")

class CrossModalFusionResult(BaseModel):
    optical_evidence: OpticalEvidence
    sar_evidence: SAREvidence
    joint_evidence: JointEvidence
    semantic_metadata: SemanticMetadata
    fusion_method: str = Field(default="CROMA_query_conditioned", description="Method used to fuse modalities")
    verification_status: str = Field(..., description="E.g., MATCH, MISMATCH, PARTIAL")
    confidence_score: float = Field(..., ge=0.0, le=1.0)
