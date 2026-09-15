from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, model_serializer
import torch

class QueryRepresentation(BaseModel):
    """
    An explicit annotated representation of a natural language query, pairing
    the structured semantic context (TaskSpec) with a neural embedding suitable
    for downstream token conditioning.
    """
    # Original Input
    original_query: str = Field(..., description="The original natural language query")
    
    # Semantic Context (from Phase 2 TaskSpec)
    primary_task: str
    target_entities: List[str] = Field(default_factory=list)
    required_modalities: List[str] = Field(default_factory=list)
    temporal_requirement: Optional[str] = "none"
    spatial_output_required: bool = False
    
    # Neural Representation
    embedding: Any = Field(..., description="Sentence embedding tensor (B, embedding_dim)")
    embedding_dim: int = Field(default=384, description="Dimensionality of the neural embedding")
    
    # Provenance
    encoder_name: str = Field(default="sentence-transformers/all-MiniLM-L6-v2")
    provenance: str = Field(default="QueryEncoder", description="Generating component")

    model_config = {"arbitrary_types_allowed": True}

    @model_serializer(mode='wrap')
    def serialize_metadata(self, handler) -> Dict[str, Any]:
        """
        Safely serializes metadata while omitting the raw neural embeddings.
        Prevents exploding audit logs with float arrays.
        """
        data = handler(self)
        if 'embedding' in data and isinstance(self.embedding, torch.Tensor):
            data['embedding'] = f"<Tensor shape={list(self.embedding.shape)} dtype={str(self.embedding.dtype)}>"
        return data
