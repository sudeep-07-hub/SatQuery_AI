import torch
from transformers import AutoTokenizer, AutoModel
from typing import List, Union

from .query_schema import QueryRepresentation
from qwen.schemas import TaskSpec

class QueryEncoder:
    """
    Produces a neural representation of a natural language query and pairs it
    with semantic context from Phase 2 TaskSpec.
    
    Loads explicitly on demand to prevent memory bloat during module imports.
    """
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None

    def load(self, device: str = "cpu"):
        """Explicitly load the text encoder into memory."""
        if self.model is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name).to(device)
            self.model.eval()

    def unload(self):
        """Explicitly remove the model from memory."""
        self.tokenizer = None
        self.model = None

    def encode(self, query: Union[str, List[str]], task_spec: TaskSpec, device: str = "cpu") -> QueryRepresentation:
        """
        Generates a mean-pooled sentence embedding for the query.
        Returns a QueryRepresentation merging the embedding with the TaskSpec semantic context.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() explicitly before encoding.")

        if isinstance(query, str):
            queries = [query]
        else:
            queries = query
            
        # Ensure model is in eval mode and deterministic
        self.model.eval()
        self.model.to(device)
        
        # Tokenize
        inputs = self.tokenizer(queries, return_tensors="pt", padding=True, truncation=True).to(device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        # Mean pooling to yield (B, 384)
        # Using the attention mask to properly mean pool (ignoring padding)
        token_embeddings = outputs.last_hidden_state
        attention_mask = inputs['attention_mask']
        
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        sentence_embeddings = sum_embeddings / sum_mask

        # For the string semantic fallback, we just take the first query representation if batched
        # but embedding tensor maintains full batch dimension
        return QueryRepresentation(
            original_query=task_spec.query,
            primary_task=task_spec.primary_task,
            target_entities=task_spec.target_entities,
            required_modalities=task_spec.required_modalities,
            temporal_requirement=task_spec.temporal_requirement,
            spatial_output_required=task_spec.spatial_output_required,
            embedding=sentence_embeddings,
            embedding_dim=sentence_embeddings.shape[1],
            encoder_name=self.model_name
        )
