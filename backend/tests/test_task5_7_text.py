import os
import torch
import pytest
import json
from backend.datasets.bigearthnet_text import BigEarthNetTextDatasetBuilder
from backend.adaptation.text_representation import FrozenQwenTextEncoder, masked_mean_pooling
from backend.adaptation.projection import TextProjectionHead, VisualProjectionHead

@pytest.fixture
def builder():
    return BigEarthNetTextDatasetBuilder()

def test_bigearthnet_record_loading(builder):
    builder.load_metadata()
    assert builder.metadata_df is not None
    assert len(builder.metadata_df) > 0

def test_caption_selection_and_join(builder):
    # Sample 771130 exists in cache from Task 5.6
    samples = builder.prepare_dataset_for_sample(771130)
    assert len(samples) > 0
    sample = samples[0]
    
    # Verify split preservation
    assert sample.split in ['train', 'validation', 'test']
    
    # Verify exact textual caption is present
    assert isinstance(sample.text, str)
    assert len(sample.text) > 0

def test_tokenizer_configuration():
    encoder = FrozenQwenTextEncoder("Qwen/Qwen3-4B-Instruct-2507")
    tokens = encoder.tokenize("A Sentinel-2 image showing forest.")
    assert "input_ids" in tokens
    assert "attention_mask" in tokens

def test_masked_pooling():
    # Synthetic test
    hidden = torch.randn(2, 10, 2560)
    mask = torch.ones(2, 10)
    mask[0, 5:] = 0 # Mask out half of sequence 1
    
    pooled = masked_mean_pooling(hidden, mask)
    assert pooled.shape == (2, 2560)
    
    # Verify it matches mean of non-masked tokens manually
    manual_mean = hidden[0, :5, :].mean(dim=0)
    assert torch.allclose(pooled[0], manual_mean, atol=1e-5)

def test_projection_interfaces_and_normalization():
    text_head = TextProjectionHead(in_dim=2560, out_dim=512)
    visual_head = VisualProjectionHead(in_dim=768, out_dim=512)
    
    # Verify no parameters updated (untrained test)
    params_before = [p.clone() for p in text_head.parameters()]
    
    text_feat = torch.randn(1, 2560)
    vis_feat = torch.randn(1, 768)
    
    out_text = text_head(text_feat)
    out_vis = visual_head(vis_feat)
    
    assert out_text.shape == (1, 512)
    assert out_vis.shape == (1, 512)
    
    # Verify normalization (L2 norm should be ~1.0)
    assert torch.allclose(torch.norm(out_text, p=2, dim=-1), torch.tensor([1.0]), atol=1e-5)
    assert torch.allclose(torch.norm(out_vis, p=2, dim=-1), torch.tensor([1.0]), atol=1e-5)
    
    params_after = list(text_head.parameters())
    for b, a in zip(params_before, params_after):
        assert torch.equal(b, a), "Parameters must not change (no optimizer/backward)."

def test_cache_serialization_and_reload(builder, tmp_path):
    samples = builder.prepare_dataset_for_sample(771130)
    if not samples:
        pytest.skip("Sample 771130 missing visual cache locally.")
    
    sample = samples[0]
    out_dir = str(tmp_path)
    record, pooled = builder.process_and_cache(sample, output_dir=out_dir)
    
    cache_file = os.path.join(out_dir, f"text_{sample.annotation_id}.json")
    assert os.path.exists(cache_file)
    
    with open(cache_file, 'r') as f:
        reloaded = json.load(f)
        
    assert reloaded['sample_id'] == '771130'
    assert reloaded['text'] == sample.text

def test_deterministic_repeatability():
    encoder = FrozenQwenTextEncoder("Qwen/Qwen3-4B-Instruct-2507")
    text = "Agricultural area with green fields."
    
    h1, m1 = encoder.get_hidden_states(text)
    p1 = masked_mean_pooling(h1, m1)
    
    h2, m2 = encoder.get_hidden_states(text)
    p2 = masked_mean_pooling(h2, m2)
    
    assert torch.allclose(p1, p2), "Repeatability verified within tolerance."
