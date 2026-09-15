import pytest
import sys
from unittest.mock import patch, MagicMock

from qwen import QwenConfig, Qwen3Inference

def test_import_no_load():
    # TEST 1 - Importing does not load the model
    # If it loaded, is_loaded would be True
    inf = Qwen3Inference()
    assert inf.is_loaded is False
    assert inf.model is None
    assert inf.tokenizer is None

def test_configuration():
    # TEST 2 - Configuration is valid
    config = QwenConfig()
    assert config.MODEL_ID == "Qwen/Qwen3-4B-Instruct-2507"
    assert config.DEFAULT_MAX_NEW_TOKENS == 256
    
def test_error_handling_pre_init():
    # TEST 7 - Calling generation before initialization produces a controlled error
    inf = Qwen3Inference()
    result = inf.generate("Test prompt")
    
    assert result["status"] == "error"
    assert "Model not loaded" in result["error"]
    assert result["response"] is None

def test_error_handling_empty_prompt():
    inf = Qwen3Inference()
    # Mock loaded state for this test
    inf.model = MagicMock()
    inf.tokenizer = MagicMock()
    
    result = inf.generate("   ")
    assert result["status"] == "error"
    assert "Empty prompt" in result["error"]

@patch("transformers.AutoTokenizer")
@patch("transformers.AutoModelForCausalLM")
def test_initialization_mock(mock_model, mock_tokenizer):
    # TEST 3 - Initialization logic test (mocked)
    inf = Qwen3Inference()
    inf.load(device="cpu")
    
    assert inf.is_loaded is True
    mock_tokenizer.from_pretrained.assert_called_once_with(
        "Qwen/Qwen3-4B-Instruct-2507", trust_remote_code=True
    )
    mock_model.from_pretrained.assert_called_once()
    
    # Test unload
    inf.unload()
    assert inf.is_loaded is False

@patch("transformers.AutoTokenizer")
@patch("transformers.AutoModelForCausalLM")
def test_generation_config(mock_model, mock_tokenizer):
    # TEST 6 - Configured parameters passed
    inf = Qwen3Inference()
    
    # Setup mocks
    inf.model = MagicMock()
    inf.tokenizer = MagicMock()
    inf.tokenizer.apply_chat_template.return_value = "Formatted prompt"
    inf.tokenizer.return_value = MagicMock(to=MagicMock(return_value=MagicMock(input_ids=[[1, 2, 3]])))
    inf.tokenizer.batch_decode.return_value = ["Mock response"]
    
    inf.model.generate.return_value = [[1, 2, 3, 4, 5]]
    
    result = inf.generate("Test", max_new_tokens=100, temperature=0.5)
    
    assert result["status"] == "ok"
    assert result["response"] == "Mock response"
    assert result["metadata"]["max_new_tokens"] == 100
    assert result["metadata"]["temperature"] == 0.5
    inf.model.generate.assert_called_once()
    
    _, kwargs = inf.model.generate.call_args
    assert kwargs["max_new_tokens"] == 100
    assert kwargs["temperature"] == 0.5

@pytest.mark.skipif(True, reason="ENVIRONMENT/BLOCKED: Live model download requires 8GB and may timeout test sandbox.")
def test_live_smoke():
    # TEST 5 - Smoke inference
    import time
    
    start_load = time.time()
    inf = Qwen3Inference()
    inf.load()
    load_time = time.time() - start_load
    
    start_gen = time.time()
    result = inf.generate("Say hello in one sentence.")
    gen_time = time.time() - start_gen
    
    assert result["status"] == "ok"
    assert len(result["response"]) > 0
    
    print(f"Load time: {load_time:.2f}s")
    print(f"Gen time: {gen_time:.2f}s")
    
    inf.unload()
