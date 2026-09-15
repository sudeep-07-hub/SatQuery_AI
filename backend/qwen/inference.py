import gc
import torch
from typing import Dict, Any, Optional

from .config import QwenConfig

class Qwen3Inference:
    """
    Isolated programmatic wrapper for the Qwen3 4B reasoning model.
    Handles explicit model loading, memory management, and clean generation.
    """
    
    def __init__(self, config: Optional[QwenConfig] = None):
        self.config = config or QwenConfig()
        self.model = None
        self.tokenizer = None
        
    @property
    def is_loaded(self) -> bool:
        return self.model is not None and self.tokenizer is not None
        
    def load(self, device: Optional[str] = None):
        """
        Explicitly loads the model into memory.
        Uses bfloat16 for efficient memory layout and accelerate's device mapping.
        """
        if self.is_loaded:
            return
            
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        target_device = device or self.config.DEFAULT_DEVICE
        
        # Determine actual device type if auto
        if target_device == "auto":
            if torch.cuda.is_available():
                device_map = "auto"
            elif torch.backends.mps.is_available():
                device_map = "mps"
            else:
                device_map = "cpu"
        else:
            device_map = target_device
            
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.MODEL_ID, 
            trust_remote_code=True
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.MODEL_ID,
            torch_dtype=torch.bfloat16,
            device_map=device_map,
            trust_remote_code=True
        )
        
        self.model.eval()

    def unload(self):
        """
        Frees model memory entirely.
        """
        if self.model is not None:
            del self.model
            self.model = None
            
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
            
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """
        Generates text using the loaded model.
        Returns a clean dictionary containing the response and metadata.
        """
        if not self.is_loaded:
            return {
                "status": "error",
                "error": "Model not loaded. Call load() before generate().",
                "response": None,
                "metadata": {}
            }
            
        if not prompt or not str(prompt).strip():
            return {
                "status": "error",
                "error": "Empty prompt provided.",
                "response": None,
                "metadata": {}
            }

        max_new_tokens = kwargs.get("max_new_tokens", self.config.DEFAULT_MAX_NEW_TOKENS)
        temperature = kwargs.get("temperature", self.config.DEFAULT_TEMPERATURE)
        do_sample = kwargs.get("do_sample", self.config.DEFAULT_DO_SAMPLE)
        
        # Apply standard chat template if applicable
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        
        try:
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **model_inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=do_sample,
                    pad_token_id=self.tokenizer.eos_token_id
                )
                
            # Slice off the input prompt tokens
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            
            response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            return {
                "status": "ok",
                "response": response.strip(),
                "metadata": {
                    "max_new_tokens": max_new_tokens,
                    "temperature": temperature,
                    "do_sample": do_sample,
                    "input_tokens": len(model_inputs.input_ids[0]),
                    "output_tokens": len(generated_ids[0])
                }
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": f"Inference failed: {str(e)}",
                "response": None,
                "metadata": {}
            }
