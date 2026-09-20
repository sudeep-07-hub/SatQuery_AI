import requests
from typing import Dict, Any, Optional
import json

class OllamaQwen3Inference:
    """
    Adapter for Ollama Qwen3 4B model to match the Qwen3Inference interface.
    """
    def __init__(self, model_name: str = "qwen3:4b", endpoint: str = "http://localhost:11434"):
        self.model_name = model_name
        self.endpoint = endpoint
        self._is_loaded = False
        
    @property
    def is_loaded(self) -> bool:
        return self._is_loaded
        
    def load(self, device: Optional[str] = None):
        """
        Ollama loads models automatically on the first request, but we can send an empty request to preload it.
        """
        if self._is_loaded:
            return
            
        try:
            tags = requests.get(f"{self.endpoint}/api/tags", timeout=5)
            tags.raise_for_status()
            available = {m.get("name") for m in tags.json().get("models", [])}
            if self.model_name not in available:
                raise RuntimeError(f"model '{self.model_name}' is not pulled (available: {sorted(available)})")
            # A dummy request to load the model into memory
            response = requests.post(
                f"{self.endpoint}/api/generate",
                json={"model": self.model_name, "prompt": "", "stream": False, "keep_alive": "15m"},
                timeout=120
            )
            response.raise_for_status()
            self._is_loaded = True
        except Exception as e:
            raise RuntimeError(f"Failed to load Ollama model {self.model_name}: {e}")
            
    def unload(self):
        """
        Instructs Ollama to unload the model from memory.
        """
        if not self._is_loaded:
            return
            
        try:
            requests.post(
                f"{self.endpoint}/api/generate",
                json={"model": self.model_name, "keep_alive": 0},
                timeout=10
            )
            self._is_loaded = False
        except Exception:
            pass # Best effort cleanup

    def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """
        Matches the generate interface expected by the controller.
        """
        if not self._is_loaded:
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
            
        # The default is 0.0, which is what this method always sent: every existing caller relies
        # on deterministic decoding and passes no temperature, so their behaviour is unchanged.
        # It is now honoured rather than read and discarded — greedy decoding sends this model
        # into repetition loops on translation prompts (see qwen/translation.py).
        temperature = kwargs.get("temperature", 0.0)
        repeat_penalty = kwargs.get("repeat_penalty")
        max_new_tokens = kwargs.get("max_new_tokens", 512)
        
        try:
            res = requests.post(
                f"{self.endpoint}/api/chat",
                json={
                    "model": self.model_name,
                    "messages": [
                        {"role": "system", "content": "You are a specialized analytical engine. Respond directly and accurately without preamble."},
                        {"role": "user", "content": prompt}
                    ],
                    "format": "json",
                    "stream": False,
                    # Structured JSON tasks do not need Qwen3's reasoning trace; skipping it keeps latency demo-friendly
                    "think": False,
                    "keep_alive": "15m",
                    "options": {
                        "temperature": temperature,
                        "num_predict": max(2048, max_new_tokens),
                        **({"repeat_penalty": repeat_penalty} if repeat_penalty else {}),
                    }
                },
                timeout=600
            )
            res.raise_for_status()
            data = res.json()

            # The chat endpoint returns the message in data["message"]["content"]
            response_text = data.get("message", {}).get("content", "").strip()
            
            return {
                "status": "ok",
                "response": response_text,
                "metadata": {
                    "total_duration": data.get("total_duration"),
                    "eval_count": data.get("eval_count"),
                    "backend": "ollama"
                }
            }
        except Exception as e:
            return {
                "status": "error",
                "error": f"Ollama Inference failed: {str(e)}",
                "response": None,
                "metadata": {}
            }
