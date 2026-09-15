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
            # A dummy request to load the model into memory
            response = requests.post(
                f"{self.endpoint}/api/generate",
                json={"model": self.model_name, "prompt": "", "stream": False, "keep_alive": "5m"},
                timeout=30
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
            
        temperature = kwargs.get("temperature", 0.7)
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
                    "options": {
                        "temperature": 0.0,
                        "num_predict": max(2048, max_new_tokens)
                    }
                },
                timeout=600
            )
            res.raise_for_status()
            data = res.json()
            
            print(f"DEBUG OLLAMA JSON: {json.dumps(data)}")
            
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
