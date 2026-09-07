class MockPaliGemmaVQAAdapter:
    """Mock Adapter for Single-Image Visual Question Answering (Track A)."""

    def __init__(self, model_id: str = "mock-paligemma-3b-pt-224"):
        self.model_id = model_id
        # Deterministic responses for testing
        self.mock_responses = {
            "what crop is visible in the first image": "wheat",
            "is there a building": "yes",
            "how many cars": "3",
            "what is the primary land cover": "forest",
        }

    def predict(self, image_input, query: str) -> dict:
        # We don't actually process the image, just check if it's openable/convertible
        if isinstance(image_input, str):
            from PIL import Image
            image = Image.open(image_input).convert("RGB")
        else:
            image = image_input.convert("RGB")

        q_lower = query.lower().strip().replace("?", "")
        
        answer = "unknown"
        for k, v in self.mock_responses.items():
            if k in q_lower:
                answer = v
                break
                
        # If no deterministic match, return a generic echo
        if answer == "unknown":
            answer = f"mock_answer_for: {query}"

        return {
            "text": answer,
            "evidence_type": "none",
            "spatial_evidence": None,
            "confidence": 0.95
        }
