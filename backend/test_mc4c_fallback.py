from mc4c.engine import MC4CEngine
import pytest
import os

def test_fallback():
    print("Testing Fallback prevention...")
    engine = MC4CEngine()
    
    payload = {
        "optical_image": "", # Missing optical image
        "sar_image": "mc4c/mock/sar.tif",
        "query": "Find the deforested areas",
        "input_profile": {"sensor": "S2", "resolution": 10},
        "task_spec": {"task": "change_detection"}
    }
    
    try:
        engine.process(payload)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "forbidden" in str(e).lower()
        print("Fallback prevention PASSED.")

if __name__ == "__main__":
    test_fallback()
