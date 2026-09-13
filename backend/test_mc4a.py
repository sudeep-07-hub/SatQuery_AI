import asyncio
import sys
import json
from PIL import Image

from mc4a_vqa.specialist import PaliGemmaVQASpecialist

def test_mc4a(filepath, query):
    specialist = PaliGemmaVQASpecialist()
    print("Is mock?", specialist.is_mock)
    
    mc1_profile = {
        "image_count": 1,
        "image_1": {
            "modality": "sar",
            "sensor": "Sentinel-1",
            "gsd_m": 10.0,
            "crs": "EPSG:32722"
        }
    }
    
    result = specialist.run(
        image_input=filepath,
        query=query,
        mc1_profile=mc1_profile,
        mc2_task_spec={}
    )
    
    print("Result:")
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    filepath = sys.argv[1]
    query = sys.argv[2]
    test_mc4a(filepath, query)
