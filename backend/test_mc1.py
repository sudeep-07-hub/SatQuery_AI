import asyncio
import sys
import os
import json
import io

from mc1.pipeline import run_mc1_pipeline

class MockUploadFile:
    def __init__(self, filename, content):
        self.filename = filename
        self.content = content
        
    async def read(self):
        return self.content
        
    async def seek(self, pos):
        pass

async def test_mc1(filepaths, query):
    files = []
    for fp in filepaths:
        with open(fp, "rb") as f:
            content = f.read()
        files.append(MockUploadFile(os.path.basename(fp), content))
        
    profile = await run_mc1_pipeline(files, query)
    print(json.dumps(profile, indent=2))

if __name__ == "__main__":
    filepaths = sys.argv[1:-1]
    query = sys.argv[-1]
    asyncio.run(test_mc1(filepaths, query))
