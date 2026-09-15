import asyncio
from job_manager import MockQwenEngine
engine = MockQwenEngine()
prompt = "task: cross_modal_fusion"
print(engine.generate(prompt))
