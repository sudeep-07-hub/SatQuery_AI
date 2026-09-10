from typing import Dict, Any, Optional
from pydantic import BaseModel, ValidationError

class MC4CInputPayload(BaseModel):
    optical_image: str
    sar_image: str
    query: str
    input_profile: Dict[str, Any]
    task_spec: Dict[str, Any]

class MC4CInputAdapter:
    @staticmethod
    def validate_inputs(payload: Dict[str, Any]) -> MC4CInputPayload:
        """
        Validates the incoming payload against the MC4C binding contract.
        Raises ValidationError if any required fields are missing or malformed.
        """
        return MC4CInputPayload(**payload)
