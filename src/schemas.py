from pydantic import BaseModel
from typing import Optional, List


class Specification(BaseModel):

    component: str

    spec_type: str

    value: Optional[str] = None

    unit: Optional[str] = None

    alternate_value: Optional[str] = None

    alternate_unit: Optional[str] = None

    vehicle: Optional[str] = None

    configuration: Optional[str] = None

    section: Optional[str] = None

    page: Optional[int] = None

    evidence: Optional[str] = None


class ExtractionResponse(BaseModel):

    status: str

    results: List[Specification]