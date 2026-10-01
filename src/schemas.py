from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


class Specification(BaseModel):

    component: str = Field(min_length=1)

    spec_type: str = Field(min_length=1)

    value: Optional[str] = None

    unit: Optional[str] = None

    part_number: Optional[str] = None

    alternate_value: Optional[str] = None

    alternate_unit: Optional[str] = None

    vehicle: Optional[str] = None

    configuration: Optional[str] = None

    section: Optional[str] = None

    page: Optional[int] = None

    evidence: str = Field(min_length=1)

    @field_validator("spec_type", mode="before")
    @classmethod
    def normalize_spec_type(cls, value):

        if not isinstance(value, str):
            return value

        normalized = (
            value.strip()
            .replace("_", " ")
            .replace("-", " ")
            .casefold()
        )

        mapping = {
            "part number": "Part Number",
            "torque": "Torque",
            "fill capacity": "Fill Capacity"
        }

        return mapping.get(
            normalized,
            value.strip()
        )
    @field_validator(
        "value",
        "alternate_value",
        mode="before"
    )
    @classmethod
    def normalize_measurement_values(cls, value):

        if value is None:
            return None

        return str(value).strip()

class ExtractionResponse(BaseModel):

    status: Literal["found", "not_found"]

    results: list[Specification]

    @model_validator(mode="after")
    def validate_status_matches_results(self):
        if self.status == "found" and not self.results:
            raise ValueError("A found response must include at least one result.")

        if self.status == "not_found" and self.results:
            raise ValueError("A not_found response must have an empty results list.")

        return self