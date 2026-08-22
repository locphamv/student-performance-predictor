from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    study_hours: float = Field(
        ge=0,
        le=24,
    )
    absences: int = Field(
        ge=0,
    )
    previous_score: float = Field(
        ge=0,
        le=10,
    )


class PredictionResponse(BaseModel):
    prediction: int
    passed: bool
    pass_probability: float
