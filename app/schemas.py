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


class ModelInfoResponse(BaseModel):
    model_version: str
    model_type: str
    feature_names: list[str]
    test_accuracy: float
    trained_at: str
