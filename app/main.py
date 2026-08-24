from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, status

from app.schemas import (
    PredictionRequest,
    PredictionResponse,
)
from app.services.prediction_service import (
    is_model_loaded,
    load_model,
    predict_student,
    unload_model,
)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    load_model()

    try:
        yield
    finally:
        unload_model()


app = FastAPI(
    title="Student Performance Predictor",
    description="Predict whether a student will pass.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get(
    "/health",
    tags=["Health"],
)
def health_check():
    model_loaded = is_model_loaded()

    if not model_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Prediction model is unavailable",
        )

    return {
        "status": "healthy",
        "model_loaded": True,
    }


@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["Prediction"],
)
def predict_student_endpoint(
    request: PredictionRequest,
) -> PredictionResponse:
    if not is_model_loaded():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Prediction model is unavailable",
        )

    try:
        prediction, pass_probability = predict_student(
            study_hours=request.study_hours,
            absences=request.absences,
            previous_score=request.previous_score,
        )
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc

    return PredictionResponse(
        prediction=prediction,
        passed=prediction == 1,
        pass_probability=round(
            pass_probability,
            3,
        ),
    )
