from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

from app.config import settings
from app.exceptions import (
    ModelNotLoadedError,
    PredictionError,
)
from app.schemas import (
    PredictionRequest,
    PredictionResponse,
    ModelInfoResponse,
)
from app.services.prediction_service import (
    get_model_metadata,
    is_model_loaded,
    load_model,
    predict_student,
    unload_model,
    get_public_model_info,
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


@app.exception_handler(ModelNotLoadedError)
async def model_not_loaded_handler(
    request: Request,
    exc: ModelNotLoadedError,
):
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={
            "detail": str(exc)
        },
    )


@app.exception_handler(PredictionError)
async def prediction_error_handler(
    request: Request,
    exc: PredictionError,
):
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "detail": "Prediction failed"
        },
    )


@app.get("/health")
def health_check():
    metadata = get_model_metadata()

    return {
        "status": "healthy",
        "environment": settings.app_environment,
        "model_loaded": is_model_loaded(),
        "model_version": metadata[
            "model_version"
        ],
    }


@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["Prediction"],
)
def predict_student_endpoint(
    request: PredictionRequest,
) -> PredictionResponse:
    prediction, pass_probability = predict_student(
        study_hours=request.study_hours,
        absences=request.absences,
        previous_score=request.previous_score,
    )

    return PredictionResponse(
        prediction=prediction,
        passed=prediction == 1,
        pass_probability=round(
            pass_probability,
            3,
        ),
    )


@app.get(
    "/model-info",
    response_model=ModelInfoResponse,
)
def model_info():
    return get_public_model_info()
