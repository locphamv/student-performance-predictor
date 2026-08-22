from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI

from app.schemas import (
    PredictionRequest,
    PredictionResponse,
)

project_directory = (
    Path(__file__).resolve().parent.parent
)

model_path = (
    project_directory
    / "models"
    / "student-pass-pipeline.joblib"
)

ml_models = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    ml_models["student_pass"] = joblib.load(
        model_path
    )

    yield

    ml_models.clear()


app = FastAPI(
    title="Student Performance Predictor",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": (
            "student_pass" in ml_models
        ),
    }


@app.post(
    "/predict",
    response_model=PredictionResponse,
)
def predict_student(
    request: PredictionRequest,
):
    pipeline = ml_models["student_pass"]

    features = np.array([
        [
            request.study_hours,
            request.absences,
            request.previous_score,
        ]
    ])

    prediction = pipeline.predict(
        features
    )[0]

    probabilities = pipeline.predict_proba(
        features
    )

    pass_probability = probabilities[0, 1]

    return PredictionResponse(
        prediction=int(prediction),
        passed=bool(prediction),
        pass_probability=round(
            float(pass_probability),
            3,
        ),
    )
