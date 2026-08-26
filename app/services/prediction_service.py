from pathlib import Path
from typing import Any

import joblib
# import numpy as np
from app.features import build_feature_array, FEATURE_NAMES

PROJECT_DIRECTORY = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    PROJECT_DIRECTORY
    / "models"
    / "student-pass-pipeline.joblib"
)

pipeline = None
model_metadata = None

def load_model() -> None:
    global pipeline
    global model_metadata

    artifact = joblib.load(
        MODEL_PATH
    )

    pipeline = artifact["pipeline"]
    model_metadata = artifact["metadata"]

    if (
        model_metadata["feature_names"]
        != FEATURE_NAMES
    ):
        raise RuntimeError(
            "Model feature contract does not match application"
        )

def unload_model() -> None:
    global pipeline
    global model_metadata

    pipeline = None
    model_metadata = None


def is_model_loaded() -> bool:
    return pipeline is not None


def predict_student(
    study_hours: float,
    absences: int,
    previous_score: float,
) -> tuple[int, float]:
    if pipeline is None:
        raise RuntimeError(
            "Model is not loaded"
        )

    features = build_feature_array(
        study_hours=study_hours,
        absences=absences,
        previous_score=previous_score,
    )

    prediction = int(
        pipeline.predict(features)[0]
    )

    probabilities = pipeline.predict_proba(
        features
    )[0]

    classes = list(pipeline.classes_)

    if 1 not in classes:
        raise RuntimeError(
            "Model does not contain class 1"
        )

    pass_index = classes.index(1)
    pass_probability = float(
        probabilities[pass_index]
    )

    return prediction, pass_probability


def get_model_metadata():
    if model_metadata is None:
        raise RuntimeError(
            "Model metadata is not loaded"
        )

    return model_metadata
