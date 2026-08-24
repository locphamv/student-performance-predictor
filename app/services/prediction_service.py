from pathlib import Path
from typing import Any

import joblib
import numpy as np


PROJECT_DIRECTORY = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    PROJECT_DIRECTORY
    / "models"
    / "student-pass-pipeline.joblib"
)

_pipeline: Any | None = None


def load_model() -> None:
    global _pipeline

    if _pipeline is not None:
        return

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )

    _pipeline = joblib.load(MODEL_PATH)


def unload_model() -> None:
    global _pipeline
    _pipeline = None


def is_model_loaded() -> bool:
    return _pipeline is not None


def predict_student(
    study_hours: float,
    absences: int,
    previous_score: float,
) -> tuple[int, float]:
    if _pipeline is None:
        raise RuntimeError(
            "Model is not loaded"
        )

    features = np.array(
        [[
            study_hours,
            absences,
            previous_score,
        ]],
        dtype=float,
    )

    prediction = int(
        _pipeline.predict(features)[0]
    )

    probabilities = _pipeline.predict_proba(
        features
    )[0]

    classes = list(_pipeline.classes_)

    if 1 not in classes:
        raise RuntimeError(
            "Model does not contain class 1"
        )

    pass_index = classes.index(1)
    pass_probability = float(
        probabilities[pass_index]
    )

    return prediction, pass_probability
