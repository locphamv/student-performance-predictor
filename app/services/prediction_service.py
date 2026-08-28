from pathlib import Path

import joblib

from app.config import settings
from app.exceptions import (
    ModelNotLoadedError,
    PredictionError,
)
from app.features import (
    FEATURE_NAMES,
    build_feature_array,
)


project_directory = Path(__file__).resolve().parents[2]

model_path = (
    project_directory
    / "models"
    / settings.model_filename
)

pipeline = None
model_metadata = None


def load_model() -> None:
    global pipeline
    global model_metadata

    artifact = joblib.load(
        model_path
    )

    pipeline = artifact["pipeline"]
    model_metadata = artifact["metadata"]

    # Invalid artifact should fail during startup.
    if (
        model_metadata["feature_names"]
        != FEATURE_NAMES
    ):
        raise RuntimeError(
            "Model feature contract does not match application"
        )

    # Class 1 is required to calculate pass probability.
    if 1 not in pipeline.classes_:
        raise RuntimeError(
            "Model does not contain class 1"
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
        raise ModelNotLoadedError(
            "Model is not loaded"
        )

    features = build_feature_array(
        study_hours=study_hours,
        absences=absences,
        previous_score=previous_score,
    )

    try:
        prediction = pipeline.predict(
            features
        )[0]

        probabilities = pipeline.predict_proba(
            features
        )

        classes = list(
            pipeline.classes_
        )

        pass_index = classes.index(1)

        pass_probability = float(
            probabilities[0, pass_index]
        )

    except Exception as exc:
        raise PredictionError(
            "Model prediction failed"
        ) from exc

    return (
        int(prediction),
        pass_probability,
    )


def get_model_metadata():
    if model_metadata is None:
        raise ModelNotLoadedError(
            "Model metadata is not loaded"
        )

    return model_metadata
