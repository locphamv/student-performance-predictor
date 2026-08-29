import logging
from pathlib import Path
from time import perf_counter

import joblib

from app.config import settings
from app.exceptions import (
    ModelArtifactError,
    ModelNotLoadedError,
    PredictionError,
)
from app.features import (
    FEATURE_NAMES,
    build_feature_array,
)

logger = logging.getLogger(__name__)

REQUIRED_ARTIFACT_KEYS = {
    "pipeline",
    "metadata",
}

REQUIRED_METADATA_KEYS = {
    "model_version",
    "feature_names",
    "model_type",
}

project_directory = Path(__file__).resolve().parents[2]

model_path = (
    project_directory
    / "models"
    / settings.model_filename
)

pipeline = None
model_metadata = None


def validate_artifact(
    artifact,
) -> None:
    if not isinstance(
        artifact,
        dict,
    ):
        raise ModelArtifactError(
            "Model artifact must be a dictionary"
        )

    missing_artifact_keys = (
        REQUIRED_ARTIFACT_KEYS
        - artifact.keys()
    )

    if missing_artifact_keys:
        missing_keys = sorted(
            missing_artifact_keys
        )

        raise ModelArtifactError(
            f"Model artifact is missing required keys: "
            f"{missing_keys}"
        )

    metadata = artifact["metadata"]

    if not isinstance(
        metadata,
        dict,
    ):
        raise ModelArtifactError(
            "Model metadata must be a dictionary"
        )

    missing_metadata_keys = (
        REQUIRED_METADATA_KEYS
        - metadata.keys()
    )

    if missing_metadata_keys:
        missing_keys = sorted(
            missing_metadata_keys
        )

        raise ModelArtifactError(
            f"Model metadata is missing required keys: "
            f"{missing_keys}"
        )


def validate_pipeline(
    loaded_pipeline,
) -> None:
    if not hasattr(
        loaded_pipeline,
        "predict",
    ):
        raise ModelArtifactError(
            "Model pipeline does not support predict()"
        )

    if not hasattr(
        loaded_pipeline,
        "predict_proba",
    ):
        raise ModelArtifactError(
            "Model pipeline does not support predict_proba()"
        )


def load_model() -> None:
    global pipeline
    global model_metadata

    artifact = joblib.load(
        model_path
    )

    validate_artifact(
        artifact
    )

    loaded_pipeline = artifact["pipeline"]
    loaded_metadata = artifact["metadata"]

    validate_pipeline(
        loaded_pipeline
    )

    # Invalid artifact should fail during startup.
    if (
        loaded_metadata["feature_names"]
        != FEATURE_NAMES
    ):
        raise RuntimeError(
            "Model feature contract does not match application"
        )

    # Class 1 is required to calculate pass probability.
    if 1 not in loaded_pipeline.classes_:
        raise RuntimeError(
            "Model does not contain class 1"
        )

    pipeline = loaded_pipeline
    model_metadata = loaded_metadata

    logger.info(
        "Model loaded version=%s type=%s",
        model_metadata["model_version"],
        model_metadata["model_type"],
    )


def unload_model() -> None:
    global pipeline
    global model_metadata

    logger.info(
        "Unloading model"
    )

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
        logger.warning(
            "Prediction requested while model is not loaded"
        )
        raise ModelNotLoadedError(
            "Model is not loaded"
        )

    features = build_feature_array(
        study_hours=study_hours,
        absences=absences,
        previous_score=previous_score,
    )

    start_time = perf_counter()

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
        logger.exception(
            "Model prediction failed"
        )
        raise PredictionError(
            "Model prediction failed"
        ) from exc

    latency_ms = (
        perf_counter()
        - start_time
    ) * 1000
    if model_metadata is None:
        raise ModelNotLoadedError(
            "Model metadata is not loaded"
        )
    logger.info(
        (
            "Prediction completed "
            "model_version=%s "
            "latency_ms=%.3f"
        ),

        model_metadata["model_version"],
        latency_ms,
    )

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
