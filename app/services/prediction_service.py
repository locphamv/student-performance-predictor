import logging
from pathlib import Path
from time import perf_counter
from uuid import UUID

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


logger = logging.getLogger(
    __name__
)


REQUIRED_ARTIFACT_KEYS = {
    "artifact_version",
    "pipeline",
    "metadata",
}

SUPPORTED_ARTIFACT_VERSIONS = {
    5,
}

REQUIRED_METADATA_KEYS = {
    "training_run_id",
    "model_version",
    "feature_names",
    "model_type",
    "mean_cv_accuracy",
    "std_cv_accuracy",
    "test_metrics",
    "trained_at",
    "dataset_size",
    "train_size",
    "test_size",
    "dataset_sha256",
    "environment",
    "source",
    "training_config",
    "class_distribution",
    "train_class_distribution",
    "test_class_distribution",
}

REQUIRED_ENVIRONMENT_KEYS = {
    "python",
    "scikit_learn",
    "numpy",
    "pandas",
    "joblib",
}

REQUIRED_SOURCE_KEYS = {
    "commit",
    "dirty",
}

REQUIRED_TRAINING_CONFIG_KEYS = {
    "test_size",
    "random_state",
    "cv_folds",
    "min_cv_accuracy",
}

REQUIRED_TEST_METRIC_KEYS = {
    "accuracy",
    "precision",
    "recall",
    "f1",
    "confusion_matrix",
}

project_directory = (
    Path(__file__)
    .resolve()
    .parents[2]
)

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
            "Model artifact is missing "
            "required keys: "
            f"{missing_keys}"
        )

    validate_artifact_version(
        artifact[
            "artifact_version"
        ]
    )

    metadata = artifact[
        "metadata"
    ]

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
            "Model metadata is missing "
            "required keys: "
            f"{missing_keys}"
        )

    validate_training_run_id(
        metadata[
            "training_run_id"
        ]
    )

    validate_git_provenance(
        metadata[
            "source"
        ]
    )

    validate_training_config(
        metadata[
            "training_config"
        ]
    )

    validate_class_distribution(
        metadata[
            "class_distribution"
        ]
    )

    validate_class_distribution(
        metadata[
            "train_class_distribution"
        ]
    )

    validate_class_distribution(
        metadata[
            "test_class_distribution"
        ]
    )

    validate_distribution_totals(
        metadata
    )

    validate_test_metrics(
        metadata[
            "test_metrics"
        ]
    )

    environment = metadata[
        "environment"
    ]

    if not isinstance(
        environment,
        dict,
    ):
        raise ModelArtifactError(
            "Model environment metadata "
            "must be a dictionary"
        )

    missing_environment_keys = (
        REQUIRED_ENVIRONMENT_KEYS
        - environment.keys()
    )

    if missing_environment_keys:
        missing_keys = sorted(
            missing_environment_keys
        )

        raise ModelArtifactError(
            "Model environment metadata "
            "is missing required keys: "
            f"{missing_keys}"
        )

    validate_sha256(
        metadata[
            "dataset_sha256"
        ]
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

    loaded_pipeline = artifact[
        "pipeline"
    ]

    loaded_metadata = artifact[
        "metadata"
    ]

    validate_pipeline(
        loaded_pipeline
    )

    if (
        loaded_metadata[
            "feature_names"
        ]
        != FEATURE_NAMES
    ):
        raise RuntimeError(
            "Model feature contract "
            "does not match application"
        )

    if (
        1
        not in loaded_pipeline.classes_
    ):
        raise RuntimeError(
            "Model does not contain class 1"
        )

    pipeline = loaded_pipeline
    model_metadata = (
        loaded_metadata
    )

    logger.info(
        (
            "Model loaded "
            "version=%s "
            "run_id=%s "
            "git_commit=%s "
            "git_dirty=%s "
            "type=%s"
        ),
        model_metadata[
            "model_version"
        ],
        model_metadata[
            "training_run_id"
        ],
        model_metadata[
            "source"
        ][
            "commit"
        ],
        model_metadata[
            "source"
        ][
            "dirty"
        ],
        model_metadata[
            "model_type"
        ],
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
    return (
        pipeline
        is not None
    )


def predict_student(
    study_hours: float,
    absences: int,
    previous_score: float,
) -> tuple[
    int,
    float,
]:
    if pipeline is None:
        logger.warning(
            "Prediction requested "
            "while model is not loaded"
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
        prediction = (
            pipeline.predict(
                features
            )[0]
        )

        probabilities = (
            pipeline.predict_proba(
                features
            )
        )

        classes = list(
            pipeline.classes_
        )

        pass_index = (
            classes.index(1)
        )

        pass_probability = float(
            probabilities[
                0,
                pass_index,
            ]
        )

    except Exception as exc:
        logger.exception(
            "Model prediction failed"
        )

        raise PredictionError(
            "Model prediction failed"
        ) from exc

    latency_ms = (
        (
            perf_counter()
            - start_time
        )
        * 1000
    )

    if model_metadata is None:
        raise ModelNotLoadedError(
            "Model metadata is not loaded"
        )

    logger.info(
        (
            "Prediction completed "
            "model_version=%s "
            "training_run_id=%s "
            "latency_ms=%.3f"
        ),
        model_metadata[
            "model_version"
        ],
        model_metadata[
            "training_run_id"
        ],
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


def validate_sha256(
    value: str,
) -> None:
    if not isinstance(
        value,
        str,
    ):
        raise ModelArtifactError(
            "Dataset SHA-256 must be a string"
        )

    if len(value) != 64:
        raise ModelArtifactError(
            "Dataset SHA-256 must contain "
            "64 hexadecimal characters"
        )

    try:
        int(
            value,
            16,
        )
    except ValueError as exc:
        raise ModelArtifactError(
            "Dataset SHA-256 must be hexadecimal"
        ) from exc


def get_public_model_info() -> dict:
    metadata = (
        get_model_metadata()
    )

    return {
        "model_version": (
            metadata[
                "model_version"
            ]
        ),
        "training_run_id": (
            metadata[
                "training_run_id"
            ]
        ),
        "model_type": (
            metadata[
                "model_type"
            ]
        ),
        "feature_names": (
            metadata[
                "feature_names"
            ]
        ),
        "test_accuracy": (
            metadata[
                "test_metrics"
            ]["accuracy"]
        ),
        "trained_at": (
            metadata[
                "trained_at"
            ]
        ),
    }


def validate_artifact_version(
    artifact_version,
) -> None:
    if not isinstance(
        artifact_version,
        int,
    ):
        raise ModelArtifactError(
            "Artifact version must be an integer"
        )

    if (
        artifact_version
        not in SUPPORTED_ARTIFACT_VERSIONS
    ):
        raise ModelArtifactError(
            "Unsupported artifact version: "
            f"{artifact_version}"
        )


def validate_training_run_id(
    training_run_id,
) -> None:
    if not isinstance(
        training_run_id,
        str,
    ):
        raise ModelArtifactError(
            "Training run ID must be a string"
        )

    try:
        UUID(
            training_run_id
        )
    except ValueError as exc:
        raise ModelArtifactError(
            "Training run ID must be a valid UUID"
        ) from exc


def validate_git_provenance(
    source,
) -> None:
    if not isinstance(
        source,
        dict,
    ):
        raise ModelArtifactError(
            "Model source metadata "
            "must be a dictionary"
        )

    missing_source_keys = (
        REQUIRED_SOURCE_KEYS
        - source.keys()
    )

    if missing_source_keys:
        missing_keys = sorted(
            missing_source_keys
        )

        raise ModelArtifactError(
            "Model source metadata "
            "is missing required keys: "
            f"{missing_keys}"
        )

    commit = source[
        "commit"
    ]

    if not isinstance(
        commit,
        str,
    ):
        raise ModelArtifactError(
            "Git commit must be a string"
        )

    if len(commit) not in {
        40,
        64,
    }:
        raise ModelArtifactError(
            "Git commit has an invalid length"
        )

    try:
        int(
            commit,
            16,
        )
    except ValueError as exc:
        raise ModelArtifactError(
            "Git commit must be hexadecimal"
        ) from exc

    if not isinstance(
        source[
            "dirty"
        ],
        bool,
    ):
        raise ModelArtifactError(
            "Git dirty flag must be boolean"
        )


def validate_training_config(
    config,
) -> None:
    if not isinstance(
        config,
        dict,
    ):
        raise ModelArtifactError(
            "Training configuration "
            "must be a dictionary"
        )

    missing_keys = (
        REQUIRED_TRAINING_CONFIG_KEYS
        - config.keys()
    )

    if missing_keys:
        raise ModelArtifactError(
            "Training configuration is "
            "missing required keys: "
            f"{sorted(missing_keys)}"
        )


def validate_class_distribution(
    distribution,
) -> None:
    if not isinstance(
        distribution,
        dict,
    ):
        raise ModelArtifactError(
            "Class distribution "
            "must be a dictionary"
        )

    expected_keys = {
        "0",
        "1",
    }

    if set(
        distribution.keys()
    ) != expected_keys:
        raise ModelArtifactError(
            "Class distribution must contain "
            "classes '0' and '1' "
        )

    for count in distribution.values():
        if not isinstance(
            count,
            int,
        ):
            raise ModelArtifactError(
                "Class distribution counts "
                "must be integers"
            )

        if count < 0:
            raise ModelArtifactError(
                "Class distribution counts "
                "cannot be negative"
            )


def validate_distribution_totals(
    metadata: dict,
) -> None:
    full_total = sum(
        metadata["class_distribution"].values()
    )

    train_total = sum(
        metadata["train_class_distribution"].values()
    )

    test_total = sum(
        metadata["test_class_distribution"].values()
    )

    if (
        full_total != metadata["dataset_size"]
    ):
        raise ModelArtifactError(
            "Class distribution total does not "
            "math dataset size"
        )

    if (
        train_total != metadata["train_size"]
    ):
        raise ModelArtifactError(
            "Training class distribution total "
            "does not match train size"
        )

    if (
        test_total != metadata["test_size"]
    ):
        raise ModelArtifactError(
            "Test class distribution total "
            "does not match test size"
        )


def validate_test_metrics(
    metrics,
) -> None:
    if not isinstance(
        metrics,
        dict,
    ):
        raise ModelArtifactError(
            "Test metrics must be a dictionary"
        )

    missing_keys = (
        REQUIRED_TEST_METRIC_KEYS
        - metrics.keys()
    )

    if missing_keys:
        raise ModelArtifactError(
            "Test metrics are missing "
            "required keys: "
            f"{sorted(missing_keys)}"
        )

    for name in (
        "accuracy",
        "precision",
        "recall",
        "f1",
    ):
        value = metrics[name]

        if not isinstance(
            value,
            (int, float),
        ):
            raise ModelArtifactError(
                f"{name} must be numeric"
            )

        if not (
            0.0
            <= value
            <= 1.0
        ):
            raise ModelArtifactError(
                f"{name} must be between "
                "0 and 1"
            )

    matrix = metrics[
        "confusion_matrix"
    ]

    if (
        not isinstance(matrix, list)
        or len(matrix) != 2
        or any(
            not isinstance(row, list)
            or len(row) != 2
            for row in matrix
        )
    ):
        raise ModelArtifactError(
            "Confusion matrix must be 2x2"
        )
