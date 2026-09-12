import numpy as np
import pytest

from app.exceptions import (
    ModelArtifactError,
    ModelNotLoadedError,
    PredictionError,
)
from app.services import (
    prediction_service,
)


VALID_RUN_ID = (
    "123e4567-e89b-12d3-a456-426614174000"
)


@pytest.fixture
def valid_artifact():
    return {
        "artifact_version": 5,
        "pipeline": object(),
        "metadata": (
            create_valid_metadata()
        ),
    }


def create_valid_metadata():
    return {
        "training_run_id": (
            VALID_RUN_ID
        ),
        "model_version": "1.0.0",
        "feature_names": [
            "study_hours",
            "absences",
            "previous_score",
        ],
        "model_type": (
            "LogisticRegression"
        ),
        "mean_cv_accuracy": 0.85,
        "std_cv_accuracy": 0.05,
        "test_metrics": {
            "accuracy": 0.80,
            "precision": 0.80,
            "recall": 0.75,
            "f1": 0.77,
            "confusion_matrix": [
                [2, 0],
                [1, 1],
            ],
        },
        "trained_at": (
            "2026-09-02T08:00:00+00:00"
        ),
        "dataset_size": 16,
        "train_size": 12,
        "test_size": 4,
        "class_distribution": {
            "0": 8,
            "1": 8,
        },
        "train_class_distribution": {
            "0": 6,
            "1": 6,
        },
        "test_class_distribution": {
            "0": 2,
            "1": 2,
        },
        "dataset_sha256": (
            "a" * 64
        ),
        "environment": {
            "python": "test",
            "scikit_learn": "test",
            "numpy": "test",
            "pandas": "test",
            "joblib": "test",
        },
        "source": {
            "commit": (
                "a" * 40
            ),
            "dirty": False,
        },
        "training_config": {
            "test_size": 0.25,
            "random_state": 42,
            "cv_folds": 5,
            "min_cv_accuracy": 0.75,
        },

    }


def test_predict_student_model_not_loaded(
    monkeypatch,
):
    monkeypatch.setattr(
        prediction_service,
        "pipeline",
        None,
    )

    with pytest.raises(
        ModelNotLoadedError
    ):
        prediction_service.predict_student(
            study_hours=6.0,
            absences=1,
            previous_score=7.5,
        )


class FakePipeline:
    def __init__(self):
        self.received_features = None

        self.classes_ = np.array([
            0,
            1,
        ])

    def predict(
        self,
        features: np.ndarray,
    ):
        self.received_features = (
            features
        )

        return np.array([
            1
        ])

    def predict_proba(
        self,
        features: np.ndarray,
    ):
        return np.array([
            [
                0.2,
                0.8,
            ]
        ])


def test_predict_student_success(
    monkeypatch,
):
    fake_pipeline = FakePipeline()

    monkeypatch.setattr(
        prediction_service,
        "pipeline",
        fake_pipeline,
    )

    monkeypatch.setattr(
        prediction_service,
        "model_metadata",
        {
            "model_version": "1.0.0",
            "training_run_id": (
                VALID_RUN_ID
            ),
        },
    )

    (
        prediction,
        probability,
    ) = prediction_service.predict_student(
        study_hours=6.0,
        absences=1,
        previous_score=7.5,
    )

    assert (
        prediction
        == 1
    )

    assert (
        probability
        == 0.8
    )

    assert (
        fake_pipeline
        .received_features
        is not None
    )

    assert (
        fake_pipeline
        .received_features
        .tolist()
        == [
            [
                6.0,
                1.0,
                7.5,
            ]
        ]
    )


class FailingPipeline:
    def predict(
        self,
        features: np.ndarray,
    ):
        raise ValueError(
            "Unexpected model error"
        )


def test_predict_student_failure(
    monkeypatch,
):
    monkeypatch.setattr(
        prediction_service,
        "pipeline",
        FailingPipeline(),
    )

    monkeypatch.setattr(
        prediction_service,
        "model_metadata",
        {
            "model_version": "1.0.0",
            "training_run_id": (
                VALID_RUN_ID
            ),
        },
    )

    with pytest.raises(
        PredictionError
    ):
        prediction_service.predict_student(
            study_hours=6.0,
            absences=1,
            previous_score=7.5,
        )


def test_prediction_error_preserves_cause(
    monkeypatch,
):
    monkeypatch.setattr(
        prediction_service,
        "pipeline",
        FailingPipeline(),
    )

    monkeypatch.setattr(
        prediction_service,
        "model_metadata",
        {
            "model_version": "1.0.0",
            "training_run_id": (
                VALID_RUN_ID
            ),
        },
    )

    with pytest.raises(
        PredictionError
    ) as exc_info:
        prediction_service.predict_student(
            study_hours=6.0,
            absences=1,
            previous_score=7.5,
        )

    assert isinstance(
        exc_info.value.__cause__,
        ValueError,
    )


def test_validate_artifact_rejects_non_dict():
    with pytest.raises(
        ModelArtifactError
    ):
        prediction_service.validate_artifact(
            "invalid artifact"
        )


def test_validate_artifact_missing_version():
    artifact = {
        "pipeline": object(),
        "metadata": {},
    }

    with pytest.raises(
        ModelArtifactError
    ):
        prediction_service.validate_artifact(
            artifact
        )


def test_validate_artifact_missing_metadata():
    artifact = {
        "artifact_version": 3,
        "pipeline": object(),
    }

    with pytest.raises(
        ModelArtifactError
    ):
        prediction_service.validate_artifact(
            artifact
        )


def test_validate_artifact_missing_model_version():
    metadata = (
        create_valid_metadata()
    )

    del metadata[
        "model_version"
    ]

    artifact = {
        "artifact_version": 3,
        "pipeline": object(),
        "metadata": metadata,
    }

    with pytest.raises(
        ModelArtifactError
    ):
        prediction_service.validate_artifact(
            artifact
        )


def test_validate_artifact_rejects_unsupported_version():
    artifact = {
        "artifact_version": 999,
        "pipeline": object(),
        "metadata": {},
    }

    with pytest.raises(
        ModelArtifactError
    ) as exc_info:
        prediction_service.validate_artifact(
            artifact
        )

    assert (
        "Unsupported artifact version"
        in str(
            exc_info.value
        )
    )


def test_validate_artifact_rejects_non_integer_version():
    artifact = {
        "artifact_version": "3",
        "pipeline": object(),
        "metadata": {},
    }

    with pytest.raises(
        ModelArtifactError
    ) as exc_info:
        prediction_service.validate_artifact(
            artifact
        )

    assert (
        "Artifact version must be an integer"
        in str(
            exc_info.value
        )
    )


def test_validate_pipeline_missing_predict():
    with pytest.raises(
        ModelArtifactError
    ):
        prediction_service.validate_pipeline(
            object()
        )


class PredictOnlyPipeline:
    def predict(
        self,
        features: np.ndarray,
    ):
        return np.array([
            1
        ])


def test_validate_pipeline_missing_predict_proba():
    with pytest.raises(
        ModelArtifactError
    ):
        prediction_service.validate_pipeline(
            PredictOnlyPipeline()
        )


def test_validate_pipeline_success():
    prediction_service.validate_pipeline(
        FakePipeline()
    )


def test_validate_artifact_rejects_invalid_environment():
    metadata = (
        create_valid_metadata()
    )

    metadata[
        "environment"
    ] = "not-a-dictionary"

    artifact = {
        "artifact_version": 3,
        "pipeline": object(),
        "metadata": metadata,
    }

    with pytest.raises(
        ModelArtifactError
    ):
        prediction_service.validate_artifact(
            artifact
        )


def test_validate_artifact_rejects_missing_environment_keys():
    metadata = (
        create_valid_metadata()
    )

    del metadata[
        "environment"
    ][
        "joblib"
    ]

    artifact = {
        "artifact_version": 3,
        "pipeline": object(),
        "metadata": metadata,
    }

    with pytest.raises(
        ModelArtifactError
    ):
        prediction_service.validate_artifact(
            artifact
        )


def test_validate_artifact_success():
    artifact = {
        "artifact_version": 5,
        "pipeline": object(),
        "metadata": (
            create_valid_metadata()
        ),
    }

    prediction_service.validate_artifact(
        artifact
    )


def test_validate_artifact_rejects_invalid_sha256():
    metadata = (
        create_valid_metadata()
    )

    metadata[
        "dataset_sha256"
    ] = "invalid"

    artifact = {
        "artifact_version": 3,
        "pipeline": object(),
        "metadata": metadata,
    }

    with pytest.raises(
        ModelArtifactError
    ):
        prediction_service.validate_artifact(
            artifact
        )


def test_validate_training_run_id_rejects_invalid_value():
    with pytest.raises(
        ModelArtifactError
    ):
        prediction_service.validate_training_run_id(
            "not-a-uuid"
        )


def test_validate_training_run_id_rejects_non_string():
    with pytest.raises(
        ModelArtifactError
    ):
        prediction_service.validate_training_run_id(
            123
        )


def test_validate_git_provenance_rejects_invalid_commit():
    source = {
        "commit": (
            "not-a-commit"
        ),
        "dirty": False,
    }

    with pytest.raises(
        ModelArtifactError
    ):
        prediction_service.validate_git_provenance(
            source
        )


def test_validate_git_provenance_rejects_invalid_dirty_flag():
    source = {
        "commit": (
            "a" * 40
        ),
        "dirty": "false",
    }

    with pytest.raises(
        ModelArtifactError
    ):
        prediction_service.validate_git_provenance(
            source
        )


def test_get_public_model_info(
    monkeypatch,
):
    metadata = (
        create_valid_metadata()
    )

    metadata[
        "model_type"
    ] = "KNN"

    metadata[
        "test_metrics"
    ]["accuracy"] = 0.75

    metadata[
        "trained_at"
    ] = (
        "2026-09-04T08:00:00+00:00"
    )

    monkeypatch.setattr(
        prediction_service,
        "model_metadata",
        metadata,
    )

    public_info = (
        prediction_service
        .get_public_model_info()
    )

    assert public_info == {
        "model_version": "1.0.0",
        "training_run_id": (
            VALID_RUN_ID
        ),
        "model_type": "KNN",
        "feature_names": [
            "study_hours",
            "absences",
            "previous_score",
        ],
        "test_accuracy": 0.75,
        "trained_at": (
            "2026-09-04T08:00:00+00:00"
        ),
    }


def test_validate_artifact_rejects_missing_training_config(
    valid_artifact,
):
    del valid_artifact["metadata"][
        "training_config"
    ]

    with pytest.raises(
        ModelArtifactError,
    ):
        prediction_service.validate_artifact(
            valid_artifact
        )


def test_validate_artifact_rejects_incomplete_training_config(
    valid_artifact,
):
    del valid_artifact["metadata"][
        "training_config"
    ]["cv_folds"]

    with pytest.raises(
        ModelArtifactError,
        match="cv_folds",
    ):
        prediction_service.validate_artifact(
            valid_artifact
        )


def test_validate_class_distribution_rejects_missing_class():
    distribution = {
        "0": 10,
    }

    with pytest.raises(
        ModelArtifactError
    ):
        prediction_service.validate_class_distribution(
            distribution
        )


def test_validate_class_distribution_rejects_negative_count():
    distribution = {
        "0": 10,
        "1": -1,
    }

    with pytest.raises(
        ModelArtifactError
    ):
        prediction_service.validate_class_distribution(
            distribution
        )


def test_validate_class_distribution_success():
    distribution = {
        "0": 8,
        "1": 8,
    }

    prediction_service.validate_class_distribution(
        distribution
    )


def test_validate_distribution_totals_success():
    metadata = create_valid_metadata()

    prediction_service.validate_distribution_totals(
        metadata
    )


def test_validate_distribution_totals_rejects_dataset_mismatch():
    metadata = create_valid_metadata()

    metadata["class_distribution"] = {
        "0": 7,
        "1": 8,
    }

    with pytest.raises(
        ModelArtifactError
    ):
        prediction_service.validate_distribution_totals(
            metadata
        )


def test_validate_distribution_totals_rejects_train_mismatch():
    metadata = create_valid_metadata()

    metadata[
        "train_class_distribution"
    ] = {
        "0": 5,
        "1": 6,
    }

    with pytest.raises(
        ModelArtifactError
    ):
        prediction_service.validate_distribution_totals(
            metadata
        )


def test_validate_distribution_totals_rejects_test_mismatch():
    metadata = create_valid_metadata()

    metadata[
        "test_class_distribution"
    ] = {
        "0": 1,
        "1": 2,
    }

    with pytest.raises(
        ModelArtifactError
    ):
        prediction_service.validate_distribution_totals(
            metadata
        )


def test_validate_test_metrics_success():
    metrics = {
        "accuracy": 0.80,
        "precision": 0.80,
        "recall": 0.75,
        "f1": 0.77,
        "confusion_matrix": [
            [2, 0],
            [1, 1],
        ],
    }

    prediction_service.validate_test_metrics(
        metrics
    )


def test_validate_test_metrics_rejects_missing_keys():
    metrics = {
        "accuracy": 0.80,
        "precision": 0.80,
        "recall": 0.75,
        "confusion_matrix": [
            [2, 0],
            [1, 1],
        ],
    }

    with pytest.raises(
        ModelArtifactError,
        match = "f1",
    ):
        prediction_service.validate_test_metrics(
            metrics
        )


def test_validate_test_metrics_rejects_invalid_range():
    metrics = {
        "accuracy": 1.5,
        "precision": 0.80,
        "recall": 0.75,
        "f1": 0.77,
        "confusion_matrix": [
            [2, 0],
            [1, 1],
        ],
    }

    with pytest.raises(
        ModelArtifactError,
        match="accuracy",
    ):
        prediction_service.validate_test_metrics(
            metrics
        )

def test_validate_test_metrics_rejects_invalid_matrix():
    metrics = {
        "accuracy": 0.80,
        "precision": 0.80,
        "recall": 0.75,
        "f1": 0.77,
        "confusion_matrix": [
            [2, 0, 1],
            [1, 1, 0],
        ],
    }

    with pytest.raises(
        ModelArtifactError,
        match="2x2",
    ):
        prediction_service.validate_test_metrics(
            metrics
        )
