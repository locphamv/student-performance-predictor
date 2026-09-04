import numpy as np
import pytest

from app.exceptions import (
    ModelArtifactError,
    ModelNotLoadedError,
    PredictionError,
)

from app.services import prediction_service


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
        self.received_features = features
        return np.array([1])

    def predict_proba(
        self,
        features: np.ndarray,
    ):
        return np.array([
            [0.2, 0.8]
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
        },
    )

    prediction, probability = (
        prediction_service.predict_student(
            study_hours=6.0,
            absences=1,
            previous_score=7.5,
        )
    )

    assert prediction == 1
    assert probability == 0.8
    assert fake_pipeline.received_features is not None
    assert (
        fake_pipeline.received_features.tolist()
        == [[6.0, 1.0, 7.5]]
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


def test_validate_artifact_missing_metadata():
    artifact = {
        "pipeline": object(),
    }

    with pytest.raises(
        ModelArtifactError
    ):
        prediction_service.validate_artifact(
            artifact
        )


def test_validate_artifact_missing_model_version():
    artifact = {
        "pipeline": object(),
        "metadata": {
            "feature_names": [
                "study_hours",
                "absences",
                "previous_score",
            ],
            "model_type": (
                "LogisticRegression"
            ),
        },
    }

    with pytest.raises(
        ModelArtifactError
    ):
        prediction_service.validate_artifact(
            artifact
        )


def test_validate_pipeline_missing_predict():
    invalid_pipeline = object()

    with pytest.raises(
        ModelArtifactError
    ):
        prediction_service.validate_pipeline(
            invalid_pipeline
        )


class PredictOnlyPipeline:
    def predict(
        self,
        features: np.ndarray,
    ):
        return np.array([1])


def test_validate_pipeline_missing_predict_proba():
    invalid_pipeline = PredictOnlyPipeline()

    with pytest.raises(
        ModelArtifactError
    ):
        prediction_service.validate_pipeline(
            invalid_pipeline
        )


def test_validate_pipeline_success():
    pipeline = FakePipeline()

    prediction_service.validate_pipeline(
        pipeline
    )


def test_validate_artifact_rejects_invalid_environment():
    artifact = {
        "pipeline": object(),
        "metadata": {
            "model_version": "1.0.0",
            "feature_names": [
                "study_hours",
                "absences",
                "previous_score",
            ],
            "model_type": "LogisticRegression",
            "mean_cv_accuracy": 0.85,
            "std_cv_accuracy": 0.05,
            "test_accuracy": 0.80,
            "trained_at": (
                "2026-09-02T08:00:00+00:00"
            ),
            "dataset_size": 16,
            "train_size": 12,
            "test_size": 4,
            "environment": "not-a-dictionary",
        },
    }

    with pytest.raises(
        ModelArtifactError
    ):
        prediction_service.validate_artifact(
            artifact
        )


def test_validate_artifact_rejects_missing_environment_keys():
    artifact = {
        "pipeline": object(),
        "metadata": {
            "model_version": "1.0.0",
            "feature_names": [
                "study_hours",
                "absences",
                "previous_score",
            ],
            "model_type": "LogisticRegression",
            "mean_cv_accuracy": 0.85,
            "std_cv_accuracy": 0.05,
            "test_accuracy": 0.80,
            "trained_at": (
                "2026-09-02T08:00:00+00:00"
            ),
            "dataset_size": 16,
            "train_size": 12,
            "test_size": 4,
            "environment": {
                "python": "test",
                "scikit_learn": "test",
                "numpy": "test",
                "pandas": "test",
            },
        },
    }

    with pytest.raises(
        ModelArtifactError
    ):
        prediction_service.validate_artifact(
            artifact
        )

def test_validate_artifact_success():
    artifact = {
        "pipeline": object(),
        "metadata": {
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
            "test_accuracy": 0.80,
            "trained_at": (
                "2026-09-02T08:00:00+00:00"
            ),
            "dataset_size": 16,
            "train_size": 12,
            "test_size": 4,
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
        },
    }

    prediction_service.validate_artifact(
        artifact
    )


def test_get_public_model_info(
    monkeypatch,
):
    metadata = {
        "model_version": "1.0.0",
        "model_type": "KNN",
        "feature_names": [
            "study_hours",
            "absences",
            "previous_score",
        ],
        "mean_cv_accuracy": 0.80,
        "std_cv_accuracy": 0.10,
        "test_accuracy": 0.75,
        "trained_at": (
            "2026-09-04T08:00:00+00:00"
        ),
        "dataset_size": 16,
        "train_size": 12,
        "test_size": 4,
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
    }

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
