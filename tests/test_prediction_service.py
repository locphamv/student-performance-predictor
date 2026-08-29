import numpy as np
import pytest

from app.exceptions import (
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
