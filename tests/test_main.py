import logging
from fastapi.testclient import TestClient

import app.main as main_module
from app.exceptions import (
    ModelNotLoadedError,
    PredictionError,
)
from app.main import app

from uuid import UUID


def test_predict_model_not_loaded(
    monkeypatch,
):
    def mock_predict_student(**kwargs):
        raise ModelNotLoadedError(
            "Model is not loaded"
        )

    monkeypatch.setattr(
        main_module,
        "predict_student",
        mock_predict_student,
    )

    with TestClient(app) as client:
        response = client.post(
            "/predict",
            json={
                "study_hours": 6,
                "absences": 1,
                "previous_score": 7.5,
            },
        )

    assert response.status_code == 503

    assert response.json() == {
        "detail": "Model is not loaded"
    }


def test_predict_prediction_error(
    monkeypatch,
):
    def mock_predict_student(**kwargs):
        raise PredictionError(
            "Model prediction failed"
        )

    monkeypatch.setattr(
        main_module,
        "predict_student",
        mock_predict_student,
    )

    with TestClient(app) as client:
        response = client.post(
            "/predict",
            json={
                "study_hours": 6,
                "absences": 1,
                "previous_score": 7.5,
            },
        )

    assert response.status_code == 500

    assert response.json() == {
        "detail": "Prediction failed"
    }


def test_health():
    with TestClient(app) as client:
        response = client.get(
            "/health"
        )

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"
    assert data["model_loaded"] is True

    assert (
        data["model_version"]
        == "1.0.0"
    )

    assert (
        data["environment"]
        == "development"
    )


def test_predict_valid_student():
    with TestClient(app) as client:
        response = client.post(
            "/predict",
            json={
                "study_hours": 6,
                "absences": 1,
                "previous_score": 7.5,
            },
        )

    assert response.status_code == 200

    data = response.json()

    assert "prediction" in data
    assert "passed" in data
    assert "pass_probability" in data

    assert data["prediction"] in [
        0,
        1,
    ]

    assert isinstance(
        data["passed"],
        bool,
    )

    assert (
        0.0
        <= data["pass_probability"]
        <= 1.0
    )


def test_predict_negative_study_hours():
    with TestClient(app) as client:
        response = client.post(
            "/predict",
            json={
                "study_hours": -2,
                "absences": 1,
                "previous_score": 7,
            },
        )

    assert response.status_code == 422


def test_predict_invalid_previous_score():
    with TestClient(app) as client:
        response = client.post(
            "/predict",
            json={
                "study_hours": 5,
                "absences": 2,
                "previous_score": 15,
            },
        )

    assert response.status_code == 422


def test_predict_missing_field():
    with TestClient(app) as client:
        response = client.post(
            "/predict",
            json={
                "study_hours": 5,
                "absences": 2,
            },
        )

    assert response.status_code == 422


def test_model_info():
    with TestClient(app) as client:
        response = client.get(
            "/model-info"
        )

    assert response.status_code == 200

    data = response.json()

    assert (
        "training_run_id"
        in data
    )

    parsed_run_id = UUID(
        data[
            "training_run_id"
        ]
    )

    assert (
        str(parsed_run_id)
        == data[
            "training_run_id"
        ]
    )

    assert (
        data["model_version"]
        == "1.0.0"
    )

    assert data["feature_names"] == [
        "study_hours",
        "absences",
        "previous_score",
    ]

    assert (
        "model_type"
        in data
    )

    assert (
        "test_accuracy"
        in data
    )

    assert (
        "trained_at"
        in data
    )

    assert (
        "environment"
        not in data
    )

    assert (
        "dataset_sha256"
        not in data
    )

    assert (
        "dataset_size"
        not in data
    )

    assert (
        "train_size"
        not in data
    )

    assert (
        "test_size"
        not in data
    )

def test_predict_logs_success(
    caplog,
):
    with caplog.at_level(
        logging.INFO
    ):
        with TestClient(app) as client:
            response = client.post(
                "/predict",
                json={
                    "study_hours": 6,
                    "absences": 1,
                    "previous_score": 7.5,
                },
            )

    assert response.status_code == 200

    assert (
        "Prediction completed"
        in caplog.text
    )

    assert (
        "model_version=1.0.0"
        in caplog.text
    )

    assert (
        "training_run_id="
        in caplog.text
    )
