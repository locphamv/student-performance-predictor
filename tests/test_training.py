import joblib
import pandas as pd
import pytest

from app.training import (
    MIN_CV_ACCURACY,
    create_model_artifact,
    create_pipeline,
    evaluate_model,
    load_training_data,
    save_model_artifact,
    train_model,
    validate_model_performance,
)


def test_create_pipeline():
    pipeline = create_pipeline()

    assert "scaler" in pipeline.named_steps
    assert "model" in pipeline.named_steps


def test_load_training_data(
    tmp_path,
):
    data_path = (
        tmp_path
        / "training.csv"
    )

    data_path.write_text(
        (
            "study_hours,absences,"
            "previous_score,passed\n"
            "6.0,1,7.5,1\n"
        ),
        encoding="utf-8",
    )

    X, y = load_training_data(
        data_path
    )

    assert list(X.columns) == [
        "study_hours",
        "absences",
        "previous_score",
    ]

    assert y.tolist() == [1]


def test_load_training_data_missing_column(
    tmp_path,
):
    data_path = (
        tmp_path
        / "training.csv"
    )

    data_path.write_text(
        (
            "study_hours,absences,"
            "passed\n"
            "6.0,1,1\n"
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="previous_score",
    ):
        load_training_data(
            data_path
        )


def test_load_training_data_empty(
    tmp_path,
):
    data_path = (
        tmp_path
        / "training.csv"
    )

    data_path.write_text(
        (
            "study_hours,absences,"
            "previous_score,passed\n"
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="Training data is empty",
    ):
        load_training_data(
            data_path
        )


def test_load_training_data_missing_values(
    tmp_path,
):
    data_path = (
        tmp_path
        / "training.csv"
    )

    data_path.write_text(
        (
            "study_hours,absences,"
            "previous_score,passed\n"
            "6.0,1,,1\n"
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="Training data contains missing values",
    ):
        load_training_data(
            data_path
        )


def test_train_model():
    X = pd.DataFrame(
        {
            "study_hours": [
                2.0,
                3.0,
                7.0,
                8.0,
            ],
            "absences": [
                5,
                4,
                1,
                0,
            ],
            "previous_score": [
                4.0,
                5.0,
                8.0,
                9.0,
            ],
        }
    )

    y = pd.Series([
        0,
        0,
        1,
        1,
    ])

    pipeline = train_model(
        X,
        y,
    )

    predictions = pipeline.predict(
        X
    )

    assert len(predictions) == len(y)


def test_create_model_artifact():
    pipeline = create_pipeline()

    artifact = create_model_artifact(
        pipeline=pipeline,
        mean_cv_accuracy=0.85,
        std_cv_accuracy=0.05,
    )

    assert artifact["pipeline"] is pipeline

    metadata = artifact["metadata"]

    assert metadata["model_version"] == "1.0.0"

    assert metadata["feature_names"] == [
        "study_hours",
        "absences",
        "previous_score",
    ]

    assert (
        metadata["model_type"]
        == "LogisticRegression"
    )

    assert (
        metadata["mean_cv_accuracy"]
        == 0.85
    )

    assert (
        metadata["std_cv_accuracy"]
        == 0.05
    )


def test_save_model_artifact(
    tmp_path,
):
    pipeline = create_pipeline()

    artifact = create_model_artifact(
        pipeline=pipeline,
        mean_cv_accuracy=0.85,
        std_cv_accuracy=0.05,
    )

    model_path = (
        tmp_path
        / "models"
        / "test-model.joblib"
    )

    save_model_artifact(
        artifact,
        model_path,
    )

    assert model_path.exists()

    loaded_artifact = joblib.load(
        model_path
    )

    assert "pipeline" in loaded_artifact
    assert "metadata" in loaded_artifact

    assert (
        loaded_artifact["metadata"]["model_version"]
        == "1.0.0"
    )


def test_evaluate_model(
    tmp_path,
):
    data_path = (
        tmp_path
        / "training.csv"
    )

    data_path.write_text(
        (
            "study_hours,absences,"
            "previous_score,passed\n"
            "1,5,3,0\n"
            "2,4,4,0\n"
            "3,4,5,0\n"
            "4,3,5.5,0\n"
            "4.5,3,6,0\n"
            "5,2,6.5,1\n"
            "6,2,7,1\n"
            "7,1,8,1\n"
            "8,1,8.5,1\n"
            "9,0,9,1\n"
        ),
        encoding="utf-8",
    )

    X, y = load_training_data(
        data_path
    )

    mean_accuracy, std_accuracy = (
        evaluate_model(
            X,
            y,
        )
    )

    assert 0.0 <= mean_accuracy <= 1.0
    assert std_accuracy >= 0.0


def test_validate_model_performance_passes():
    validate_model_performance(
        MIN_CV_ACCURACY
    )


def test_validate_model_performance_fails():
    with pytest.raises(
        ValueError
    ):
        validate_model_performance(
            MIN_CV_ACCURACY - 0.1
        )


def test_training_flow(
    tmp_path,
):
    data_path = (
        tmp_path
        / "training.csv"
    )

    model_path = (
        tmp_path
        / "models"
        / "model.joblib"
    )

    data_path.write_text(
        (
            "study_hours,absences,"
            "previous_score,passed\n"
            "1,5,3,0\n"
            "2,4,4,0\n"
            "3,4,5,0\n"
            "4,3,5.5,0\n"
            "4.5,3,6,0\n"
            "5,2,6.5,1\n"
            "6,2,7,1\n"
            "7,1,8,1\n"
            "8,1,8.5,1\n"
            "9,0,9,1\n"
        ),
        encoding="utf-8",
    )

    X, y = load_training_data(
        data_path
    )

    mean_accuracy, std_accuracy = (
        evaluate_model(
            X,
            y,
        )
    )

    validate_model_performance(
        mean_accuracy
    )

    pipeline = train_model(
        X,
        y,
    )

    artifact = create_model_artifact(
        pipeline=pipeline,
        mean_cv_accuracy=mean_accuracy,
        std_cv_accuracy=std_accuracy,
    )

    save_model_artifact(
        artifact,
        model_path,
    )

    loaded_artifact = joblib.load(
        model_path
    )

    assert model_path.exists()
    assert "pipeline" in loaded_artifact
    assert "metadata" in loaded_artifact

    assert (
        loaded_artifact["metadata"]["mean_cv_accuracy"]
        == mean_accuracy
    )

    assert (
        loaded_artifact["metadata"]["std_cv_accuracy"]
        == std_accuracy
    )

    loaded_pipeline = loaded_artifact[
        "pipeline"
    ]

    predictions = loaded_pipeline.predict(
        X
    )

    assert len(predictions) == len(y)
