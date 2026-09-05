from datetime import datetime
import joblib
import pandas as pd
import pytest

from app.training import (
    ARTIFACT_VERSION,
    MIN_CV_ACCURACY,
    create_model_artifact,
    create_candidate_models,
    evaluate_model,
    load_training_data,
    save_model_artifact,
    train_model,
    validate_model_performance,
    split_training_data,
    evaluate_final_model,
    ModelSelectionResult,
    select_best_model,
    TrainingResult,
    train_and_evaluate_best_model,
    calculate_file_sha256,
)


def test_select_best_model_rejects_empty_candidates():
    with pytest.raises(
        ValueError
    ):
        select_best_model(
            {},
            pd.DataFrame(),
            pd.Series(dtype=int),
        )


def test_create_candidate_models():
    models = create_candidate_models()

    assert set(models.keys()) == {
        "LogisticRegression",
        "DecisionTree",
        "KNN",
    }


def test_select_best_model(
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

    candidates = (
        create_candidate_models()
    )

    result = select_best_model(
        candidates,
        X,
        y,
    )

    assert isinstance(
        result,
        ModelSelectionResult,
    )

    assert (
        result.model_name
        in candidates
    )

    assert (
        0.0
        <= result.mean_cv_accuracy
        <= 1.0
    )

    assert (
        result.std_cv_accuracy
        >= 0.0
    )


def test_candidate_model_steps():
    models = create_candidate_models()

    logistic = models[
        "LogisticRegression"
    ]

    tree = models[
        "DecisionTree"
    ]

    knn = models[
        "KNN"
    ]

    assert "scaler" in logistic.named_steps
    assert "scaler" not in tree.named_steps
    assert "scaler" in knn.named_steps


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

    models = create_candidate_models()

    pipeline = models[
        "LogisticRegression"
    ]

    trained_pipeline = train_model(
        pipeline,
        X,
        y,
    )

    predictions = trained_pipeline.predict(
        X
    )

    assert len(predictions) == len(y)


def test_create_model_artifact():
    pipeline = (
        create_candidate_models()[
            "LogisticRegression"
        ]
    )

    result = TrainingResult(
        model_name="LogisticRegression",
        pipeline=pipeline,
        mean_cv_accuracy=0.85,
        std_cv_accuracy=0.05,
        test_accuracy=0.80,
    )

    dataset_sha256 = (
        "a" * 64
    )

    artifact = create_model_artifact(
        result=result,
        dataset_size=16,
        train_size=12,
        test_size=4,
        dataset_sha256=(
            dataset_sha256
        ),
    )

    assert (
    artifact["artifact_version"]
    == ARTIFACT_VERSION
)

    assert (
        artifact["pipeline"]
        is pipeline
    )

    metadata = artifact[
        "metadata"
    ]

    assert (
        metadata["model_version"]
        == "1.0.0"
    )

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

    assert (
        metadata["test_accuracy"]
        == 0.80
    )

    assert (
        metadata["dataset_size"]
        == 16
    )

    assert (
        metadata["train_size"]
        == 12
    )

    assert (
        metadata["test_size"]
        == 4
    )

    assert (
        metadata["dataset_sha256"]
        == dataset_sha256
    )

    environment = metadata[
        "environment"
    ]

    assert "python" in environment

    assert (
        "scikit_learn"
        in environment
    )

    assert "numpy" in environment
    assert "pandas" in environment
    assert "joblib" in environment

    trained_at = metadata[
        "trained_at"
    ]

    parsed_timestamp = (
        datetime.fromisoformat(
            trained_at
        )
    )

    assert (
        parsed_timestamp.tzinfo
        is not None
    )


def test_save_model_artifact(
    tmp_path,
):
    model_path = (
        tmp_path
        / "models"
        / "model.joblib"
    )

    pipeline = (
        create_candidate_models()[
            "LogisticRegression"
        ]
    )

    result = TrainingResult(
        model_name="LogisticRegression",
        pipeline=pipeline,
        mean_cv_accuracy=0.85,
        std_cv_accuracy=0.05,
        test_accuracy=0.80,
    )

    dataset_sha256 = (
        "a" * 64
    )

    artifact = create_model_artifact(
        result=result,
        dataset_size=16,
        train_size=12,
        test_size=4,
        dataset_sha256=(
            dataset_sha256
        ),
    )

    save_model_artifact(
        artifact,
        model_path,
    )

    loaded_artifact = joblib.load(
        model_path
    )

    assert model_path.exists()

    metadata = loaded_artifact[
        "metadata"
    ]

    assert (
        metadata["model_version"]
        == "1.0.0"
    )

    assert (
        metadata["model_type"]
        == "LogisticRegression"
    )

    assert (
        metadata["dataset_size"]
        == 16
    )

    assert (
        metadata["train_size"]
        == 12
    )

    assert (
        metadata["test_size"]
        == 4
    )

    assert (
        metadata["dataset_sha256"]
        == dataset_sha256
    )

    assert (
        metadata["test_accuracy"]
        == 0.80
    )

    assert (
        "environment"
        in metadata
    )

    assert (
        "trained_at"
        in metadata
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
            "1.0,6,3.5,0\n"
            "1.5,6,4.0,0\n"
            "2.0,5,4.2,0\n"
            "2.5,5,4.5,0\n"
            "3.0,4,5.0,0\n"
            "3.5,4,5.2,0\n"
            "4.0,3,5.5,0\n"
            "4.5,3,6.0,1\n"
            "5.0,3,5.8,0\n"
            "5.5,2,6.5,1\n"
            "6.0,2,7.0,1\n"
            "6.5,1,7.2,1\n"
            "7.0,1,7.8,1\n"
            "7.5,1,8.0,1\n"
            "8.0,0,8.5,1\n"
            "8.5,0,9.0,1\n"
        ),
        encoding="utf-8",
    )

    X, y = load_training_data(
        data_path
    )

    models = create_candidate_models()

    pipeline = models[
        "LogisticRegression"
    ]

    mean_accuracy, std_accuracy = (
        evaluate_model(
            pipeline,
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


def test_split_training_data(
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
            "5,2,6,1\n"
            "6,2,7,1\n"
            "7,1,8,1\n"
            "8,1,8.5,1\n"
        ),
        encoding="utf-8",
    )

    X, y = load_training_data(
        data_path
    )

    (
        X_train,
        X_test,
        y_train,
        y_test,
    ) = split_training_data(
        X,
        y,
    )

    assert (
        len(X_train)
        + len(X_test)
        == len(X)
    )

    assert (
        len(y_train)
        + len(y_test)
        == len(y)
    )


def test_evaluate_final_model(
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
            "5,2,6,1\n"
            "6,2,7,1\n"
            "7,1,8,1\n"
            "8,1,8.5,1\n"
        ),
        encoding="utf-8",
    )

    X, y = load_training_data(
        data_path
    )

    (
        X_train,
        X_test,
        y_train,
        y_test,
    ) = split_training_data(
        X,
        y,
    )

    models = create_candidate_models()

    pipeline = models[
        "LogisticRegression"
    ]

    fitted_pipeline = train_model(
        pipeline,
        X_train,
        y_train,
    )

    accuracy = evaluate_final_model(
        fitted_pipeline,
        X_test,
        y_test,
    )

    assert (
        0.0
        <= accuracy
        <= 1.0
    )


def test_calculate_file_sha256(
    tmp_path,
):
    file_path = (
        tmp_path
        / "data.csv"
    )

    file_path.write_text(
        "a,b\n1,2\n",
        encoding="utf-8",
    )

    first_hash = (
        calculate_file_sha256(
            file_path
        )
    )

    second_hash = (
        calculate_file_sha256(
            file_path
        )
    )

    assert (
        first_hash
        == second_hash
    )

    assert (
        len(first_hash)
        == 64
    )

def test_file_hash_changes_with_content(
    tmp_path,
):
    file_path = (
        tmp_path
        / "data.csv"
    )

    file_path.write_text(
        "a,b\n1,2\n",
        encoding="utf-8",
    )

    first_hash = (
        calculate_file_sha256(
            file_path
        )
    )

    file_path.write_text(
        "a,b\n1,3\n",
        encoding="utf-8",
    )

    second_hash = (
        calculate_file_sha256(
            file_path
        )
    )

    assert (
        first_hash
        != second_hash
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
            "1.0,6,3.5,0\n"
            "1.5,6,4.0,0\n"
            "2.0,5,4.2,0\n"
            "2.5,5,4.5,0\n"
            "3.0,4,5.0,0\n"
            "3.5,4,5.2,0\n"
            "4.0,3,5.5,0\n"
            "4.5,3,6.0,1\n"
            "5.0,3,5.8,0\n"
            "5.5,2,6.5,1\n"
            "6.0,2,7.0,1\n"
            "6.5,1,7.2,1\n"
            "7.0,1,7.8,1\n"
            "7.5,1,8.0,1\n"
            "8.0,0,8.5,1\n"
            "8.5,0,9.0,1\n"
        ),
        encoding="utf-8",
    )

    dataset_sha256 = (
        calculate_file_sha256(
            data_path
        )
    )

    X, y = load_training_data(
        data_path
    )

    (
        X_train,
        X_test,
        y_train,
        y_test,
    ) = split_training_data(
        X,
        y,
    )

    result = (
        train_and_evaluate_best_model(
            X_train,
            X_test,
            y_train,
            y_test,
        )
    )

    artifact = create_model_artifact(
        result=result,
        dataset_size=len(X),
        train_size=len(X_train),
        test_size=len(X_test),
        dataset_sha256=(
            dataset_sha256
        ),
    )

    save_model_artifact(
        artifact,
        model_path,
    )

    loaded_artifact = joblib.load(
        model_path
    )

    assert model_path.exists()

    metadata = loaded_artifact[
        "metadata"
    ]

    assert (
        metadata["mean_cv_accuracy"]
        == result.mean_cv_accuracy
    )

    assert (
        metadata["std_cv_accuracy"]
        == result.std_cv_accuracy
    )

    assert (
        metadata["model_type"]
        == result.model_name
    )

    assert (
        metadata["test_accuracy"]
        == result.test_accuracy
    )

    assert (
        metadata["dataset_size"]
        == len(X)
    )

    assert (
        metadata["train_size"]
        == len(X_train)
    )

    assert (
        metadata["test_size"]
        == len(X_test)
    )

    assert (
        metadata["dataset_sha256"]
        == dataset_sha256
    )

    assert (
        "trained_at"
        in metadata
    )

    assert (
        "environment"
        in metadata
    )

    loaded_pipeline = (
        loaded_artifact["pipeline"]
    )

    predictions = (
        loaded_pipeline.predict(
            X_test
        )
    )

    assert (
        len(predictions)
        == len(y_test)
    )
