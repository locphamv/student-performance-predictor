from datetime import datetime
import subprocess
from uuid import UUID

import joblib
import pandas as pd
import pytest

from app import training
from app.training import (
    TrainingConfig,
    ARTIFACT_VERSION,
    ModelSelectionResult,
    TrainingResult,
    calculate_file_sha256,
    create_candidate_models,
    create_model_artifact,
    create_training_run_id,
    evaluate_final_model,
    evaluate_model,
    load_training_data,
    save_model_artifact,
    select_best_model,
    split_training_data,
    train_and_evaluate_best_model,
    train_model,
    validate_model_performance,
    validate_training_config_values,
    validate_target_distribution,
    validate_cv_compatibility,
)


def test_select_best_model_rejects_empty_candidates():
    config = TrainingConfig()

    with pytest.raises(
        ValueError
    ):
        select_best_model(
            {},
            pd.DataFrame(),
            pd.Series(dtype=int),
            config,
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

    config = TrainingConfig()

    result = select_best_model(
        candidates,
        X,
        y,
        config,
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

    assert (
        "scaler"
        in logistic.named_steps
    )

    assert (
        "scaler"
        not in tree.named_steps
    )

    assert (
        "scaler"
        in knn.named_steps
    )


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

    assert (
        y.tolist()
        == [1]
    )


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
        match=(
            "Training data contains "
            "missing values"
        ),
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

    models = (
        create_candidate_models()
    )

    pipeline = models[
        "LogisticRegression"
    ]

    trained_pipeline = train_model(
        pipeline,
        X,
        y,
    )

    predictions = (
        trained_pipeline.predict(
            X
        )
    )

    assert (
        len(predictions)
        == len(y)
    )


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

    git_provenance = {
        "commit": (
            "b" * 40
        ),
        "dirty": False,
    }

    config = TrainingConfig()

    artifact = create_model_artifact(
        result=result,
        dataset_size=16,
        train_size=12,
        test_size=4,
        dataset_sha256=(
            dataset_sha256
        ),
        git_provenance=(
            git_provenance
        ),
        config=config,
    )

    assert (
        artifact["artifact_version"]
        == 3
    )

    assert (
        artifact["pipeline"]
        is pipeline
    )

    metadata = artifact[
        "metadata"
    ]

    training_run_id = metadata[
        "training_run_id"
    ]

    parsed_run_id = UUID(
        training_run_id
    )

    assert (
        str(parsed_run_id)
        == training_run_id
    )

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

    source = metadata[
        "source"
    ]

    assert (
        source["commit"]
        == "b" * 40
    )

    assert (
        source["dirty"]
        is False
    )

    environment = metadata[
        "environment"
    ]

    assert (
        "python"
        in environment
    )

    assert (
        "scikit_learn"
        in environment
    )

    assert (
        "numpy"
        in environment
    )

    assert (
        "pandas"
        in environment
    )

    assert (
        "joblib"
        in environment
    )

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

    training_config = (
        artifact["metadata"][
            "training_config"
        ]
    )

    assert (
        training_config["test_size"]
        == 0.25
    )

    assert (
        training_config["random_state"]
        == 42
    )

    assert (
        training_config["cv_folds"]
        == 5
    )

    assert (
        training_config[
            "min_cv_accuracy"
        ]
        == 0.75
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

    git_provenance = {
        "commit": (
            "b" * 40
        ),
        "dirty": False,
    }

    config = TrainingConfig()
    artifact = create_model_artifact(
        result=result,
        dataset_size=16,
        train_size=12,
        test_size=4,
        dataset_sha256=(
            dataset_sha256
        ),
        git_provenance=(
            git_provenance
        ),
        config=config,
    )

    save_model_artifact(
        artifact,
        model_path,
    )

    loaded_artifact = joblib.load(
        model_path
    )

    assert model_path.exists()

    assert (
        loaded_artifact[
            "artifact_version"
        ]
        == ARTIFACT_VERSION
    )

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

    source = metadata[
        "source"
    ]

    assert (
        source["commit"]
        == "b" * 40
    )

    assert (
        source["dirty"]
        is False
    )

    training_run_id = metadata[
        "training_run_id"
    ]

    parsed_run_id = UUID(
        training_run_id
    )

    assert (
        str(parsed_run_id)
        == training_run_id
    )

    assert (
        "environment"
        in metadata
    )

    assert (
        "trained_at"
        in metadata
    )

    training_config = metadata[
        "training_config"
    ]

    assert (
        training_config["test_size"]
        == 0.25
    )

    assert (
        training_config["random_state"]
        == 42
    )

    assert (
        training_config["cv_folds"]
        == 5
    )

    assert (
        training_config[
            "min_cv_accuracy"
        ]
        == 0.75
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

    models = (
        create_candidate_models()
    )

    pipeline = models[
        "LogisticRegression"
    ]

    config = TrainingConfig()

    (
        mean_accuracy,
        std_accuracy,
    ) = evaluate_model(
        pipeline,
        X,
        y,
        config,
    )

    assert (
        0.0
        <= mean_accuracy
        <= 1.0
    )

    assert (
        std_accuracy
        >= 0.0
    )


def test_validate_model_performance_passes():
    config = TrainingConfig()

    validate_model_performance(
        0.80,
        config,
    )


def test_validate_model_performance_fails():
    config = TrainingConfig()

    with pytest.raises(
        ValueError
    ):
        validate_model_performance(
            0.50,
            config,
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
    config = TrainingConfig()
    (
        X_train,
        X_test,
        y_train,
        y_test,
    ) = split_training_data(
        X,
        y,
        config,
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

    config = TrainingConfig()

    (
        X_train,
        X_test,
        y_train,
        y_test,
    ) = split_training_data(
        X,
        y,
        config,
    )

    models = (
        create_candidate_models()
    )

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

    config = TrainingConfig()

    (
        X_train,
        X_test,
        y_train,
        y_test,
    ) = split_training_data(
        X,
        y,
        config,
    )

    result = (
        train_and_evaluate_best_model(
            X_train,
            X_test,
            y_train,
            y_test,
            config,
        )
    )

    git_provenance = {
        "commit": (
            "b" * 40
        ),
        "dirty": False,
    }

    artifact = create_model_artifact(
        result=result,
        dataset_size=len(X),
        train_size=len(X_train),
        test_size=len(X_test),
        dataset_sha256=(
            dataset_sha256
        ),
        git_provenance=(
            git_provenance
        ),
        config=config,
    )

    save_model_artifact(
        artifact,
        model_path,
    )

    loaded_artifact = joblib.load(
        model_path
    )

    assert model_path.exists()

    assert (
        loaded_artifact[
            "artifact_version"
        ]
        == ARTIFACT_VERSION
    )

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

    source = metadata[
        "source"
    ]

    assert (
        source["commit"]
        == "b" * 40
    )

    assert (
        source["dirty"]
        is False
    )

    training_run_id = metadata[
        "training_run_id"
    ]

    parsed_run_id = UUID(
        training_run_id
    )

    assert (
        str(parsed_run_id)
        == training_run_id
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
        loaded_artifact[
            "pipeline"
        ]
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


def test_create_training_run_id():
    run_id = (
        create_training_run_id()
    )

    parsed = UUID(
        run_id
    )

    assert (
        str(parsed)
        == run_id
    )


def test_training_run_ids_are_unique():
    first_run_id = (
        create_training_run_id()
    )

    second_run_id = (
        create_training_run_id()
    )

    assert (
        first_run_id
        != second_run_id
    )


def test_get_git_provenance(
    monkeypatch,
    tmp_path,
):
    responses = [
        subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout=(
                "a" * 40
                + "\n"
            ),
        ),
        subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout="",
        ),
    ]

    def fake_run(
        *args,
        **kwargs,
    ):
        return responses.pop(0)

    monkeypatch.setattr(
        training.subprocess,
        "run",
        fake_run,
    )

    provenance = (
        training.get_git_provenance(
            tmp_path
        )
    )

    assert provenance == {
        "commit": (
            "a" * 40
        ),
        "dirty": False,
    }


def test_get_git_provenance_detects_dirty_tree(
    monkeypatch,
    tmp_path,
):
    responses = [
        subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout=(
                "b" * 40
                + "\n"
            ),
        ),
        subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout=(
                " M app/training.py\n"
            ),
        ),
    ]

    def fake_run(
        *args,
        **kwargs,
    ):
        return responses.pop(0)

    monkeypatch.setattr(
        training.subprocess,
        "run",
        fake_run,
    )

    provenance = (
        training.get_git_provenance(
            tmp_path
        )
    )

    assert (
        provenance["commit"]
        == "b" * 40
    )

    assert (
        provenance["dirty"]
        is True
    )


def test_training_config_defaults():
    config = TrainingConfig()

    assert config.test_size == 0.25
    assert config.random_state == 42
    assert config.cv_folds == 5
    assert (
        config.min_cv_accuracy
        == 0.75
    )


def test_training_config_can_be_customized():
    config = TrainingConfig(
        test_size=0.2,
        random_state=123,
        cv_folds=3,
        min_cv_accuracy=0.8,
    )

    assert config.test_size == 0.2
    assert config.random_state == 123
    assert config.cv_folds == 3
    assert (
        config.min_cv_accuracy
        == 0.8
    )


def test_training_config_rejects_invalid_test_size():
    config = TrainingConfig(
        test_size=1.5
    )

    with pytest.raises(
        ValueError
    ):
        validate_training_config_values(
            config
        )


def test_training_config_rejects_invalid_cv_folds():
    config = TrainingConfig(
        cv_folds=1
    )

    with pytest.raises(
        ValueError
    ):
        validate_training_config_values(
            config
        )


def test_training_config_rejects_invalid_accuracy_threshold():
    config = TrainingConfig(
        min_cv_accuracy=1.5
    )

    with pytest.raises(
        ValueError
    ):
        validate_training_config_values(
            config
        )


def test_validate_target_distribution():
    y= pd.Series([
        0,
        0,
        1,
        1,
    ])

    validate_target_distribution(
        y
    )

def test_validate_target_distribution_rejects_one_class():
    y = pd.Series([
        1,
        1,
        1,
        1,
    ])

    with pytest.raises(
        ValueError
    ):
        validate_target_distribution(
            y
        )


def test_validate_target_distribution_rejects_wrong_classes():
    y = pd.Series([
        1,
        1,
        2,
        2,
    ])

    with pytest.raises(
        ValueError
    ):
        validate_target_distribution(
            y
        )


def test_validate_target_distribution_rejects_tiny_class():
    y = pd.Series([
        0,
        0,
        0,
        1,
    ])

    with pytest.raises(
        ValueError
    ):
        validate_target_distribution(
            y
        )


def test_validate_cv_compatibility():
    y_train = pd.Series([
        0,
        0,
        0,
        0,
        0,
        1,
        1,
        1,
        1,
        1,
    ])

    config = TrainingConfig(
        cv_folds=5
    )

    validate_cv_compatibility(
        y_train,
        config,
    )


def test_validate_cv_compatibility_rejects_too_many_folds():
    y_train = pd.Series([
        0,
        0,
        0,
        0,
        0,
        1,
        1,
        1,
    ])

    config = TrainingConfig(
        cv_folds=4
    )

    with pytest.raises(
        ValueError
    ):
        validate_cv_compatibility(
            y_train,
            config,
        )


def test_validate_cv_compatibility_error_message():
    y_train = pd.Series([
        0,
        0,
        0,
        1,
        1,
    ])

    config = TrainingConfig(
        cv_folds=3
    )

    with pytest.raises(
        ValueError
    ) as exc_info:
        validate_cv_compatibility(
            y_train,
            config,
        )

    assert (
        "smallest class"
        in str(exc_info.value)
    )
