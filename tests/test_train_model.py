import argparse
from pathlib import Path

from app.training import (
    ClassificationMetrics,
    TrainingConfig,
)
from types import SimpleNamespace

import pandas as pd

import train_model


def test_main_exists():
    assert callable(
        train_model.main
    )


def test_main_runs_training_workflow(
    monkeypatch,
    tmp_path,
):
    X = pd.DataFrame(
        {
            "study_hours": [
                1.0,
                6.0,
            ],
            "absences": [
                5,
                1,
            ],
            "previous_score": [
                3.0,
                8.0,
            ],
        }
    )

    y = pd.Series([
        0,
        1,
    ])

    result = SimpleNamespace(
        model_name=(
            "LogisticRegression"
        ),
        mean_cv_accuracy=0.8,
        std_cv_accuracy=0.1,
        test_metrics=ClassificationMetrics(
            accuracy=0.75,
            precision=0.8,
            recall=0.7,
            f1=0.75,
            confusion_matrix=[
                [2, 1],
                [1, 2],
            ],
        ),
    )

    monkeypatch.setattr(
        train_model,
        "load_training_data",
        lambda path: (
            X,
            y,
        ),
    )

    monkeypatch.setattr(
        train_model,
        "validate_training_config_values",
        lambda config: None,
    )

    monkeypatch.setattr(
        train_model,
        "validate_target_distribution",
        lambda y: None,
    )

    monkeypatch.setattr(
        train_model,
        "validate_cv_compatibility",
        lambda y, config: None,
    )

    monkeypatch.setattr(
        train_model,
        "calculate_file_sha256",
        lambda path: (
            "a" * 64
        ),
    )

    monkeypatch.setattr(
        train_model,
        "get_git_provenance",
        lambda path: {
            "commit": (
                "b" * 40
            ),
            "dirty": False,
        },
    )

    monkeypatch.setattr(
        train_model,
        "split_training_data",
        lambda X, y, config: (
            X,
            X,
            y,
            y,
        ),
    )

    monkeypatch.setattr(
        train_model,
        "train_and_evaluate_best_model",
        lambda X_train, X_test, y_train, y_test, config: result,
    )

    fake_artifact = {
        "artifact_version": 3,
    }

    monkeypatch.setattr(
        train_model,
        "create_model_artifact",
        lambda **kwargs: (
            fake_artifact
        ),
    )

    saved = {}

    def fake_save(
        artifact,
        model_path,
    ):
        saved[
            "artifact"
        ] = artifact

        saved[
            "model_path"
        ] = model_path

    monkeypatch.setattr(
        train_model,
        "save_model_artifact",
        fake_save,
    )

    data_path = (
        tmp_path
        / "training.csv"
    )

    model_path = (
        tmp_path
        / "model.joblib"
    )

    config = TrainingConfig()

    train_model.main(
        data_path=data_path,
        model_path=model_path,
        config=config,
    )

    assert (
        saved["artifact"]
        == fake_artifact
    )

    assert (
        saved["model_path"]
        == model_path
    )


def test_parse_args(
    monkeypatch,
):
    monkeypatch.setattr(
        "sys.argv",
        [
            "train_model.py",
            "--data",
            "custom.csv",
            "--output",
            "custom.joblib",
            "--test-size",
            "0.2",
            "--random-state",
            "123",
            "--cv-folds",
            "3",
            "--min-cv-accuracy",
            "0.8",
        ],
    )

    args = train_model.parse_args()

    assert (
        args.data
        == Path(
            "custom.csv"
        )
    )

    assert (
        args.output
        == Path(
            "custom.joblib"
        )
    )

    assert args.test_size == 0.2
    assert args.random_state == 123
    assert args.cv_folds == 3

    assert (
        args.min_cv_accuracy
        == 0.8
    )


def test_parse_args_defaults(
    monkeypatch,
):
    monkeypatch.setattr(
        "sys.argv",
        [
            "train_model.py",
        ],
    )

    args = train_model.parse_args()

    assert (
        args.data
        == Path(
            "data/training_data.csv"
        )
    )

    assert (
        args.output
        == Path(
            "models/"
            "student-pass-pipeline.joblib"
        )
    )


def test_build_training_config():
    args = argparse.Namespace(
        test_size=0.2,
        random_state=123,
        cv_folds=3,
        min_cv_accuracy=0.8,
    )

    config = (
        train_model
        .build_training_config(
            args
        )
    )

    assert config == TrainingConfig(
        test_size=0.2,
        random_state=123,
        cv_folds=3,
        min_cv_accuracy=0.8,
    )
