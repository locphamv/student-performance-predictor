from dataclasses import dataclass
from datetime import (
    datetime,
    timezone,
)
from importlib.metadata import version
import platform
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (
    cross_val_score,
    train_test_split,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

from app.features import FEATURE_NAMES
import hashlib

TARGET_NAME = "passed"
MODEL_VERSION = "1.0.0"
MIN_CV_ACCURACY = 0.75

def get_environment_versions() -> dict[str, str]:
    return {
        "python": (
            platform.python_version()
        ),
        "scikit_learn": version(
            "scikit-learn"
        ),
        "numpy": version(
            "numpy"
        ),
        "pandas": version(
            "pandas"
        ),
        "joblib": version(
            "joblib"
        )
    }


def load_training_data(
    data_path: Path,
) -> tuple[pd.DataFrame, pd.Series]:
    data = pd.read_csv(
        data_path
    )

    if data.empty:
        raise ValueError(
            "Training data is empty"
        )

    required_columns = (
        set(FEATURE_NAMES)
        | {TARGET_NAME}
    )

    missing_columns = (
        required_columns
        - set(data.columns)
    )

    if missing_columns:
        raise ValueError(
            "Training data is missing columns:"
            f"{sorted(missing_columns)}"
        )

    if data.isnull().any().any():
        raise ValueError(
            "Training data contains missing values"
        )

    X = data[
        FEATURE_NAMES
    ]

    y = data[
        TARGET_NAME
    ]

    return X, y


def create_candidate_models() -> dict[str, Pipeline]:
    return {
        "LogisticRegression": Pipeline([
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "model",
                LogisticRegression(),
            ),
        ]),
        "DecisionTree": Pipeline([
            (
                "model",
                DecisionTreeClassifier(
                    max_depth=3,
                    random_state=42,
                ),
            ),
        ]),

        "KNN": Pipeline([
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "model",
                KNeighborsClassifier(
                    n_neighbors=3,
                ),
            ),
        ]),
    }


def evaluate_model(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
) -> tuple[float, float]:

    scores = cross_val_score(
        pipeline,
        X,
        y,
        cv=5,
        scoring="accuracy",
    )

    return (
        float(scores.mean()),
        float(scores.std()),
    )


def train_model(
        pipeline: Pipeline,
        X: pd.DataFrame,
        y: pd.Series,
) -> Pipeline:

    pipeline.fit(
        X,
        y,
    )

    return pipeline


@dataclass
class ModelSelectionResult:
    model_name: str
    pipeline: Pipeline
    mean_cv_accuracy: float
    std_cv_accuracy: float


def save_model_artifact(
    artifact: dict,
    model_path: Path,
) -> None:
    model_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        artifact,
        model_path,
    )


def validate_model_performance(
    mean_accuracy: float,
) -> None:
    if mean_accuracy < MIN_CV_ACCURACY:
        raise ValueError(
            "Model did not meet the minimum "
            "cross-validation accuracy: "
            f"{mean_accuracy:.3f} "
            f"< {MIN_CV_ACCURACY:.3f}"
        )


def split_training_data(
    X: pd.DataFrame,
    y: pd.Series,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
]:
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    return X_train, X_test, y_train, y_test


def evaluate_final_model(
        pipeline: Pipeline,
        X_test: pd.DataFrame,
        y_test: pd.Series,
) -> float:
    predictions = pipeline.predict(
        X_test
    )

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    return float(accuracy)


def select_best_model(
    candidate_models: dict[str, Pipeline],
    X: pd.DataFrame,
    y: pd.Series,
) -> ModelSelectionResult:
    if not candidate_models:
        raise ValueError(
            "No candidate models were provided"
        )

    best_result = None

    for model_name, pipeline in (
        candidate_models.items()
    ):
        mean_accuracy, std_accuracy = (
            evaluate_model(
                pipeline,
                X,
                y,
            )
        )

        print(
            f"{model_name}: "
            f"mean={mean_accuracy:.3f}, "
            f"std={std_accuracy:.3f}"
        )

        current_result = (
            ModelSelectionResult(
                model_name=model_name,
                pipeline=pipeline,
                mean_cv_accuracy=(
                    mean_accuracy
                ),
                std_cv_accuracy=(
                    std_accuracy
                ),
            )
        )

        if (
            best_result is None
            or current_result.mean_cv_accuracy
            > best_result.mean_cv_accuracy
        ):
            best_result = (
                current_result
            )
    assert best_result is not None
    return best_result


@dataclass
class TrainingResult:
    model_name: str
    pipeline: Pipeline
    mean_cv_accuracy: float
    std_cv_accuracy: float
    test_accuracy: float


def train_and_evaluate_best_model(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> TrainingResult:
    candidate_models = (
        create_candidate_models()
    )

    selection = select_best_model(
        candidate_models,
        X_train,
        y_train,
    )

    validate_model_performance(
        selection.mean_cv_accuracy
    )

    fitted_pipeline = train_model(
        selection.pipeline,
        X_train,
        y_train,
    )

    test_accuracy = (
        evaluate_final_model(
            fitted_pipeline,
            X_test,
            y_test,
        )
    )

    return TrainingResult(
        model_name=selection.model_name,
        pipeline=fitted_pipeline,
        mean_cv_accuracy=(
            selection.mean_cv_accuracy
        ),
        std_cv_accuracy=(
            selection.std_cv_accuracy
        ),
        test_accuracy=test_accuracy,
    )

def create_model_artifact(
    result: TrainingResult,
    dataset_size: int,
    train_size: int,
    test_size: int,
    dataset_sha256: str,
) -> dict:
    environment_versions = (
        get_environment_versions()
    )

    trained_at = datetime.now(
        timezone.utc
    ).isoformat()

    return {
        "pipeline": result.pipeline,
        "metadata": {
            "model_version": (
                MODEL_VERSION
            ),
            "model_type": (
                result.model_name
            ),
            "feature_names": (
                FEATURE_NAMES
            ),
            "mean_cv_accuracy": (
                result.mean_cv_accuracy
            ),
            "std_cv_accuracy": (
                result.std_cv_accuracy
            ),
            "test_accuracy": (
                result.test_accuracy
            ),
            "trained_at": (
                trained_at
            ),
            "dataset_size": (
                dataset_size
            ),
            "train_size": (
                train_size
            ),
            "test_size": (
                test_size
            ),
            "dataset_sha256": (
                dataset_sha256
            ),
            "environment": (
                environment_versions
            ),
        },
    }

def calculate_file_sha256(
        file_path: Path,
) -> str:
    sha256 = hashlib.sha256()

    with file_path.open(
        "rb"
    ) as file:
        while chunk:= file.read(
            8192
        ):
            sha256.update(
                chunk
            )

    return sha256.hexdigest()

