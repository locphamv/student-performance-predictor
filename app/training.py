from dataclasses import dataclass
from datetime import (
    datetime,
    timezone,
)
import hashlib
from importlib.metadata import version
from pathlib import Path
import platform
import subprocess
from uuid import uuid4

import joblib
import pandas as pd
from sklearn.linear_model import (
    LogisticRegression,
)
from sklearn.metrics import (
    accuracy_score,
)
from sklearn.model_selection import (
    cross_val_score,
    train_test_split,
)
from sklearn.neighbors import (
    KNeighborsClassifier,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    StandardScaler,
)
from sklearn.tree import (
    DecisionTreeClassifier,
)

from app.features import FEATURE_NAMES


MODEL_VERSION = "1.0.0"
ARTIFACT_VERSION = 3


@dataclass(frozen=True)
class TrainingConfig:
    test_size: float = 0.25
    random_state: int = 42
    cv_folds: int = 5
    min_cv_accuracy: float = 0.75


@dataclass
class ModelSelectionResult:
    model_name: str
    pipeline: Pipeline
    mean_cv_accuracy: float
    std_cv_accuracy: float


@dataclass
class TrainingResult:
    model_name: str
    pipeline: Pipeline
    mean_cv_accuracy: float
    std_cv_accuracy: float
    test_accuracy: float


def create_candidate_models() -> dict[
    str,
    Pipeline,
]:
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


def load_training_data(
    data_path: Path,
) -> tuple[
    pd.DataFrame,
    pd.Series,
]:
    data = pd.read_csv(
        data_path
    )

    if data.empty:
        raise ValueError(
            "Training data is empty"
        )

    required_columns = set(
        FEATURE_NAMES
        + ["passed"]
    )

    missing_columns = (
        required_columns
        - set(data.columns)
    )

    if missing_columns:
        missing = sorted(
            missing_columns
        )

        raise ValueError(
            "Training data is missing "
            "required columns: "
            f"{missing}"
        )

    if (
        data[
            list(required_columns)
        ]
        .isnull()
        .any()
        .any()
    ):
        raise ValueError(
            "Training data contains missing values"
        )

    X = data[
        FEATURE_NAMES
    ]

    y = data[
        "passed"
    ]

    return X, y


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


def evaluate_model(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    config: TrainingConfig,
) -> tuple[
    float,
    float,
]:
    scores = cross_val_score(
        pipeline,
        X,
        y,
        cv=config.cv_folds,
        scoring="accuracy",
    )

    return (
        float(scores.mean()),
        float(scores.std()),
    )


def validate_model_performance(
    mean_accuracy: float,
    config: TrainingConfig,
) -> None:
    if (
        mean_accuracy
        < config.min_cv_accuracy
    ):
        raise ValueError(
            "Model did not meet the minimum "
            "cross-validation accuracy: "
            f"{mean_accuracy:.3f} "
            f"< "
            f"{config.min_cv_accuracy:.3f}"
        )


def split_training_data(
    X: pd.DataFrame,
    y: pd.Series,
    config: TrainingConfig,
):
    return train_test_split(
        X,
        y,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=y,
    )


def evaluate_final_model(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> float:
    predictions = pipeline.predict(
        X_test
    )

    return float(
        accuracy_score(
            y_test,
            predictions,
        )
    )


def select_best_model(
    candidate_models: dict[
        str,
        Pipeline,
    ],
    X: pd.DataFrame,
    y: pd.Series,
    config: TrainingConfig
) -> ModelSelectionResult:
    if not candidate_models:
        raise ValueError(
            "No candidate models were provided"
        )

    best_result = None

    for (
        model_name,
        pipeline,
    ) in candidate_models.items():
        (
            mean_accuracy,
            std_accuracy,
        ) = evaluate_model(
            pipeline,
            X,
            y,
            config,
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
            or (
                current_result
                .mean_cv_accuracy
                > best_result
                .mean_cv_accuracy
            )
        ):
            best_result = (
                current_result
            )

    assert (
        best_result
        is not None
    )

    return best_result


def train_and_evaluate_best_model(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    config: TrainingConfig,
) -> TrainingResult:
    candidate_models = (
        create_candidate_models()
    )

    selection = select_best_model(
        candidate_models,
        X_train,
        y_train,
        config,
    )

    validate_model_performance(
        selection.mean_cv_accuracy,
        config,
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
        model_name=(
            selection.model_name
        ),
        pipeline=fitted_pipeline,
        mean_cv_accuracy=(
            selection.mean_cv_accuracy
        ),
        std_cv_accuracy=(
            selection.std_cv_accuracy
        ),
        test_accuracy=test_accuracy,
    )


def get_environment_versions() -> dict[
    str,
    str,
]:
    return {
        "python": (
            platform.python_version()
        ),
        "scikit_learn": (
            version(
                "scikit-learn"
            )
        ),
        "numpy": version(
            "numpy"
        ),
        "pandas": version(
            "pandas"
        ),
        "joblib": version(
            "joblib"
        ),
    }


def calculate_file_sha256(
    file_path: Path,
) -> str:
    sha256 = hashlib.sha256()

    with file_path.open(
        "rb"
    ) as file:
        while chunk := file.read(
            8192
        ):
            sha256.update(
                chunk
            )

    return sha256.hexdigest()


def create_training_run_id() -> str:
    return str(
        uuid4()
    )


def get_git_provenance(
    project_directory: Path,
) -> dict[
    str,
    str | bool,
]:
    try:
        commit_result = (
            subprocess.run(
                [
                    "git",
                    "rev-parse",
                    "HEAD",
                ],
                cwd=project_directory,
                check=True,
                capture_output=True,
                text=True,
            )
        )

        status_result = (
            subprocess.run(
                [
                    "git",
                    "status",
                    "--porcelain",
                ],
                cwd=project_directory,
                check=True,
                capture_output=True,
                text=True,
            )
        )

    except (
        subprocess.CalledProcessError,
        FileNotFoundError,
    ) as exc:
        raise RuntimeError(
            "Could not read Git provenance"
        ) from exc

    commit = (
        commit_result
        .stdout
        .strip()
    )

    dirty = bool(
        status_result
        .stdout
        .strip()
    )

    return {
        "commit": commit,
        "dirty": dirty,
    }


def create_model_artifact(
    result: TrainingResult,
    dataset_size: int,
    train_size: int,
    test_size: int,
    dataset_sha256: str,
    git_provenance: dict[
        str,
        str | bool,
    ],
    config: TrainingConfig,
) -> dict:
    environment_versions = (
        get_environment_versions()
    )

    trained_at = datetime.now(
        timezone.utc
    ).isoformat()

    training_run_id = (
        create_training_run_id()
    )

    return {
        "artifact_version": (
            ARTIFACT_VERSION
        ),
        "pipeline": (
            result.pipeline
        ),
        "metadata": {
            "training_run_id": (
                training_run_id
            ),
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
            "source": (
                git_provenance
            ),
            "training_config": {
                "test_size": (
                    config.test_size
                ),
                "random_state": (
                    config.random_state
                ),
                "cv_folds": (
                    config.cv_folds
                ),
                "min_cv_accuracy": (
                    config.min_cv_accuracy
                ),
            },
        }
    }


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


def validate_training_config_values(
    config: TrainingConfig,
) -> None:
    if not (
        0.0
        < config.test_size
        < 1.0
    ):
        raise ValueError(
            "test_size must be between "
            "0 and 1"
        )

    if config.cv_folds < 2:
        raise ValueError(
            "cv_folds must be at least 2"
        )

    if not (
        0.0
        <= config.min_cv_accuracy
        <= 1.0
    ):
        raise ValueError(
            "min_cv_accuracy must be "
            "between 0 and 1"
        )


def validate_target_distribution(
        y: pd.Series
) -> None:
    class_counts = (
        y.value_counts()
    )

    expected_classes = {
        0,
        1,
    }

    actual_classes = set(
        class_counts.index
    )

    if (
        actual_classes
        != expected_classes
    ):
        raise ValueError(
            "Training target must contain "
            "exactly classes 0 and 1"
        )

    if (
        class_counts.min()
        <2
    ):
        raise ValueError(
            "Each target class must contain "
            "at least two samples"
        )


def validate_cv_compatibility(
        y_train: pd.Series,
        config: TrainingConfig,
) -> None:
    class_counts = (
        y_train.value_counts()
    )

    smallest_class_size = int(
        class_counts.min()
    )

    if(
        config.cv_folds> smallest_class_size
    ):
        raise ValueError(
            "cv_folds cannot exceed "
            "the number of training samples "
            "in the smallest class: "
            f"{config.cv_folds} > "
            f"{smallest_class_size}"
        )
