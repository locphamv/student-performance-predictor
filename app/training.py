from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

from app.features import FEATURE_NAMES

TARGET_NAME = "passed"
MODEL_VERSION = "1.0.0"
MIN_CV_ACCURACY = 0.75


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


def create_pipeline() -> Pipeline:
    return Pipeline([
        (
            "scaler",
            StandardScaler(),
        ),
        (
            "model",
            LogisticRegression(),
        ),
    ])


def evaluate_model(
    X: pd.DataFrame,
    y: pd.Series,
) -> tuple[float, float]:
    pipeline = create_pipeline()

    scores = cross_val_score(
        pipeline,
        X,
        y,
        cv=5,
        scoring="accuracy",
    )

    mean_accuracy = scores.mean()
    std_accuracy = scores.std()

    return (
        float(mean_accuracy),
        float(std_accuracy),
    )


def train_model(
        X: pd.DataFrame,
        y: pd.Series,
) -> Pipeline:
    pipeline = create_pipeline()

    pipeline.fit(
        X,
        y,
    )

    return pipeline


def create_model_artifact(
        pipeline: Pipeline,
        mean_cv_accuracy: float,
        std_cv_accuracy: float,
) -> dict:
    return {
        "pipeline": pipeline,
        "metadata": {
            "model_version": (
                MODEL_VERSION
            ),
            "feature_names": (
                FEATURE_NAMES
            ),
            "model_type": (
                "LogisticRegression"
            ),
            "mean_cv_accuracy": mean_cv_accuracy,
            "std_cv_accuracy": std_cv_accuracy,
        },
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
