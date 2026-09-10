import argparse
from pathlib import Path

from app.training import (
    calculate_file_sha256,
    create_model_artifact,
    get_git_provenance,
    load_training_data,
    save_model_artifact,
    split_training_data,
    train_and_evaluate_best_model,
    TrainingConfig,
    validate_training_config_values,
    validate_cv_compatibility,
    validate_target_distribution,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train and save student "
            "performance prediction model."
        )
    )

    parser.add_argument(
        "--data",
        type=Path,
        default=Path(
            "data/training_data.csv"
        ),
        help=(
            "Path to the training CSV file."
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "models/"
            "student-pass-pipeline.joblib"
        ),
        help=(
            "Path where the model artifact "
            "will be saved."
        ),
    )

    parser.add_argument(
        "--test-size",
        type=float,
        default=0.25,
        help=(
            "Fraction of data reserved "
            "for final testing."
        ),
    )

    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help=(
            "Random seed used for "
            "data splitting."
        ),
    )

    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help=(
            "Number of cross-validation "
            "folds."
        ),
    )

    parser.add_argument(
        "--min-cv-accuracy",
        type=float,
        default=0.75,
        help=(
            "Minimum required mean "
            "cross-validation accuracy."
        ),
    )

    return parser.parse_args()


def build_training_config(
    args: argparse.Namespace,
) -> TrainingConfig:
    return TrainingConfig(
        test_size=args.test_size,
        random_state=args.random_state,
        cv_folds=args.cv_folds,
        min_cv_accuracy=(
            args.min_cv_accuracy
        ),
    )


def main(
    data_path: Path,
    model_path: Path,
    config: TrainingConfig,
) -> None:
    validate_training_config_values(
        config
    )

    project_directory = (
        Path(__file__).parent
    )

    X, y = load_training_data(
        data_path
    )

    validate_target_distribution(
        y
    )

    dataset_sha256 = (
        calculate_file_sha256(
            data_path
        )
    )

    git_provenance = (
        get_git_provenance(
            project_directory
        )
    )

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

    validate_cv_compatibility(
        y_train,
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

    print(
        "\nBest model:",
        result.model_name,
    )

    print(
        "Best mean CV accuracy:",
        round(
            result.mean_cv_accuracy,
            3,
        ),
    )

    print(
        "Best CV accuracy std:",
        round(
            result.std_cv_accuracy,
            3,
        ),
    )

    print(
        "Final test accuracy:",
        round(
            result.test_accuracy,
            3,
        ),
    )

    print(
        "Dataset SHA-256:",
        dataset_sha256,
    )

    print(
        "Git commit:",
        git_provenance[
            "commit"
        ],
    )

    print(
        "Git working tree dirty:",
        git_provenance[
            "dirty"
        ],
    )

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

    print(
        "Model saved to:",
        model_path,
    )


if __name__ == "__main__":
    args = parse_args()

    config = build_training_config(
        args
    )

    main(
        data_path=args.data,
        model_path=args.output,
        config=config,
    )
