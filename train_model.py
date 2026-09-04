from pathlib import Path

from app.training import (
    calculate_file_sha256,
    create_model_artifact,
    load_training_data,
    save_model_artifact,
    split_training_data,
    train_and_evaluate_best_model,
)


project_directory = Path(
    __file__
).parent

data_path = (
    project_directory
    / "data"
    / "training_data.csv"
)

model_path = (
    project_directory
    / "models"
    / "student-pass-pipeline.joblib"
)


X, y = load_training_data(
    data_path
)


dataset_sha256 = (
    calculate_file_sha256(
        data_path
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
)


result = train_and_evaluate_best_model(
    X_train,
    X_test,
    y_train,
    y_test,
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


artifact = create_model_artifact(
    result=result,
    dataset_size=len(X),
    train_size=len(X_train),
    test_size=len(X_test),
    dataset_sha256=dataset_sha256,
)


save_model_artifact(
    artifact,
    model_path,
)


print(
    "Model saved to:",
    model_path,
)
