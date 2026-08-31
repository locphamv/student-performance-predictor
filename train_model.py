from pathlib import Path

from app.training import (
    create_model_artifact,
    load_training_data,
    save_model_artifact,
    train_model,
    evaluate_model,
    validate_model_performance,
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

mean_accuracy, std_accuracy = (
    evaluate_model(
        X,
        y,
    )
)

print(
    "Mean CV accuracy:",
    round(
        mean_accuracy,
        3,
    ),
)

print(
    "CV accuracy std:",
    round(
        std_accuracy,
        3,
    ),
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

print(
    "Model saved to:",
    model_path,
)
