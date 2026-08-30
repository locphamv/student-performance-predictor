from pathlib import Path

from app.training import (
    create_model_artifact,
    load_training_data,
    save_model_artifact,
    train_model
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

pipeline = train_model(
    X,
    y,
)

artifact = create_model_artifact(
    pipeline
)

save_model_artifact(
    artifact,
    model_path,
)

print(
    "Model saved to:",
    model_path
)
