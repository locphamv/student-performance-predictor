from pathlib import Path

from app.training import (
    create_candidate_models,
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
candidate_models = (
    create_candidate_models()
)

best_model_name = None
best_pipeline = None

best_mean_accuracy = -1.0
best_std_accuracy = 0.0

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
        f"{model_name}:"
        f"mean={mean_accuracy: .3f},"
        f"std={std_accuracy:.3f}"
    )
    if (
        mean_accuracy
        > best_mean_accuracy
    ):
        best_model_name = model_name
        best_pipeline = pipeline
        best_mean_accuracy = (
            mean_accuracy
        )
        best_std_accuracy = (
            std_accuracy
        )


print(
    "\nBest model:",
    best_model_name,
)

print(
    "Best mean CV accuracy:",
    round(
        best_mean_accuracy,
        3,
    ),
)

print(
    "Best CV accuracy std:",
    round(
        best_std_accuracy,
        3,
    ),
)

validate_model_performance(
    best_mean_accuracy
)

if best_pipeline is None:
    raise RuntimeError(
        "No candidate model was selected"
    )

final_pipeline = train_model(
    best_pipeline,
    X,
    y,
)

if best_model_name is None:
    raise RuntimeError(
        "No candidate model name was selected"
    )

artifact = create_model_artifact(
    model_name=best_model_name,
    pipeline=final_pipeline,
    mean_cv_accuracy=best_mean_accuracy,
    std_cv_accuracy=best_std_accuracy,
)

save_model_artifact(
    artifact,
    model_path,
)

print(
    "Model saved to:",
    model_path,
)
