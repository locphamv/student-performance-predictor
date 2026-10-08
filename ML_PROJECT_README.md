# Student Performance Predictor

An end-to-end Machine Learning project that trains, evaluates, versions, and serves a student pass/fail classifier through a FastAPI API.

The goal of this repository is not just to train a model, but to practice the full ML engineering lifecycle:

```text
data
↓
validation
↓
train / test split
↓
cross-validation
↓
candidate model comparison
↓
final held-out evaluation
↓
refit winning recipe on all labeled data
↓
versioned model artifact
↓
FastAPI inference service
↓
tests + reproducibility metadata
```

---

## Project Goals

This project was built to practice the transition from notebook-style machine learning into a small production-style ML system.

It covers:

- supervised binary classification
- preprocessing with scikit-learn `Pipeline`
- cross-validation
- model selection
- held-out test evaluation
- model artifact persistence with `joblib`
- artifact validation and versioning
- reproducibility metadata
- FastAPI model serving
- service-layer separation
- structured error handling
- automated tests

---

## Tech Stack

- Python
- pandas
- NumPy
- scikit-learn
- FastAPI
- Uvicorn
- joblib
- pytest

---

## Project Structure

```text
student-performance-predictor/
├── app/
│   ├── __init__.py
│   ├── config.py
│   ├── exceptions.py
│   ├── features.py
│   ├── main.py
│   ├── schemas.py
│   ├── training.py
│   └── services/
│       └── prediction_service.py
├── data/
│   └── training_data.csv
├── models/
│   └── student-pass-pipeline.joblib
├── tests/
│   ├── test_features.py
│   ├── test_main.py
│   ├── test_prediction_service.py
│   ├── test_training.py
│   └── test_train_model.py
├── requirements.txt
├── train_model.py
└── README.md
```

---

## Training Pipeline

The training workflow follows a strict separation between model selection, final evaluation, and deployment training.

```text
full labeled dataset
↓
train / test split
↓
cross-validation on training split only
↓
compare candidate models
↓
fit selected evaluation model on train split
↓
evaluate once on untouched test split
↓
freeze winning model recipe
↓
clone the pipeline
↓
refit fresh deployment pipeline on all labeled data
↓
save versioned artifact
```

The important rule is that the final deployment model is trained on all labeled data **only after** the winning recipe has already been selected and evaluated.

The saved test metrics therefore represent held-out evidence for the selected model recipe, not an evaluation of the final refitted deployment instance.

---

## Candidate Models

The training pipeline compares multiple scikit-learn models:

```text
Logistic Regression
Decision Tree
K-Nearest Neighbors
```

The models are wrapped in pipelines where preprocessing is required.

Example:

```python
Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression()),
])
```

KNN also uses feature scaling, while the Decision Tree does not require it.

Cross-validation is used to compare candidate models before the final held-out test evaluation.

---

## Evaluation Metrics

The selected classifier is evaluated using:

- Accuracy
- Precision
- Recall
- F1 score
- Confusion matrix

Conceptually:

```text
ClassificationMetrics
├── accuracy
├── precision
├── recall
├── f1
└── confusion_matrix
```

The final test set is evaluated once after model selection.

---

## Model Artifact

The trained deployment pipeline is stored as:

```text
models/student-pass-pipeline.joblib
```

The artifact contains more than just the fitted estimator.

It also stores metadata used to validate and understand the model at inference time, including information such as:

```text
artifact version
training run ID
selected model
test metrics
dataset size
deployment training size
dataset fingerprint
environment metadata
Git provenance
```

The service validates the artifact before using it.

This helps prevent an incompatible or incomplete model file from being silently loaded.

---

## Reproducibility

The training pipeline records reproducibility information such as:

- dataset SHA-256 fingerprint
- Python / environment metadata
- training run UUID
- Git provenance
- artifact version
- dataset size
- deployment training size

This makes it easier to answer questions such as:

```text
Which data produced this model?
Which training run produced this artifact?
Which code revision was used?
Which model recipe won?
What held-out metrics were recorded?
```

---

## FastAPI Service

The model is served through FastAPI.

The application loads the model during application startup and releases it during shutdown using the FastAPI lifespan lifecycle.

The HTTP layer is kept separate from prediction logic:

```text
FastAPI endpoint
↓
request schema
↓
prediction service
↓
feature construction
↓
scikit-learn pipeline
↓
prediction / probability
↓
response schema
```

The prediction service can therefore be tested independently from HTTP.

---

## API Endpoints

### Health Check

```http
GET /health
```

Used to verify that the API is running.

Example:

```bash
curl http://127.0.0.1:8000/health
```

---

### Predict

```http
POST /predict
```

Accepts student feature values and returns a pass/fail prediction together with model output information.

The request and response schemas are defined with Pydantic.

After starting the API, the easiest way to inspect and test the exact request format is:

```text
http://127.0.0.1:8000/docs
```

---

### Model Information

```http
GET /model-info
```

Returns public metadata about the currently loaded model.

The public API intentionally exposes only selected metadata rather than the entire internal artifact structure.

Example:

```bash
curl http://127.0.0.1:8000/model-info
```

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/locphamv/student-performance-prediction.git
cd student-performance-predictor
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it.

macOS / Linux:

```bash
source .venv/bin/activate
```

Windows:

```bash
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

---

## Train the Model

Run:

```bash
python train_model.py
```

The training script performs the complete model-selection and artifact-building workflow.

If you want to inspect the available CLI options:

```bash
python train_model.py --help
```

A successful training run produces or updates:

```text
models/student-pass-pipeline.joblib
```

---

## Run the API

Start FastAPI with Uvicorn:

```bash
uvicorn app.main:app --reload
```

Then open:

```text
http://127.0.0.1:8000/docs
```

The root route `/` is intentionally not required, so a `404` there does not mean the application is broken.

Use `/health`, `/predict`, `/model-info`, or `/docs`.

---

## Run Tests

Run the full test suite:

```bash
pytest
```

Or with more detail:

```bash
pytest -v
```

The tests cover areas including:

```text
feature construction
training utilities
model selection
artifact behavior
prediction service
FastAPI endpoints
training entry point
```

---

## Architecture

A simplified view of the project:

```text
                    TRAINING
                       │
                       ▼
              data/training_data.csv
                       │
                       ▼
                validation / split
                       │
                       ▼
              candidate model CV
                       │
                       ▼
                winning recipe
                       │
                       ▼
             held-out evaluation
                       │
                       ▼
             refit on full dataset
                       │
                       ▼
        student-pass-pipeline.joblib
                       │
                       │
              ─────────┴─────────
                       │
                    SERVING
                       │
                       ▼
                 FastAPI lifespan
                       │
                       ▼
              PredictionService
                       │
                       ▼
              build_feature_array
                       │
                       ▼
             sklearn Pipeline
                       │
                       ▼
            prediction + probability
```

---

## Engineering Decisions

### Pipeline-based preprocessing

Preprocessing is kept inside the scikit-learn pipeline so the same transformations are applied during training and inference.

This avoids manually reproducing preprocessing logic at prediction time.

### Service layer

Prediction logic lives outside the HTTP route handlers.

This keeps:

```text
HTTP concerns
```

separate from:

```text
ML inference concerns
```

and makes unit testing easier.

### Artifact versioning

The model artifact has an explicit version.

The prediction service supports known artifact versions and rejects incompatible artifacts instead of assuming every serialized file is valid.

### Best recipe vs deployment model

The model evaluated on the held-out test set and the final deployment model are not the same fitted object.

```text
evaluation model
→ fitted on training split
→ evaluated on untouched test split

deployment model
→ fresh clone of winning recipe
→ fitted on all labeled data
→ saved for inference
```

This preserves a clean final evaluation while allowing the deployed model to learn from all available labeled examples.

---

## Current Limitations

This repository is an educational ML engineering project.

The included dataset is intentionally very small, so the reported metrics should **not** be interpreted as strong evidence that the model will generalize to real student populations.

In particular:

- the dataset contains only a small number of labeled examples
- the held-out test set is very small
- model comparison results can therefore be unstable
- the API is intended as a learning deployment, not a production service

The value of this project is primarily the **engineering workflow**, not the predictive performance of the toy dataset.

---

## What I Learned

This project helped me move beyond:

```python
model.fit(X, y)
```

and understand how a complete ML system fits together:

```text
data quality
↓
reproducible training
↓
model selection
↓
evaluation discipline
↓
artifact packaging
↓
runtime validation
↓
service architecture
↓
API serving
↓
testing
```

It serves as the bridge between my classical Machine Learning fundamentals and my next phase of Deep Learning / PyTorch work.

---

## Future Improvements

Possible extensions include:

- use a larger real-world dataset
- add stronger feature validation
- add structured logging and monitoring
- containerize the API with Docker
- add CI for automated tests
- add model registry / external artifact storage
- expose richer model monitoring metadata
- deploy the API to a cloud platform

---

## Project Status

```text
Training pipeline        ✅
Model selection          ✅
Held-out evaluation      ✅
Full-data deployment fit ✅
Artifact versioning      ✅
Reproducibility metadata ✅
FastAPI serving          ✅
Automated tests          ✅
```

Core project complete.
