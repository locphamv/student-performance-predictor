# Student Performance Predictor

An end-to-end machine learning project for predicting whether a student will pass based on study hours, absences, and previous score.

## Current Progress

- Data preprocessing
- Train/test split
- scikit-learn Pipeline
- Model training and evaluation
- Model persistence with joblib
- Model metadata and versioning
- Training/inference feature contract
- Environment-based configuration
- FastAPI model serving
- Request validation with Pydantic
- Health and model-info endpoints
- Custom model-serving exception handling
- Inference logging and latency measurement
- API/integration tests
- Unit tests for the prediction service

## ML Serving Flow

```text
training data
↓
preprocessing + model
↓
trained pipeline
↓
model artifact
↓
FastAPI startup
↓
load and validate model
↓
POST /predict
↓
request validation
↓
feature construction
↓
model inference
↓
JSON response
```

## Goal

The project is designed to practice the complete ML workflow:

data → preprocessing → training → evaluation → model artifact
→ API inference

## Status

Work in progress.
