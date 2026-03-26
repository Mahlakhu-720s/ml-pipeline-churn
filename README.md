# End-to-End ML Pipeline: Customer Churn Prediction

> **Netcare AI Engineer Portfolio Project 1** — Demonstrates production-grade MLOps on Databricks with MLflow, Delta Lake, Spark, and automated CI/CD.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![MLflow](https://img.shields.io/badge/MLflow-2.x-orange)](https://mlflow.org/)
[![PySpark](https://img.shields.io/badge/PySpark-3.5-red)](https://spark.apache.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![CI](https://img.shields.io/badge/CI-GitHub%20Actions-black)](https://github.com/features/actions)

---

## Overview

This project implements a **full machine learning lifecycle** for customer churn prediction — from raw data ingestion through to a live, monitored REST API endpoint. It is designed to demonstrate every stage of an enterprise MLOps workflow as practised on Databricks, and is directly transferable to healthcare predictive analytics.

The dataset used is the **IBM Telco Customer Churn** dataset (publicly available, 7,043 records). The methodology, pipeline architecture, and governance patterns are identical to those applied in clinical prediction use cases such as patient readmission, no-show prediction, and length-of-stay modelling.

---

## Architecture

```
Raw Data (Delta Table)
        │
        ▼
┌─────────────────────┐
│  Feature Engineering │  ← Spark transformations, Feature Store
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│  Model Training      │  ← MLflow experiment tracking, cross-validation
│  (XGBoost / RF)     │
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│  Model Registry      │  ← Staging → Production promotion gates
│  (MLflow)           │
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│  Model Serving       │  ← REST endpoint, input validation
│  (FastAPI / Databricks Serving) │
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│  Monitoring          │  ← Drift detection, retraining trigger
│  & Governance        │
└─────────────────────┘
```

---

## Key Features

| Feature | Implementation |
|---|---|
| **Distributed feature engineering** | PySpark transformations on Delta Lake tables |
| **Experiment tracking** | MLflow — parameters, metrics, artifacts, tags |
| **Model versioning** | MLflow Model Registry with Staging/Production gates |
| **Reproducibility** | Data lineage, MLflow run IDs, pinned dependency hashes |
| **REST serving** | FastAPI with JWT auth, Pydantic input validation, structured logging |
| **Data drift detection** | PSI & KS-test based drift monitoring with automated alerts |
| **Retraining trigger** | Automated GitHub Actions workflow on drift threshold breach |
| **Responsible AI** | SHAP explainability, bias audit across demographic subgroups |
| **CI/CD** | GitHub Actions — linting, unit tests, integration tests, model validation |

---

## Repository Structure

```
ml-pipeline-churn/
│
├── README.md                    # This file
├── LICENSE
├── .gitignore
├── requirements.txt             # Pinned dependencies
├── setup.py                     # Package installation
│
├── configs/
│   ├── model_config.yaml        # Hyperparameter search space
│   ├── feature_config.yaml      # Feature definitions
│   └── serving_config.yaml      # API and monitoring thresholds
│
├── src/
│   ├── data/
│   │   ├── ingest.py            # Raw data → Delta Lake ingestion
│   │   └── validate.py          # Great Expectations data quality checks
│   │
│   ├── features/
│   │   ├── engineering.py       # Spark feature transformations
│   │   ├── feature_store.py     # Feature Store read/write helpers
│   │   └── schemas.py           # Feature schemas and type definitions
│   │
│   ├── models/
│   │   ├── train.py             # Model training with MLflow logging
│   │   ├── evaluate.py          # Metrics, SHAP, bias audit
│   │   └── registry.py          # Model promotion logic
│   │
│   ├── serving/
│   │   ├── api.py               # FastAPI application
│   │   ├── schemas.py           # Request/response Pydantic models
│   │   └── predict.py           # Inference helpers
│   │
│   └── monitoring/
│       ├── drift.py             # PSI and KS-test drift detection
│       └── alerts.py            # Alert routing (email/Slack webhook)
│
├── tests/
│   ├── test_features.py
│   ├── test_models.py
│   ├── test_serving.py
│   └── conftest.py
│
├── notebooks/
│   ├── 01_eda.ipynb             # Exploratory data analysis
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_model_evaluation.ipynb
│
├── docs/
│   ├── model_card.md            # Model card (intended use, limitations, fairness)
│   ├── data_card.md             # Data card (source, lineage, licensing)
│   ├── runbook.md               # Operational runbook
│   └── adr/
│       └── 001-algorithm-selection.md  # Architecture Decision Record
│
└── .github/
    └── workflows/
        ├── ci.yml               # Lint + test on every PR
        └── retrain.yml          # Scheduled retraining pipeline
```

---

## Quick Start

### Prerequisites

- Python 3.10+
- Java 11+ (for PySpark)
- MLflow tracking server (local or remote)

### 1. Clone and install

```bash
git clone https://github.com/<your-username>/ml-pipeline-churn.git
cd ml-pipeline-churn
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

### 2. Download the dataset

```bash
# Via Kaggle CLI
kaggle datasets download -d blastchar/telco-customer-churn
unzip telco-customer-churn.zip -d data/raw/

# Or manually: https://www.kaggle.com/datasets/blastchar/telco-customer-churn
```

### 3. Start MLflow tracking server

```bash
mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlruns \
  --host 0.0.0.0 --port 5000
```

### 4. Run the full pipeline

```bash
# Ingest raw data → Delta format
python -m src.data.ingest --input data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv

# Engineer features
python -m src.features.engineering

# Train models (logs to MLflow)
python -m src.models.train

# Evaluate best model + generate reports
python -m src.models.evaluate

# Promote best model to Staging
python -m src.models.registry --action promote --stage Staging

# Launch serving API
uvicorn src.serving.api:app --host 0.0.0.0 --port 8000
```

### 5. Test the endpoint

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "tenure": 24,
    "monthly_charges": 65.5,
    "total_charges": 1572.0,
    "contract": "Month-to-month",
    "internet_service": "Fiber optic",
    "payment_method": "Electronic check",
    "num_products": 3
  }'
```

---

## Running on Databricks

This project is designed for Databricks but runs fully locally for development. To run on Databricks:

1. Import this repo as a **Databricks Repo** (Repos → Add Repo → your fork URL)
2. Create a cluster with DBR 14.x ML runtime
3. Set environment variables in the cluster configuration:
   ```
   MLFLOW_TRACKING_URI=databricks
   FEATURE_STORE_DB=churn_features
   ```
4. Run notebooks in `notebooks/` sequentially, or trigger the pipeline via **Databricks Workflows**

Full Databricks deployment notes are in [`docs/runbook.md`](docs/runbook.md).

---

## Model Performance

| Model | AUC-ROC | F1 (Churn) | Precision | Recall |
|---|---|---|---|---|
| Logistic Regression (baseline) | 0.831 | 0.612 | 0.654 | 0.575 |
| Random Forest | 0.851 | 0.637 | 0.693 | 0.589 |
| **XGBoost (production)** | **0.867** | **0.661** | **0.714** | **0.615** |

*All metrics on 20% hold-out test set. Full evaluation report in `docs/model_card.md`.*

---

## Documentation

| Document | Description |
|---|---|
| [Model Card](docs/model_card.md) | Intended use, performance, fairness evaluation, limitations |
| [Data Card](docs/data_card.md) | Dataset origin, preprocessing steps, licensing, lineage |
| [Runbook](docs/runbook.md) | Deployment, scaling, rollback, on-call guide |
| [ADR 001](docs/adr/001-algorithm-selection.md) | Why XGBoost was selected over alternatives |

---

## CI/CD

Every pull request triggers:
1. `flake8` + `black` linting
2. Unit tests (`pytest tests/`)
3. Integration test: train a mini model and validate MLflow logging

Every Monday at 06:00 UTC, a scheduled workflow:
1. Checks for data drift vs the production training set
2. If PSI > 0.2 on any feature, triggers a full retrain
3. Promotes the new model to Staging if AUC-ROC improves by ≥ 1%

---

## Responsible AI

- **Explainability**: SHAP values computed for every prediction batch; global SHAP summary plots logged as MLflow artifacts
- **Bias audit**: Demographic parity and equalized odds checked across `SeniorCitizen`, `gender`, and `Partner` subgroups
- **Data governance**: Full data lineage from raw CSV → Delta table → Feature Store → model input, tracked in MLflow
- **Model governance**: Every model promotion requires documented justification in the Registry description field

---

## License

MIT — see [LICENSE](LICENSE).

---

## Author

Built as a portfolio project demonstrating production MLOps skills for the **Netcare AI Engineer** role.
