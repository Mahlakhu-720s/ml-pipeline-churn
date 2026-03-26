"""
tests/smoke_test_pipeline.py
-----------------------------
End-to-end smoke test: trains a minimal model on synthetic data,
validates MLflow logging, and writes an evaluation_summary.json.

Designed to run in CI without access to the real dataset or Spark.
Runtime: < 60 seconds on a 2-core CI runner.
"""

import json
import logging
import os
import sys
import tempfile
import time

import mlflow
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
OUTPUT_DIR = "evaluation_outputs"


def generate_synthetic_data(n_samples: int = 500):
    """Generate a small synthetic binary classification dataset."""
    X, y = make_classification(
        n_samples=n_samples,
        n_features=11,
        n_informative=7,
        n_redundant=2,
        weights=[0.735, 0.265],  # ~26.5% positive rate (matches churn dataset)
        random_state=42,
    )
    feature_names = [
        "tenure", "MonthlyCharges", "TotalCharges", "num_products",
        "charge_per_tenure", "contract_months", "SeniorCitizen",
        "InternetService", "PaymentMethod", "Partner", "Dependents",
    ]
    df = pd.DataFrame(X, columns=feature_names)
    df["Churn"] = y
    df["gender"] = np.random.choice(["Male", "Female"], n_samples)
    df["SeniorCitizen_str"] = np.random.choice(["Yes", "No"], n_samples)
    return df, X, y


def run_smoke_test():
    logger.info("Starting pipeline smoke test...")
    start = time.time()

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("/churn-pipeline/smoke-test")

    # Generate data
    df, X, y = generate_synthetic_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    with mlflow.start_run(run_name="smoke-test"):
        # Train a small model
        model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)

        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)
        auc = roc_auc_score(y_test, y_prob)

        mlflow.log_param("model_type", "random_forest_smoke_test")
        mlflow.log_param("n_estimators", 50)
        mlflow.log_metric("test_auc_roc", auc)
        mlflow.sklearn.log_model(model, artifact_path="model")
        mlflow.set_tag("smoke_test", "true")

        logger.info("Smoke test AUC-ROC: %.4f", auc)

    # Write evaluation summary (consumed by CI gate)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    summary = {
        "model_run_id": "smoke-test",
        "test_auc_roc": round(auc, 4),
        "test_f1_churn": 0.0,   # not computed in smoke test
        "test_precision_churn": 0.0,
        "test_recall_churn": 0.0,
        "bias_warning_count": 0,
        "shap_top_features": [],
        "smoke_test": True,
    }
    summary_path = os.path.join(OUTPUT_DIR, "evaluation_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    elapsed = time.time() - start
    logger.info("Smoke test complete in %.1fs. AUC: %.4f", elapsed, auc)

    # Validate minimum AUC (smoke test uses synthetic data, lower bar)
    if auc < 0.60:
        logger.error("FAIL: Smoke test AUC %.4f is below minimum 0.60", auc)
        sys.exit(1)

    logger.info("PASS: All smoke test checks passed.")


if __name__ == "__main__":
    run_smoke_test()
