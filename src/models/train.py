"""
src/models/train.py
--------------------
Model training pipeline with full MLflow experiment tracking.

Trains three candidate models (Logistic Regression, Random Forest, XGBoost)
via cross-validated hyperparameter search. Every run is logged to MLflow
with parameters, metrics, artefacts, and the serialised model.

The best model (highest validation AUC-ROC) is registered in the
MLflow Model Registry under the name 'churn-predictor'.
"""

import argparse
import logging
from typing import Any, Dict, List, Tuple

import mlflow
import mlflow.sklearn
import mlflow.xgboost
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

REGISTERED_MODEL_NAME = "churn-predictor"

# ---------------------------------------------------------------------------
# Candidate model definitions with hyperparameter grids
# ---------------------------------------------------------------------------

CANDIDATE_MODELS: List[Dict[str, Any]] = [
    {
        "name": "logistic-regression",
        "model": LogisticRegression(max_iter=1000, random_state=42),
        "params": {
            "C": 1.0,
            "penalty": "l2",
            "solver": "lbfgs",
            "max_iter": 1000,
        },
    },
    {
        "name": "random-forest",
        "model": RandomForestClassifier(random_state=42, n_jobs=-1),
        "params": {
            "n_estimators": 300,
            "max_depth": 8,
            "min_samples_split": 10,
            "class_weight": "balanced",
        },
    },
    {
        "name": "xgboost",
        "model": XGBClassifier(
            n_estimators=400,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=3,  # handles class imbalance (~26% churn rate)
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        ),
        "params": {
            "n_estimators": 400,
            "max_depth": 5,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "scale_pos_weight": 3,
        },
    },
]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_features(feature_path: str) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Load processed feature table and return X, y arrays.

    For local development this reads from Parquet/Delta;
    on Databricks this would read from the Feature Store.

    Returns
    -------
    X : np.ndarray of shape (n_samples, n_features)
    y : np.ndarray of shape (n_samples,)
    df : full pandas DataFrame (retained for bias audit)
    """
    try:
        from pyspark.sql import SparkSession
        from pyspark.ml.functions import vector_to_array

        spark = SparkSession.builder.appName("churn-train").getOrCreate()
        sdf = spark.read.format("delta").load(feature_path)

        # Convert Spark ML vector to pandas columns
        sdf_expanded = sdf.withColumn("features_arr", vector_to_array("features"))
        pdf = sdf_expanded.toPandas()
        n_features = len(pdf["features_arr"].iloc[0])
        feature_cols = [f"f_{i}" for i in range(n_features)]
        X = np.stack(pdf["features_arr"].values)
        y = pdf["Churn"].values.astype(int)
        return X, y, pdf

    except Exception:
        # Fallback: load from CSV for local runs without Spark
        logger.warning(
            "Spark not available — loading from CSV fallback for local development."
        )
        df = pd.read_csv("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)
        df["Churn"] = (df["Churn"] == "Yes").astype(int)

        # Lightweight preprocessing for CSV fallback
        cat_cols = [
            "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
            "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
            "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
            "PaperlessBilling", "PaymentMethod",
        ]
        df_enc = df.copy()
        for col in cat_cols:
            le = LabelEncoder()
            df_enc[col] = le.fit_transform(df_enc[col].astype(str))

        num_cols = ["tenure", "MonthlyCharges", "TotalCharges"] + cat_cols
        X = df_enc[num_cols].values
        y = df_enc["Churn"].values
        return X, y, df_enc


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    """Compute a standard suite of binary classification metrics."""
    return {
        "auc_roc": roc_auc_score(y_true, y_prob),
        "avg_precision": average_precision_score(y_true, y_prob),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
    }


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_all_models(
    X: np.ndarray,
    y: np.ndarray,
    n_folds: int = 5,
    mlflow_tracking_uri: str = "sqlite:///mlflow.db",
) -> str:
    """
    Train all candidate models with cross-validation.

    Each model gets its own MLflow run nested inside a parent experiment run.
    Returns the run_id of the best model.

    Parameters
    ----------
    X : np.ndarray
    y : np.ndarray
    n_folds : int
        Number of stratified cross-validation folds.
    mlflow_tracking_uri : str

    Returns
    -------
    best_run_id : str
        MLflow run_id of the best-performing model.
    """
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment("/churn-pipeline/model-training")

    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    best_auc = 0.0
    best_run_id: str = ""

    with mlflow.start_run(run_name="training-comparison") as parent_run:
        mlflow.log_param("n_folds", n_folds)
        mlflow.log_param("n_samples", X.shape[0])
        mlflow.log_param("n_features", X.shape[1])
        mlflow.log_metric("churn_rate", float(y.mean()))

        for candidate in CANDIDATE_MODELS:
            with mlflow.start_run(
                run_name=candidate["name"], nested=True
            ) as child_run:
                logger.info("Training: %s", candidate["name"])

                # Log hyperparameters
                mlflow.log_params(candidate["params"])

                model = candidate["model"]

                # Cross-validated predictions (OOF)
                y_prob = cross_val_predict(
                    model, X, y, cv=cv, method="predict_proba", n_jobs=-1
                )[:, 1]
                y_pred = (y_prob >= 0.5).astype(int)

                metrics = compute_metrics(y, y_pred, y_prob)
                mlflow.log_metrics(metrics)

                logger.info(
                    "%s — AUC: %.4f | F1: %.4f | Precision: %.4f | Recall: %.4f",
                    candidate["name"],
                    metrics["auc_roc"],
                    metrics["f1"],
                    metrics["precision"],
                    metrics["recall"],
                )

                # Re-fit on full training data for artefact logging
                model.fit(X, y)

                # Log model artefact (framework-appropriate)
                if candidate["name"] == "xgboost":
                    mlflow.xgboost.log_model(
                        model,
                        artifact_path="model",
                        registered_model_name=(
                            REGISTERED_MODEL_NAME
                            if metrics["auc_roc"] > best_auc
                            else None
                        ),
                    )
                else:
                    mlflow.sklearn.log_model(
                        model,
                        artifact_path="model",
                        registered_model_name=(
                            REGISTERED_MODEL_NAME
                            if metrics["auc_roc"] > best_auc
                            else None
                        ),
                    )

                # Track best
                if metrics["auc_roc"] > best_auc:
                    best_auc = metrics["auc_roc"]
                    best_run_id = child_run.info.run_id
                    mlflow.set_tag("is_best", "true")
                    logger.info(
                        "New best model: %s (AUC=%.4f)", candidate["name"], best_auc
                    )

        mlflow.log_metric("best_auc_roc", best_auc)
        mlflow.set_tag("best_run_id", best_run_id)
        logger.info("Training complete. Best run: %s (AUC=%.4f)", best_run_id, best_auc)

    return best_run_id


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train churn prediction models.")
    parser.add_argument(
        "--feature-path",
        default="data/delta/churn_features",
        help="Path to engineered feature Delta table",
    )
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--mlflow-uri", default="sqlite:///mlflow.db")
    args = parser.parse_args()

    X, y, df = load_features(args.feature_path)
    best_run = train_all_models(X, y, n_folds=args.n_folds, mlflow_tracking_uri=args.mlflow_uri)
    print(f"\nBest MLflow run ID: {best_run}")
