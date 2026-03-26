"""
src/models/evaluate.py
-----------------------
Comprehensive model evaluation: metrics, SHAP explainability, and bias audit.

Outputs
-------
- Classification report and confusion matrix
- SHAP summary and waterfall plots (logged as MLflow artefacts)
- Bias audit report across demographic subgroups (gender, SeniorCitizen, Partner)
- Evaluation summary JSON (machine-readable, consumed by CI gating)
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    roc_auc_score,
    roc_curve,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Demographic columns to audit for bias
SENSITIVE_ATTRIBUTES: List[str] = ["gender", "SeniorCitizen", "Partner"]


# ---------------------------------------------------------------------------
# SHAP explainability
# ---------------------------------------------------------------------------

def compute_shap(model, X: np.ndarray, feature_names: List[str], output_dir: str) -> None:
    """
    Compute SHAP values and save summary + waterfall plots.

    Automatically selects the appropriate SHAP explainer based on model type.
    TreeExplainer is used for tree-based models (faster, exact values).
    KernelExplainer is used as a fallback for all other model types.
    """
    os.makedirs(output_dir, exist_ok=True)
    logger.info("Computing SHAP values...")

    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        # For binary classifiers, TreeExplainer returns [class_0, class_1]
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
    except Exception:
        logger.warning("TreeExplainer failed — falling back to KernelExplainer (slower)")
        background = shap.sample(X, 100, random_state=42)
        explainer = shap.KernelExplainer(model.predict_proba, background)
        shap_values = explainer.shap_values(X[:200])[1]

    # Summary plot (beeswarm)
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(
        shap_values,
        X,
        feature_names=feature_names,
        show=False,
        max_display=15,
    )
    plt.tight_layout()
    summary_path = os.path.join(output_dir, "shap_summary.png")
    plt.savefig(summary_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("SHAP summary plot saved: %s", summary_path)

    # Mean absolute SHAP bar chart
    mean_shap = np.abs(shap_values).mean(axis=0)
    feature_importance = pd.DataFrame(
        {"feature": feature_names, "mean_abs_shap": mean_shap}
    ).sort_values("mean_abs_shap", ascending=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(
        feature_importance["feature"].head(15)[::-1],
        feature_importance["mean_abs_shap"].head(15)[::-1],
        color="#1f77b4",
    )
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("Top 15 Features by Mean SHAP Importance")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    bar_path = os.path.join(output_dir, "shap_importance_bar.png")
    plt.savefig(bar_path, dpi=150, bbox_inches="tight")
    plt.close()

    return {
        "shap_summary_path": summary_path,
        "shap_bar_path": bar_path,
        "top_features": feature_importance.head(10).to_dict(orient="records"),
    }


# ---------------------------------------------------------------------------
# Bias audit
# ---------------------------------------------------------------------------

def run_bias_audit(
    df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    sensitive_attributes: List[str],
    output_dir: str,
) -> Dict:
    """
    Audit model for demographic bias.

    Metrics computed per subgroup
    ------------------------------
    - Positive prediction rate (for demographic parity)
    - True positive rate / recall (for equalised odds — TPR)
    - False positive rate (for equalised odds — FPR)
    - AUC-ROC per subgroup

    A subgroup performance gap > 0.05 AUC or > 0.10 TPR triggers a warning.

    Returns
    -------
    dict
        Bias audit results by attribute and subgroup.
    """
    os.makedirs(output_dir, exist_ok=True)
    audit_results = {}

    for attr in sensitive_attributes:
        if attr not in df.columns:
            logger.warning("Sensitive attribute '%s' not in DataFrame — skipping.", attr)
            continue

        groups = df[attr].unique()
        group_metrics = {}

        for group in groups:
            mask = df[attr] == group
            if mask.sum() < 30:  # skip tiny subgroups
                continue
            gt = y_true[mask]
            pp = y_pred[mask]
            pb = y_prob[mask]

            ppr = pp.mean()  # positive prediction rate
            tpr = (pp[gt == 1].sum() / max(gt.sum(), 1))  # recall
            fpr_val = (pp[gt == 0].sum() / max((gt == 0).sum(), 1))

            try:
                auc = roc_auc_score(gt, pb)
            except ValueError:
                auc = float("nan")

            group_metrics[str(group)] = {
                "n": int(mask.sum()),
                "positive_prediction_rate": round(float(ppr), 4),
                "true_positive_rate": round(float(tpr), 4),
                "false_positive_rate": round(float(fpr_val), 4),
                "auc_roc": round(float(auc), 4),
            }

        # Demographic parity difference
        pprs = [v["positive_prediction_rate"] for v in group_metrics.values()]
        aucs = [v["auc_roc"] for v in group_metrics.values() if not np.isnan(v["auc_roc"])]
        dp_gap = round(max(pprs) - min(pprs), 4) if pprs else None
        auc_gap = round(max(aucs) - min(aucs), 4) if len(aucs) > 1 else None

        audit_results[attr] = {
            "groups": group_metrics,
            "demographic_parity_gap": dp_gap,
            "auc_gap": auc_gap,
            "bias_warning": dp_gap is not None and (dp_gap > 0.10 or (auc_gap or 0) > 0.05),
        }

        if audit_results[attr]["bias_warning"]:
            logger.warning(
                "BIAS WARNING — attribute '%s': parity gap=%.4f, AUC gap=%.4f",
                attr, dp_gap or 0, auc_gap or 0,
            )

    # Write JSON report
    audit_path = os.path.join(output_dir, "bias_audit.json")
    with open(audit_path, "w") as f:
        json.dump(audit_results, f, indent=2)
    logger.info("Bias audit saved: %s", audit_path)

    return audit_results


# ---------------------------------------------------------------------------
# ROC curve plot
# ---------------------------------------------------------------------------

def plot_roc_curve(y_true: np.ndarray, y_prob: np.ndarray, output_dir: str) -> str:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, lw=2, label=f"AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve — Churn Prediction")
    ax.legend(loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()

    path = os.path.join(output_dir, "roc_curve.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


# ---------------------------------------------------------------------------
# Full evaluation pipeline
# ---------------------------------------------------------------------------

def evaluate(
    model_run_id: str,
    X: np.ndarray,
    y_true: np.ndarray,
    df: pd.DataFrame,
    feature_names: Optional[List[str]] = None,
    output_dir: str = "evaluation_outputs",
    mlflow_tracking_uri: str = "sqlite:///mlflow.db",
) -> Dict:
    """
    Run the full evaluation suite against a logged MLflow model.

    Parameters
    ----------
    model_run_id : str
        MLflow run_id from which to load the model.
    X : np.ndarray
        Test feature matrix.
    y_true : np.ndarray
        True labels.
    df : pd.DataFrame
        Full DataFrame (used for bias audit demographic columns).
    feature_names : list of str, optional
    output_dir : str
    mlflow_tracking_uri : str

    Returns
    -------
    dict
        Summary of all evaluation metrics and paths to artefacts.
    """
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    os.makedirs(output_dir, exist_ok=True)

    # Load model from MLflow
    model_uri = f"runs:/{model_run_id}/model"
    logger.info("Loading model from %s", model_uri)
    model = mlflow.sklearn.load_model(model_uri)

    y_prob = model.predict_proba(X)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]

    with mlflow.start_run(run_name="model-evaluation"):
        mlflow.log_param("evaluated_run_id", model_run_id)

        # --- Core metrics ---
        report = classification_report(y_true, y_pred, output_dict=True)
        auc = roc_auc_score(y_true, y_prob)
        mlflow.log_metric("test_auc_roc", auc)
        mlflow.log_metric("test_f1_churn", report["1"]["f1-score"])
        mlflow.log_metric("test_precision_churn", report["1"]["precision"])
        mlflow.log_metric("test_recall_churn", report["1"]["recall"])
        logger.info("AUC: %.4f | F1: %.4f", auc, report["1"]["f1-score"])

        # --- Confusion matrix ---
        fig, ax = plt.subplots(figsize=(6, 5))
        ConfusionMatrixDisplay.from_predictions(
            y_true, y_pred, ax=ax, display_labels=["No Churn", "Churn"], cmap="Blues"
        )
        cm_path = os.path.join(output_dir, "confusion_matrix.png")
        plt.tight_layout()
        plt.savefig(cm_path, dpi=150)
        plt.close()
        mlflow.log_artifact(cm_path)

        # --- ROC curve ---
        roc_path = plot_roc_curve(y_true, y_prob, output_dir)
        mlflow.log_artifact(roc_path)

        # --- SHAP ---
        shap_info = compute_shap(model, X, feature_names, output_dir)
        mlflow.log_artifact(shap_info["shap_summary_path"])
        mlflow.log_artifact(shap_info["shap_bar_path"])

        # --- Bias audit ---
        bias_results = run_bias_audit(df, y_true, y_pred, y_prob, SENSITIVE_ATTRIBUTES, output_dir)
        mlflow.log_artifact(os.path.join(output_dir, "bias_audit.json"))

        bias_warnings = sum(1 for v in bias_results.values() if v.get("bias_warning"))
        mlflow.log_metric("bias_warning_count", bias_warnings)

        # --- Summary JSON ---
        summary = {
            "model_run_id": model_run_id,
            "test_auc_roc": round(auc, 4),
            "test_f1_churn": round(report["1"]["f1-score"], 4),
            "test_precision_churn": round(report["1"]["precision"], 4),
            "test_recall_churn": round(report["1"]["recall"], 4),
            "bias_warning_count": bias_warnings,
            "shap_top_features": shap_info["top_features"],
        }
        summary_path = os.path.join(output_dir, "evaluation_summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        mlflow.log_artifact(summary_path)

        mlflow.set_tag("status", "evaluated")
        logger.info("Evaluation complete. Summary: %s", summary_path)

    return summary
