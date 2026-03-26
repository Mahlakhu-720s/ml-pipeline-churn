# ADR 001: Algorithm Selection — XGBoost over Alternatives

**Status:** Accepted  
**Date:** 2025  
**Decision maker:** AI Engineering Team  
**Reviewers:** Data Science Lead, ML Engineering Lead  

---

## Context

The churn prediction pipeline requires a binary classifier that meets these requirements:

1. AUC-ROC ≥ 0.85 on the hold-out test set
2. Prediction latency < 100ms p99 for online serving
3. Feature importance must be explainable to non-technical stakeholders (SHAP)
4. Must handle class imbalance (~26.5% positive rate) without manual oversampling
5. Must be loggable and servable via MLflow with a standard `predict_proba` interface

---

## Decision

We selected **XGBoost (Gradient Boosted Decision Trees)** as the production model.

---

## Alternatives Considered

### Option A: Logistic Regression

**Pros:**
- Highly interpretable — coefficients map directly to feature influence
- Fast training and inference
- Native probability calibration

**Cons:**
- AUC-ROC 0.831 — below the 0.85 target
- Requires manual feature interaction terms to capture non-linear relationships
- Sensitive to feature scaling (requires StandardScaler, which adds a pipeline step)

**Verdict:** Retained as a baseline and for interpretability comparisons; not selected as production model.

### Option B: Random Forest

**Pros:**
- Strong AUC-ROC (0.851) — above the threshold
- Robust to outliers and missing values
- Supports SHAP via `TreeExplainer`

**Cons:**
- Slightly lower AUC-ROC than XGBoost (0.851 vs 0.867)
- Larger model size (300 trees × full depth) — slower inference than XGBoost
- No native handling of class imbalance (`class_weight='balanced'` adds overhead)

**Verdict:** Strong alternative. Selected XGBoost due to higher AUC and better class imbalance handling.

### Option C: XGBoost (selected)

**Pros:**
- Highest AUC-ROC (0.867) across all candidates
- `scale_pos_weight` natively handles class imbalance without resampling
- Gradient boosting captures non-linear interactions without manual feature engineering
- Excellent SHAP support via `TreeExplainer` (exact values, not approximations)
- Widely adopted in tabular ML benchmarks — strong community and documentation
- Serialises cleanly with `mlflow.xgboost.log_model`
- Inference is fast: < 5ms per prediction on CPU

**Cons:**
- More hyperparameters than Logistic Regression (risk of overfitting without CV)
- Less interpretable than Logistic Regression at the coefficient level
- Requires careful `n_estimators` and `learning_rate` tuning

**Verdict:** Selected. Meets all five requirements.

### Option D: Neural Network (MLP)

**Pros:**
- Can theoretically learn complex representations

**Cons:**
- No clear performance advantage over XGBoost on structured/tabular data
- Requires GPU for efficient training
- SHAP values are approximations only (KernelExplainer), which is slower and noisier
- Harder to debug and explain to clinical or business stakeholders
- Does not meet the < 100ms latency requirement without model compression

**Verdict:** Not suitable for this use case at current scale. May be reconsidered if unstructured data (clinical notes, images) is incorporated.

---

## Consequences

- The preprocessing pipeline must normalise inputs (handled by Spark ML `StandardScaler`) even though XGBoost is tree-based, because other models in the comparison require it and we want a unified pipeline.
- SHAP `TreeExplainer` is the primary explainability tool — this is a constraint on future model changes (neural networks would require a different explainer).
- If the model is replaced in future (e.g. with a neural network for multimodal inputs), this ADR must be superseded and a new bias audit completed.

---

## Review trigger

This decision should be re-evaluated if:
- A new algorithm achieves > 0.90 AUC on the test set
- Unstructured data (e.g. clinical notes) is added to the feature set
- Inference latency requirements tighten below 10ms p99
