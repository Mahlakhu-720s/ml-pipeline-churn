# Model Card: Customer Churn Prediction

**Model name:** `churn-predictor`  
**Version:** 1.0.0  
**Date:** 2025  
**Status:** Production  
**Owner:** AI Engineering Team  

---

## Model Overview

| Field | Value |
|---|---|
| **Task** | Binary classification — predict whether a customer will churn |
| **Algorithm** | XGBoost (Gradient Boosted Decision Trees) |
| **MLflow model name** | `churn-predictor` |
| **Serving stage** | `Production` |
| **Input** | 11 customer account and service features |
| **Output** | Churn probability (0–1) + binary decision + risk tier |
| **Inference latency** | < 50ms p99 (single prediction) |
| **Retraining cadence** | Weekly drift check; retrain triggered if PSI > 0.2 |

---

## Intended Use

### Primary use case

This model is designed to predict the probability that a customer will cancel their subscription within the next billing period. It is intended to support **retention teams** in prioritising outreach to high-risk customers.

### Downstream actions

Predictions are expected to be used to:
1. Rank customers by churn risk for targeted retention campaigns
2. Trigger automated retention offers for HIGH-risk customers
3. Inform capacity planning for retention team staffing

### Out-of-scope uses

This model should **not** be used for:
- Making decisions that materially affect a customer's access to services or credit
- Inferring sensitive attributes (health, financial risk, etc.) from churn likelihood
- Any use case outside of voluntary customer retention outreach
- Deployment on populations significantly different from the training demographic (see Evaluation)

---

## Training Data

See the companion [Data Card](data_card.md) for full details.

**Dataset:** IBM Telco Customer Churn  
**Source:** Kaggle — [blastchar/telco-customer-churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)  
**Records:** 7,043 customers  
**Positive class (churn):** ~26.5% of records  
**Time period:** Single cross-sectional snapshot (not longitudinal)  
**Preprocessing:** See `src/features/engineering.py`

---

## Features

| Feature | Type | Description | Source |
|---|---|---|---|
| `tenure` | Numeric | Months as a customer | Raw |
| `MonthlyCharges` | Numeric | Current monthly bill | Raw |
| `TotalCharges` | Numeric | Cumulative charges | Raw |
| `num_products` | Numeric | Count of subscribed add-on services | **Engineered** |
| `charge_per_tenure` | Numeric | Monthly cost efficiency ratio | **Engineered** |
| `contract_months` | Numeric | Ordinal contract length (1/12/24) | **Engineered** |
| `gender` | Categorical | Customer gender | Raw |
| `InternetService` | Categorical | DSL / Fiber optic / No | Raw |
| `Contract` | Categorical | Contract term | Raw |
| `PaymentMethod` | Categorical | Payment channel | Raw |
| `Partner` / `Dependents` | Categorical | Household composition | Raw |

---

## Model Performance

All metrics are computed on a **stratified 20% hold-out test set** (1,408 records).

### Overall metrics

| Metric | Value |
|---|---|
| AUC-ROC | 0.867 |
| Average Precision | 0.718 |
| F1 Score (Churn class) | 0.661 |
| Precision (Churn class) | 0.714 |
| Recall (Churn class) | 0.615 |
| Accuracy | 0.806 |

### Model comparison (cross-validated AUC-ROC)

| Model | AUC-ROC | F1 |
|---|---|---|
| Logistic Regression (baseline) | 0.831 | 0.612 |
| Random Forest | 0.851 | 0.637 |
| **XGBoost (selected)** | **0.867** | **0.661** |

### Key drivers (SHAP importance)

1. `contract_months` — Customers on month-to-month contracts are significantly higher risk
2. `tenure` — Recent customers churn at much higher rates
3. `charge_per_tenure` — High cost-to-loyalty ratio is strongly predictive
4. `MonthlyCharges` — Higher bills correlate with churn
5. `InternetService` (Fiber optic) — Fiber customers churn more often than DSL customers

Full SHAP summary plots are logged as MLflow artefacts under the registered model run.

---

## Fairness and Bias Evaluation

The model was audited across three demographic subgroups using two fairness metrics:

**Demographic Parity Gap** — difference in positive prediction rates between groups. A gap > 0.10 triggers a warning.

**AUC-ROC Gap** — difference in predictive performance between groups. A gap > 0.05 triggers a warning.

| Attribute | Groups | PPR Gap | AUC Gap | Warning |
|---|---|---|---|---|
| `gender` | Male / Female | 0.032 | 0.018 | None |
| `SeniorCitizen` | Senior / Non-senior | 0.074 | 0.031 | None |
| `Partner` | Yes / No | 0.051 | 0.022 | None |

**Finding:** No demographic parity gaps exceed the 0.10 threshold. Senior citizens have a notably higher predicted churn rate (PPR gap = 0.074), reflecting a genuine behavioural difference in the data rather than model bias. Monitoring continues weekly.

Full bias audit results are in `evaluation_outputs/bias_audit.json`.

---

## Limitations

1. **Cross-sectional training data.** The model was trained on a single snapshot, not longitudinal data. It cannot model seasonal churn patterns or temporal trends.

2. **No causal inference.** A high churn probability does not mean an intervention will prevent churn. The model identifies correlation, not causation.

3. **26.5% churn rate.** The production population's churn rate may differ from the training set. If the base rate shifts significantly, calibration should be re-evaluated.

4. **Missing features.** Customer satisfaction scores, support ticket history, and recent usage data were not available in the training data. These would likely improve recall.

5. **Geographic specificity.** Training data reflects a North American telecoms company. Direct application to South African healthcare populations (e.g. patient retention) requires domain adaptation and retraining on local data.

---

## Ethical Considerations

- Predictions should be used **in conjunction with** human judgment, not as the sole basis for action.
- Retention offers triggered by this model must be **equally accessible** to all customer segments — no discriminatory exclusions.
- Customers should not be penalised for being predicted as high-churn (e.g. through price increases).
- The model should not be used to make inferences about a customer's financial distress or personal circumstances.

---

## Monitoring and Maintenance

| Activity | Frequency | Owner |
|---|---|---|
| Drift check (PSI + KS) | Weekly (Monday 06:00 UTC) | Automated — GitHub Actions |
| Performance review | Monthly | AI Engineering |
| Bias audit re-run | Monthly or after any retrain | AI Engineering |
| Full model review | Quarterly | AI Engineering + Business |

**Retraining is triggered automatically when:**
- Any numeric feature has PSI > 0.2, **or**
- Any feature has KS-test p-value < 0.05, **or**
- Manual trigger via `workflow_dispatch` in GitHub Actions

---

## Contact

For questions about this model, contact the AI Engineering team or open an issue in this repository.
