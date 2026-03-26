# Operational Runbook: Churn Prediction Pipeline

**Version:** 1.0  
**Audience:** MLOps engineers, on-call data scientists  
**Last reviewed:** 2025  

---

## 1. Service Overview

| Component | Technology | Purpose |
|---|---|---|
| Feature pipeline | PySpark + Delta Lake | Data ingestion and feature engineering |
| Experiment tracking | MLflow | Parameter, metric, and artefact logging |
| Model registry | MLflow Model Registry | Version control and promotion gates |
| Serving API | FastAPI + Uvicorn | REST endpoint for real-time predictions |
| Monitoring | Custom PSI/KS + MLflow | Drift detection and alerting |
| CI/CD | GitHub Actions | Automated testing and retraining |

**SLA targets:**
- API p99 latency: < 100ms
- API availability: 99.5%
- Retraining cycle: weekly drift check, retrain within 24h of trigger

---

## 2. Deployment Guide

### 2.1 Local development

```bash
# Clone and install
git clone https://github.com/<org>/ml-pipeline-churn.git
cd ml-pipeline-churn
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .

# Start MLflow
mlflow server --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlruns --port 5000

# Run pipeline (assumes dataset in data/raw/)
python -m src.data.ingest --input data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv
python -m src.features.engineering
python -m src.models.train
python -m src.models.evaluate

# Start API
uvicorn src.serving.api:app --host 0.0.0.0 --port 8000
```

### 2.2 Databricks deployment

1. Connect this GitHub repo as a **Databricks Repo**:  
   `Workspace → Repos → Add Repo → paste repo URL`

2. Create a cluster with **DBR 14.3 ML** runtime (includes PySpark, MLflow, Delta)

3. Set cluster environment variables:
   ```
   MLFLOW_TRACKING_URI=databricks
   FEATURE_STORE_DB=churn_features
   JWT_SECRET=<your-secret-from-key-vault>
   ```

4. Create a **Databricks Workflow** with these tasks in order:
   - Task 1: `src/data/ingest.py`
   - Task 2: `src/features/engineering.py` (depends on Task 1)
   - Task 3: `src/models/train.py` (depends on Task 2)
   - Task 4: `src/models/evaluate.py` (depends on Task 3)

5. Deploy the serving endpoint:
   - Open MLflow Model Registry → `churn-predictor` → latest Staging version
   - Click **Deploy** → **Databricks Model Serving**
   - Set compute: 1 CPU node, auto-scaling 0–2

### 2.3 Docker deployment (standalone)

```bash
docker build -t churn-api:latest .
docker run -p 8000:8000 \
  -e MLFLOW_TRACKING_URI=http://mlflow-server:5000 \
  -e JWT_SECRET=<secret> \
  churn-api:latest
```

---

## 3. Configuration Reference

### `configs/model_config.yaml`

```yaml
xgboost:
  n_estimators: 400
  max_depth: 5
  learning_rate: 0.05
  subsample: 0.8
  colsample_bytree: 0.8
  scale_pos_weight: 3
  random_state: 42

training:
  n_folds: 5
  test_size: 0.2
  random_state: 42
  min_auc_gate: 0.80
```

### `configs/serving_config.yaml`

```yaml
model:
  name: churn-predictor
  stage: Production

api:
  host: "0.0.0.0"
  port: 8000
  max_request_size_kb: 100

monitoring:
  psi_severe_threshold: 0.20
  psi_moderate_threshold: 0.10
  ks_p_value_threshold: 0.05
```

---

## 4. Generating API Tokens (Development)

For local development, generate a JWT token using:

```python
from jose import jwt
import time

payload = {
    "sub": "dev-user",
    "role": "engineer",
    "exp": int(time.time()) + 86400,  # 24h expiry
}
token = jwt.encode(payload, "dev-secret-change-in-production", algorithm="HS256")
print(token)
```

Then use as a Bearer token:
```bash
curl -X POST http://localhost:8000/predict \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{"tenure": 12, "monthly_charges": 70.0, "total_charges": 840.0, ...}'
```

---

## 5. Monitoring and Alerts

### 5.1 Health check

```bash
curl http://localhost:8000/health
# Expected: {"status": "ok", "model_loaded": true, ...}
```

### 5.2 Metrics endpoint

```bash
curl http://localhost:8000/metrics
# Returns: total_requests, error_rate, avg_latency_ms
```

### 5.3 Drift monitoring

The scheduled GitHub Actions workflow (`retrain.yml`) runs every Monday at 06:00 UTC.

To run drift detection manually:
```bash
python -m src.monitoring.drift \
  --reference data/reference/training_snapshot.csv \
  --current data/production/recent_30_days.csv \
  --output monitoring/drift_report.json
```

**Drift thresholds:**

| PSI value | Severity | Action |
|---|---|---|
| < 0.10 | None | Continue monitoring |
| 0.10 – 0.20 | Moderate | Log warning, increase monitoring frequency |
| > 0.20 | Severe | **Trigger retraining** |

---

## 6. Model Promotion Procedure

Models flow through three stages in the MLflow Registry: **None → Staging → Production**.

### Promoting to Staging (automated)

Happens automatically after a successful training run if AUC-ROC ≥ current + 0.01.

### Promoting Staging → Production (manual gate)

Requires human sign-off. Steps:
1. Review the evaluation report in `evaluation_outputs/evaluation_summary.json`
2. Review the bias audit in `evaluation_outputs/bias_audit.json` — no bias warnings outstanding
3. In MLflow UI: `churn-predictor → Staging → Transition to → Production`
4. Add a **description** justifying the promotion (required governance step):
   ```
   Promoted by: <your name>
   Date: YYYY-MM-DD
   Reason: AUC improved from 0.867 to 0.881 after drift-triggered retrain.
   Bias audit: No demographic parity gaps above threshold.
   ```
5. Restart the serving API (or redeploy) to load the new Production model.

### Rolling back Production

```bash
# In MLflow UI: transition old version back to Production
# OR via Python:
from mlflow import MlflowClient
client = MlflowClient()
client.transition_model_version_stage(
    name="churn-predictor",
    version="<old-version>",
    stage="Production",
)
```

---

## 7. Canary Deployment

For gradual rollouts, use the `canary_weight` environment variable:

```
CANARY_MODEL_VERSION=<new-version>
CANARY_WEIGHT=0.10  # 10% of traffic to new model
```

The serving API will route `CANARY_WEIGHT` fraction of requests to the canary model and compare latency and error rates over 24 hours before full promotion.

---

## 8. Troubleshooting

### API returns 503 on /predict

**Cause:** Model failed to load at startup.  
**Check:** Container logs for MLflow connection errors.  
**Fix:** Verify `MLFLOW_TRACKING_URI` is reachable and `churn-predictor/Production` exists.

### AUC drops > 5% in monitoring

**Cause:** Concept drift — the relationship between features and churn has changed.  
**Action:** Trigger immediate retraining via `workflow_dispatch` on `retrain.yml`.

### PSI > 0.2 but retrain didn't trigger

**Check:** GitHub Actions `retrain.yml` run logs.  
**Common cause:** The reference snapshot in `data/reference/` is stale.  
**Fix:** Refresh the reference snapshot with the most recent training data.

### OOM during feature engineering

**Cause:** Insufficient Spark executor memory.  
**Fix:** Increase `spark.executor.memory` in `configs/spark_config.yaml` or scale up the cluster.

---

## 9. On-Call Escalation

| Severity | Condition | First response | Escalate to |
|---|---|---|---|
| P1 | API down (503 > 5 min) | Restart serving process; rollback model | Platform engineering |
| P2 | Error rate > 5% | Check input validation logs; inspect recent requests | AI Engineering lead |
| P3 | Drift detected | Verify retraining workflow triggered | AI Engineering |
| P4 | Bias warning in weekly audit | Schedule model review within 5 business days | AI Engineering + Ethics |
