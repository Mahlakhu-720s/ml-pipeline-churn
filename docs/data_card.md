# Data Card: IBM Telco Customer Churn Dataset

**Last updated:** 2025  
**Maintained by:** AI Engineering Team  

---

## Dataset Summary

| Field | Value |
|---|---|
| **Name** | IBM Sample Data — Telco Customer Churn |
| **Version used** | Static snapshot (Kaggle, as of 2024) |
| **Records** | 7,043 customers |
| **Features (raw)** | 21 columns |
| **Target** | `Churn` (binary: Yes / No) |
| **Positive class rate** | 26.54% (1,869 churned customers) |
| **Storage format** | CSV (raw) → Delta Lake (processed) |
| **License** | [IBM Community License](https://community.ibm.com/community/user/businessanalytics/blogs/steven-macko/2019/07/11/telco-customer-churn-1113) — publicly available for research and demonstration |

---

## Source and Provenance

**Original source:** IBM Developer — Watson Analytics sample datasets  
**Access:** [Kaggle dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)  
**Citation:**  
> IBM. (2019). Telco Customer Churn. IBM Developer. Redistributed via Kaggle by blastchar.

This dataset is a **simulated** dataset created by IBM for analytics education. It does not contain real customer data.

---

## Data Schema

### Raw columns

| Column | Type | Description | Notes |
|---|---|---|---|
| `customerID` | String | Unique customer identifier | Not used as a feature |
| `gender` | Categorical | Male / Female | |
| `SeniorCitizen` | Integer | 1 = senior citizen, 0 = not | Note: binary integer, not string |
| `Partner` | Categorical | Yes / No | |
| `Dependents` | Categorical | Yes / No | |
| `tenure` | Integer | Months with the company | Range: 0–72 |
| `PhoneService` | Categorical | Yes / No | |
| `MultipleLines` | Categorical | Yes / No / No phone service | |
| `InternetService` | Categorical | DSL / Fiber optic / No | |
| `OnlineSecurity` | Categorical | Yes / No / No internet service | |
| `OnlineBackup` | Categorical | Yes / No / No internet service | |
| `DeviceProtection` | Categorical | Yes / No / No internet service | |
| `TechSupport` | Categorical | Yes / No / No internet service | |
| `StreamingTV` | Categorical | Yes / No / No internet service | |
| `StreamingMovies` | Categorical | Yes / No / No internet service | |
| `Contract` | Categorical | Month-to-month / One year / Two year | |
| `PaperlessBilling` | Categorical | Yes / No | |
| `PaymentMethod` | Categorical | 4 payment types | See notes |
| `MonthlyCharges` | Float | Current monthly bill (USD) | Range: ~18–119 |
| `TotalCharges` | String | Cumulative charges | **11 records have whitespace — treated as 0.0** |
| `Churn` | Categorical | Yes / No | Target variable |

### Known data quality issues

| Issue | Records affected | Resolution |
|---|---|---|
| `TotalCharges` whitespace (new customers with 0 tenure) | 11 | Cast to 0.0 (see `src/data/ingest.py`) |
| Class imbalance (26.5% churn) | Entire dataset | `scale_pos_weight=3` in XGBoost |

---

## Preprocessing Steps

All transformations are version-controlled in `src/data/ingest.py` and `src/features/engineering.py`. Steps are applied in this order:

1. **Schema enforcement** — Spark `StructType` schema applied at read time to catch type violations early.
2. **TotalCharges cleaning** — Whitespace values cast to `0.0`.
3. **Target encoding** — `Churn`: `'Yes'` → `1`, `'No'` → `0`.
4. **Ingestion metadata** — `_ingested_at` timestamp and `_row_id` appended for lineage.
5. **Feature engineering** — Three derived features added:
   - `num_products`: count of subscribed add-on services
   - `charge_per_tenure`: `MonthlyCharges / (tenure + 1)`
   - `contract_months`: ordinal contract duration (1 / 12 / 24)
6. **Categorical encoding** — `StringIndexer` + `OneHotEncoder` via Spark ML Pipeline.
7. **Numeric normalisation** — `StandardScaler` (zero mean, unit variance).

The fitted preprocessing pipeline is logged as an MLflow artefact (`preprocessing_pipeline`) alongside every training run, ensuring full reproducibility.

---

## Data Lineage

```
Kaggle CSV (raw)
     │
     ▼ [src/data/ingest.py]
Delta Table: data/delta/churn_raw
  - Schema enforced
  - TotalCharges cleaned
  - Target encoded
  - _ingested_at, _row_id appended
     │
     ▼ [src/features/engineering.py]
Delta Table: data/delta/churn_features
  - Domain features added
  - Categorical encoding applied
  - Numeric normalisation applied
  - Spark ML Pipeline fitted and logged
     │
     ▼ [src/models/train.py]
MLflow Experiment Runs (training snapshots)
     │
     ▼ [MLflow Model Registry]
churn-predictor (Staging → Production)
```

All steps are tracked in MLflow with run IDs linking each artefact to its lineage.

---

## Splits

| Split | Size | Method |
|---|---|---|
| Training | 80% (5,634) | Stratified split (maintains 26.5% churn rate) |
| Test | 20% (1,408) | Held out — not used during training or hyperparameter tuning |
| Cross-validation | 5-fold (within training set) | Stratified K-Fold |

---

## Licensing and Legal

- **License:** IBM Community License — permitted for research, education, and demonstration purposes.
- **PII:** The dataset is simulated. No real customer names, addresses, or contact information are present.
- **POPIA / GDPR:** Not applicable — dataset is entirely synthetic.
- **Usage restrictions:** This data must not be presented as real customer data in any commercial product.

---

## Relevance to Healthcare (Netcare Context)

While this dataset originates from telecoms, the pipeline architecture and methodology are directly transferable to:

| Healthcare use case | Analogous telco concept |
|---|---|
| Patient no-show prediction | Customer churn |
| Readmission risk | Service cancellation risk |
| Length-of-stay modelling | Contract duration |
| Appointment engagement | Product add-on uptake |

The feature engineering, Spark pipeline, MLflow lifecycle, and bias audit patterns are identical regardless of domain. Domain adaptation requires: re-specifying `NUMERIC_FEATURES` and `CATEGORICAL_FEATURES` in `configs/feature_config.yaml` and retraining on clinical data.
