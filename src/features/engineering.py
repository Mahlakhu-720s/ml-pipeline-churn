"""
src/features/engineering.py
----------------------------
Spark-based feature engineering pipeline.

Transforms raw Delta table columns into model-ready features:
- Encodes categorical variables (OHE / label encoding)
- Engineers interaction and aggregate features
- Normalises numeric columns
- Writes the processed feature set to a Feature Store table
- Logs feature metadata and data statistics to MLflow
"""

import logging
from typing import List

import mlflow
import mlflow.spark
from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    OneHotEncoder,
    StandardScaler,
    StringIndexer,
    VectorAssembler,
)
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feature definitions — single source of truth
# ---------------------------------------------------------------------------

NUMERIC_FEATURES: List[str] = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "num_products",          # engineered
    "charge_per_tenure",     # engineered
    "contract_months",       # engineered
]

CATEGORICAL_FEATURES: List[str] = [
    "gender",
    "InternetService",
    "Contract",
    "PaymentMethod",
    "Partner",
    "Dependents",
    "PaperlessBilling",
]

TARGET_COLUMN: str = "Churn"
CUSTOMER_ID_COLUMN: str = "customerID"


# ---------------------------------------------------------------------------
# Feature engineering transformations
# ---------------------------------------------------------------------------

def engineer_features(df: DataFrame) -> DataFrame:
    """
    Apply domain-driven feature engineering.

    New features
    ------------
    num_products : int
        Count of add-on services the customer subscribes to.
        Higher values indicate stickiness and lower churn probability.

    charge_per_tenure : float
        MonthlyCharges / (tenure + 1). Represents cost-per-month-of-loyalty.
        New customers (low tenure) with high charges are high-risk.

    contract_months : int
        Numerical encoding of contract term (Month-to-month=1, One year=12,
        Two year=24) to preserve ordinal relationship for tree models.
    """
    add_on_services = [
        "PhoneService",
        "MultipleLines",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
    ]

    # Count add-on services ("Yes" flags)
    num_products_expr = sum(
        F.when(F.col(svc) == "Yes", 1).otherwise(0) for svc in add_on_services
    )
    df = df.withColumn("num_products", num_products_expr)

    # Cost efficiency ratio
    df = df.withColumn(
        "charge_per_tenure",
        F.col("MonthlyCharges") / (F.col("tenure") + 1),
    )

    # Ordinal contract encoding
    df = df.withColumn(
        "contract_months",
        F.when(F.col("Contract") == "Month-to-month", 1)
        .when(F.col("Contract") == "One year", 12)
        .when(F.col("Contract") == "Two year", 24)
        .otherwise(1),
    )

    return df


def build_preprocessing_pipeline(
    numeric_features: List[str],
    categorical_features: List[str],
) -> Pipeline:
    """
    Build a Spark ML Pipeline for preprocessing.

    Steps
    -----
    1. StringIndexer — label-encodes each categorical column
    2. OneHotEncoder — OHE the indexed columns
    3. VectorAssembler — combines numeric + OHE features into a single vector
    4. StandardScaler — z-score normalisation on the assembled vector

    Returns
    -------
    Pipeline
        Unfitted Spark ML Pipeline ready for .fit(df).
    """
    # Step 1: Index categoricals
    indexers = [
        StringIndexer(
            inputCol=col,
            outputCol=f"{col}_idx",
            handleInvalid="keep",
        )
        for col in categorical_features
    ]

    # Step 2: One-hot encode indexed columns
    encoders = [
        OneHotEncoder(
            inputCol=f"{col}_idx",
            outputCol=f"{col}_ohe",
            dropLast=True,  # avoid dummy variable trap
        )
        for col in categorical_features
    ]

    ohe_feature_cols = [f"{col}_ohe" for col in categorical_features]

    # Step 3: Assemble into feature vector
    assembler = VectorAssembler(
        inputCols=numeric_features + ohe_feature_cols,
        outputCol="features_raw",
        handleInvalid="keep",
    )

    # Step 4: Normalise
    scaler = StandardScaler(
        inputCol="features_raw",
        outputCol="features",
        withMean=True,
        withStd=True,
    )

    return Pipeline(stages=indexers + encoders + [assembler, scaler])


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_feature_engineering(
    delta_input_path: str = "data/delta/churn_raw",
    feature_output_path: str = "data/delta/churn_features",
    mlflow_tracking_uri: str = "sqlite:///mlflow.db",
) -> None:
    """
    End-to-end feature engineering run.

    Reads raw Delta table → engineers features → fits preprocessing pipeline
    → writes feature-engineered data to a new Delta table → logs everything
    to MLflow.
    """
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment("/churn-pipeline/feature-engineering")

    spark = SparkSession.builder.appName("churn-features").getOrCreate()

    with mlflow.start_run(run_name="feature-engineering"):
        logger.info("Loading raw data from %s", delta_input_path)
        df = spark.read.format("delta").load(delta_input_path)

        mlflow.log_param("input_rows", df.count())
        mlflow.log_param("numeric_features", NUMERIC_FEATURES)
        mlflow.log_param("categorical_features", CATEGORICAL_FEATURES)

        # Domain feature engineering
        df = engineer_features(df)
        logger.info("Domain features engineered: num_products, charge_per_tenure, contract_months")

        # Log descriptive statistics as MLflow metrics
        stats = df.select(NUMERIC_FEATURES).summary("mean", "stddev", "min", "max")
        for row in stats.collect():
            stat_name = row["summary"]
            for feat in NUMERIC_FEATURES:
                try:
                    mlflow.log_metric(f"{stat_name}_{feat}", float(row[feat]))
                except (ValueError, TypeError):
                    pass

        # Build and fit preprocessing pipeline
        logger.info("Fitting preprocessing pipeline")
        pipeline = build_preprocessing_pipeline(NUMERIC_FEATURES, CATEGORICAL_FEATURES)
        pipeline_model = pipeline.fit(df)

        # Transform features
        feature_df = pipeline_model.transform(df).select(
            CUSTOMER_ID_COLUMN,
            TARGET_COLUMN,
            "features",
            "_ingested_at",
            *NUMERIC_FEATURES,
            *[f"{c}_idx" for c in CATEGORICAL_FEATURES],
        )

        # Write feature table
        logger.info("Writing feature table to %s", feature_output_path)
        feature_df.write.format("delta").mode("overwrite").option(
            "overwriteSchema", "true"
        ).save(feature_output_path)

        output_rows = feature_df.count()
        mlflow.log_metric("output_rows", output_rows)
        mlflow.log_metric(
            "churn_rate", df.filter(F.col(TARGET_COLUMN) == 1).count() / output_rows
        )

        # Save pipeline model as MLflow artifact for reproducibility
        mlflow.spark.log_model(pipeline_model, artifact_path="preprocessing_pipeline")
        mlflow.set_tag("status", "success")

        logger.info(
            "Feature engineering complete — %d rows written to %s",
            output_rows,
            feature_output_path,
        )


if __name__ == "__main__":
    run_feature_engineering()
