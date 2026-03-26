"""
src/data/ingest.py
------------------
Raw data ingestion pipeline: CSV → Delta Lake table.

Responsibilities:
- Load raw CSV from local path or cloud storage
- Apply schema enforcement and type casting
- Write to Delta Lake in append or overwrite mode
- Log ingestion metadata to MLflow for lineage tracking
"""

import argparse
import logging
from pathlib import Path

import mlflow
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    DoubleType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema definition
# ---------------------------------------------------------------------------

RAW_SCHEMA = StructType(
    [
        StructField("customerID", StringType(), nullable=False),
        StructField("gender", StringType(), nullable=True),
        StructField("SeniorCitizen", IntegerType(), nullable=True),
        StructField("Partner", StringType(), nullable=True),
        StructField("Dependents", StringType(), nullable=True),
        StructField("tenure", IntegerType(), nullable=True),
        StructField("PhoneService", StringType(), nullable=True),
        StructField("MultipleLines", StringType(), nullable=True),
        StructField("InternetService", StringType(), nullable=True),
        StructField("OnlineSecurity", StringType(), nullable=True),
        StructField("OnlineBackup", StringType(), nullable=True),
        StructField("DeviceProtection", StringType(), nullable=True),
        StructField("TechSupport", StringType(), nullable=True),
        StructField("StreamingTV", StringType(), nullable=True),
        StructField("StreamingMovies", StringType(), nullable=True),
        StructField("Contract", StringType(), nullable=True),
        StructField("PaperlessBilling", StringType(), nullable=True),
        StructField("PaymentMethod", StringType(), nullable=True),
        StructField("MonthlyCharges", DoubleType(), nullable=True),
        StructField("TotalCharges", StringType(), nullable=True),  # has whitespace in raw
        StructField("Churn", StringType(), nullable=True),
    ]
)


# ---------------------------------------------------------------------------
# Ingestion logic
# ---------------------------------------------------------------------------

def get_spark() -> SparkSession:
    """Return an active SparkSession (local dev) or the running Databricks session."""
    return (
        SparkSession.builder.appName("churn-ingestion")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config(
            "spark.sql.catalog.spark_catalog",
            "org.apache.spark.sql.delta.catalog.DeltaCatalog",
        )
        .getOrCreate()
    )


def clean_total_charges(df):
    """
    TotalCharges arrives as a string with occasional whitespace for new customers.
    Cast to double, defaulting nulls to 0.0.
    """
    return df.withColumn(
        "TotalCharges",
        F.when(F.trim(F.col("TotalCharges")) == "", F.lit(0.0))
        .otherwise(F.col("TotalCharges").cast(DoubleType())),
    )


def encode_target(df):
    """Encode Churn: 'Yes' → 1, 'No' → 0."""
    return df.withColumn(
        "Churn",
        F.when(F.col("Churn") == "Yes", 1).otherwise(0).cast(IntegerType()),
    )


def add_ingestion_metadata(df):
    """Append ingestion timestamp and a monotonically increasing row ID."""
    return df.withColumn("_ingested_at", F.current_timestamp()).withColumn(
        "_row_id", F.monotonically_increasing_id()
    )


def ingest(
    input_path: str,
    delta_output_path: str = "data/delta/churn_raw",
    mode: str = "overwrite",
    mlflow_tracking_uri: str = "sqlite:///mlflow.db",
) -> None:
    """
    Full ingestion pipeline.

    Parameters
    ----------
    input_path : str
        Path to the raw CSV file.
    delta_output_path : str
        Target Delta table path (local or DBFS).
    mode : str
        Write mode — 'overwrite' for initial load, 'append' for incremental.
    mlflow_tracking_uri : str
        MLflow tracking server URI for lineage logging.
    """
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment("/churn-pipeline/ingestion")

    spark = get_spark()
    logger.info("Reading raw CSV: %s", input_path)

    with mlflow.start_run(run_name="data-ingestion"):
        # Log provenance
        mlflow.log_param("input_path", input_path)
        mlflow.log_param("delta_output_path", delta_output_path)
        mlflow.log_param("write_mode", mode)

        # Read with enforced schema
        raw_df = spark.read.csv(input_path, header=True, schema=RAW_SCHEMA)
        row_count_raw = raw_df.count()
        logger.info("Rows read: %d", row_count_raw)
        mlflow.log_metric("rows_raw", row_count_raw)

        # Transformations
        df = (
            raw_df.pipe(clean_total_charges)
            .pipe(encode_target)
            .pipe(add_ingestion_metadata)
        )

        # Data quality: count nulls in critical columns
        null_counts = {
            col: df.filter(F.col(col).isNull()).count()
            for col in ["customerID", "tenure", "MonthlyCharges", "TotalCharges", "Churn"]
        }
        for col, null_count in null_counts.items():
            mlflow.log_metric(f"nulls_{col}", null_count)
            if null_count > 0:
                logger.warning("Column '%s' has %d null values", col, null_count)

        # Write to Delta
        df.write.format("delta").mode(mode).save(delta_output_path)
        row_count_final = df.count()
        mlflow.log_metric("rows_written", row_count_final)
        mlflow.set_tag("status", "success")

        logger.info(
            "Ingestion complete — %d rows written to %s", row_count_final, delta_output_path
        )


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest raw churn CSV into Delta Lake.")
    parser.add_argument("--input", required=True, help="Path to raw CSV file")
    parser.add_argument("--output", default="data/delta/churn_raw", help="Delta output path")
    parser.add_argument("--mode", default="overwrite", choices=["overwrite", "append"])
    parser.add_argument("--mlflow-uri", default="sqlite:///mlflow.db")
    args = parser.parse_args()

    ingest(
        input_path=args.input,
        delta_output_path=args.output,
        mode=args.mode,
        mlflow_tracking_uri=args.mlflow_uri,
    )
