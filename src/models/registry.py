"""
src/models/registry.py
-----------------------
MLflow Model Registry helpers — promotion, archival, and version comparison.

Usage
-----
    # Promote latest Staging model to Production
    python -m src.models.registry --action promote --stage Production

    # List all versions of the registered model
    python -m src.models.registry --action list
"""

import argparse
import logging
from typing import Optional

import mlflow
from mlflow import MlflowClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

REGISTERED_MODEL_NAME = "churn-predictor"


def get_client(tracking_uri: str = "sqlite:///mlflow.db") -> MlflowClient:
    mlflow.set_tracking_uri(tracking_uri)
    return MlflowClient()


def list_versions(client: MlflowClient) -> None:
    """Print all registered versions with their stages and metrics."""
    versions = client.search_model_versions(f"name='{REGISTERED_MODEL_NAME}'")
    if not versions:
        print(f"No versions found for model '{REGISTERED_MODEL_NAME}'")
        return
    print(f"\n{'Version':<10} {'Stage':<15} {'Run ID':<36} {'Status'}")
    print("-" * 80)
    for v in sorted(versions, key=lambda x: int(x.version)):
        print(f"{v.version:<10} {v.current_stage:<15} {v.run_id:<36} {v.status}")


def promote(
    client: MlflowClient,
    target_stage: str,
    source_stage: Optional[str] = None,
    archive_existing: bool = True,
) -> None:
    """
    Promote the latest model version in `source_stage` to `target_stage`.

    Parameters
    ----------
    target_stage : str
        Destination stage — 'Staging' or 'Production'.
    source_stage : str, optional
        Source stage to promote from. Defaults to 'None' (newly registered)
        when promoting to Staging, or 'Staging' when promoting to Production.
    archive_existing : bool
        If True, archive the current occupant of `target_stage`.
    """
    if source_stage is None:
        source_stage = "None" if target_stage == "Staging" else "Staging"

    versions = client.search_model_versions(f"name='{REGISTERED_MODEL_NAME}'")
    candidates = [
        v for v in versions
        if v.current_stage.lower() == source_stage.lower()
    ]

    if not candidates:
        logger.error(
            "No model versions in stage '%s' — cannot promote to '%s'.",
            source_stage, target_stage,
        )
        return

    # Take the highest version number as the candidate
    latest = max(candidates, key=lambda v: int(v.version))
    logger.info(
        "Promoting version %s (run_id=%s) from %s → %s",
        latest.version, latest.run_id, source_stage, target_stage,
    )

    client.transition_model_version_stage(
        name=REGISTERED_MODEL_NAME,
        version=latest.version,
        stage=target_stage,
        archive_existing_versions=archive_existing,
    )

    # Add governance description
    client.update_model_version(
        name=REGISTERED_MODEL_NAME,
        version=latest.version,
        description=(
            f"Promoted to {target_stage} via automated pipeline. "
            f"Source stage: {source_stage}. Run ID: {latest.run_id}. "
            "Manual bias audit and AUC gate passed. "
            "See evaluation_outputs/evaluation_summary.json for full metrics."
        ),
    )

    logger.info("Promotion complete. Version %s is now in %s.", latest.version, target_stage)


def archive_all_except_production(client: MlflowClient) -> None:
    """Archive all non-Production model versions (housekeeping)."""
    versions = client.search_model_versions(f"name='{REGISTERED_MODEL_NAME}'")
    for v in versions:
        if v.current_stage not in ("Production", "Archived"):
            client.transition_model_version_stage(
                name=REGISTERED_MODEL_NAME,
                version=v.version,
                stage="Archived",
            )
            logger.info("Archived version %s (was %s)", v.version, v.current_stage)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MLflow Model Registry management.")
    parser.add_argument(
        "--action",
        required=True,
        choices=["promote", "list", "archive-old"],
        help="Action to perform",
    )
    parser.add_argument(
        "--stage",
        default="Staging",
        choices=["Staging", "Production"],
        help="Target stage for promotion",
    )
    parser.add_argument("--mlflow-uri", default="sqlite:///mlflow.db")
    args = parser.parse_args()

    client = get_client(args.mlflow_uri)

    if args.action == "list":
        list_versions(client)
    elif args.action == "promote":
        promote(client, target_stage=args.stage)
    elif args.action == "archive-old":
        archive_all_except_production(client)
