import os
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

import wandb
from tweet_sentiment_analysis.utils import upload_to_gcs

logger.remove()
logger.add(sys.stdout, level="DEBUG")


def get_best_model_artifact(entity_name: str, project_name: str, sweep_name: str) -> str:
    """Fetches the best model artifact from W&B and uploads it to GCS."""
    logger.info("Fetching the best model artifact")
    api = wandb.Api()
    sweep = api.sweep(f"{entity_name}/{project_name}/{sweep_name}")

    # Filter out only successful runs
    successful_runs = [run for run in sweep.runs if run.state == "finished"]
    if not successful_runs:
        logger.error("No successful runs found for the specified sweep")
        return None

    sorted_runs = sorted(successful_runs, key=lambda run: run.summary.get("eval/loss", float("inf")))
    best_run = sorted_runs[0]
    logger.info(f"Best run selected: {best_run.name} with eval_loss={best_run.summary.get('eval/loss')}")

    for artifact in best_run.logged_artifacts():
        artifact_path = artifact.download()
        logger.info(f"Downloaded artifact to: {artifact_path}")

        # Save the artifact to the GCS bucket under 'model/<timestamp>'
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        bucket_name = "sentiment-output-dtu"
        destination_blob_name = f"models/{timestamp}"  # Correctly create a 'model/<timestamp>' folder structure
        upload_to_gcs(Path(artifact_path), bucket_name, destination_blob_name)

    return f"gs://{bucket_name}/{destination_blob_name}"


if __name__ == "__main__":
    load_dotenv()
    wandb_project = os.getenv("WANDB_PROJECT")
    wandb_entity = os.getenv("WANDB_ENTITY")
    wandb_sweep_name = os.getenv("WANDB_SWEEP_NAME")

    get_best_model_artifact(wandb_entity, wandb_project, wandb_sweep_name)
