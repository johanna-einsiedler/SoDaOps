import os
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from google.cloud import storage
from loguru import logger
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

import wandb

logger.remove()
logger.add(sys.stdout, level="DEBUG")


def upload_to_gcs(local_path: Path, bucket_name: str, destination_blob_name: str):
    """Uploads a file or directory to a GCP bucket."""
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    if local_path.is_dir():
        for file_path in local_path.glob("**/*"):
            if file_path.is_file():
                relative_path = file_path.relative_to(local_path)
                blob = bucket.blob(f"{destination_blob_name}/{relative_path}")
                blob.upload_from_filename(str(file_path))
                logger.info(f"Uploaded {file_path} to gs://{bucket_name}/{destination_blob_name}/{relative_path}")
    else:
        blob = bucket.blob(destination_blob_name)
        blob.upload_from_filename(str(local_path))
        logger.info(f"Uploaded {local_path} to gs://{bucket_name}/{destination_blob_name}")


def get_best_model_artifact(entity_name: str, project_name: str, sweep_name: str) -> str:
    """Fetches the best model artifact based on the lowest eval_loss and saves it to the GCP bucket."""
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

        # Save the artifact to the GCP bucket with a timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        bucket_name = "sentiment-output-dtu"
        destination_blob_name = f"model_{timestamp}"
        upload_to_gcs(Path(artifact_path), bucket_name, destination_blob_name)

    return f"gs://{bucket_name}/{destination_blob_name}"


def load_sentiment_pipeline():
    """Loads the model and tokenizer from the best artifact."""
    logger.info("Loading sentiment analysis pipeline")

    load_dotenv()
    wandb_project = os.getenv("WANDB_PROJECT")
    wandb_entity = os.getenv("WANDB_ENTITY")
    wandb_sweep_name = os.getenv("WANDB_SWEEP_NAME")
    max_length = int(os.getenv("max_length", 78))
    # Fetch the best model artifact
    model_dir = get_best_model_artifact(
        entity_name=wandb_entity, project_name=wandb_project, sweep_name=wandb_sweep_name
    )
    if not model_dir:
        logger.error("Failed to load model artifact")
        sys.exit(1)

    # Load both the model and tokenizer from the artifact directory
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    logger.info("Model and tokenizer loaded successfully")

    # Initialize the sentiment analysis pipeline
    sentiment_pipeline = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        truncation=True,
        padding="max_length",
        max_length=max_length,
    )
    logger.info("Sentiment analysis pipeline initialized")
    return sentiment_pipeline


class SentimentPipeline:
    def __init__(self):
        load_dotenv()
        max_length = int(os.getenv("max_length", 78))
        try:
            self.pipe = load_sentiment_pipeline()
        except Exception:
            model_path = "cardiffnlp/twitter-roberta-base-sentiment-latest"
            self.model_path = model_path
            self.pipe = pipeline(
                "text-classification",
                model=self.model_path,
                tokenizer=self.model_path,
                truncation=True,
                padding="max_length",
                max_length=max_length,
            )
            logger.warning(f"Best model not retrieved, default used: {self.model_path}")

    def predict(self, text):
        return self.pipe(text)


if __name__ == "__main__":
    # Set up logging
    logger.remove()
    LOG_LEVEL = "INFO"
    logger.add(sys.stderr, level=LOG_LEVEL)
    logger.add("logs/inference_logs.log", level=LOG_LEVEL, rotation="10 MB", retention="10 days")

    pipeline = SentimentPipeline()

    # Analyze sample text
    text_to_analyze = "This movie was fantastic! I loved it!"
    output = pipeline.predict(text_to_analyze)
    logger.info(f"Text: {text_to_analyze}")
    logger.info(f"Sentiment: {output[0]['label']}, Score: {output[0]['score']:.4f}")
