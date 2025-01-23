import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from google.cloud import storage
from loguru import logger
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

from tweet_sentiment_analysis.utils import download_from_gcs

logger.remove()
logger.add(sys.stdout, level="DEBUG")


def get_latest_model_timestamp(bucket_name: str, prefix: str = "models/") -> str:
    """
    Gets the latest timestamped model folder in the specified GCS bucket under the given prefix.

    Args:
        bucket_name (str): Name of the GCS bucket.
        prefix (str): Prefix for the model folders (e.g., 'model/').

    Returns:
        str: The latest timestamp folder name.
    """
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blobs = list(bucket.list_blobs(prefix=prefix))

    # Extract folder names directly under the prefix (e.g., 'model/20250123_153000/')
    timestamps = [
        blob.name[len(prefix) :].split("/")[0]
        for blob in blobs
        if blob.name[len(prefix) :].strip() and "/" in blob.name[len(prefix) :]
    ]

    if not timestamps:
        logger.error("No model folders found in the GCS bucket.")
        raise FileNotFoundError("No model folders found in the GCS bucket.")

    # Sort timestamps and return the latest one
    latest_timestamp = sorted(timestamps)[-1]
    logger.info(f"Latest model timestamp found: {latest_timestamp}")
    return latest_timestamp


class SentimentPipeline:
    def __init__(self, timestamp: str | None = None):
        """
        Initialize the SentimentPipeline.

        Args:
            timestamp (str | None): The timestamp for the model folder. If None, fetches the latest model timestamp.
        """
        load_dotenv()
        max_length = int(os.getenv("max_length"))
        bucket_name = "sentiment-output-dtu"
        local_model_dir = Path("downloaded_model")

        try:
            # Get the latest model timestamp from GCS
            # Determine the model timestamp to use
            if timestamp is None:
                logger.info("Fetching the latest model timestamp from GCS")
                timestamp = get_latest_model_timestamp(bucket_name)
            else:
                logger.info(f"Using provided timestamp: {timestamp}")

            # Define the local path for the model
            timestamp_model_dir = local_model_dir / timestamp

            # Check if the model folder already exists locally
            if timestamp_model_dir.exists():
                logger.info(f"Model folder already exists locally: {timestamp_model_dir}")
            else:
                # Download the model from GCS
                logger.info(f"Downloading the model for timestamp {timestamp} from GCS")
                model_folder_name = f"models/{timestamp}"
                download_from_gcs(bucket_name, model_folder_name, str(timestamp_model_dir))

            # Load model and tokenizer from the local directory
            self.pipe = pipeline(
                "text-classification",
                model=AutoModelForSequenceClassification.from_pretrained(timestamp_model_dir),
                tokenizer=AutoTokenizer.from_pretrained(timestamp_model_dir),
                truncation=True,
                padding="max_length",
                max_length=max_length,
            )
            logger.info("Model pipeline loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model from GCS: {e}")
            model_path = "cardiffnlp/twitter-roberta-base-sentiment-latest"
            self.pipe = pipeline(
                "text-classification",
                model=model_path,
                tokenizer=model_path,
                truncation=True,
                padding="max_length",
                max_length=max_length,
            )
            logger.warning(f"Using default model: {model_path}")

    def predict(self, text: str):
        return self.pipe(text)


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="INFO")

    pipeline = SentimentPipeline()
    text_to_analyze = "This movie was fantastic! I loved it!"
    output = pipeline.predict(text_to_analyze)
    logger.info(f"Text: {text_to_analyze}")
    logger.info(f"Sentiment: {output[0]['label']}, Score: {output[0]['score']:.4f}")
