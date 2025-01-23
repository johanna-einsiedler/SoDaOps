from pathlib import Path
from google.cloud import storage

import pandas as pd
import typer
from loguru import logger
from sklearn.metrics import f1_score

from tweet_sentiment_analysis.data import load_data, preprocess
from tweet_sentiment_analysis.model import SentimentPipeline


def evaluate(use_test_set: bool = False) -> None:
    "Evaluating model performance"
    train, test, val = load_data()


    # train = pd.read_parquet("../../data/processed/train.parquet")
    # test = pd.read_parquet("../../data/processed/test.parquet")
    # val = pd.read_parquet("../../data/processed/val.parquet")
    pipe = SentimentPipeline()

    text_input = train["tweet_text"].iloc[0]
    if not isinstance(text_input, str):
        text_input = str(text_input)
 
    result = pipe.predict(text_input)
    logger.debug(f"Check if model can produce results: {result}")

    val["predicted_sentiment"] = val["tweet_text"].apply(lambda x: pipe.predict(x)[0]["label"])

    if use_test_set:
        test["predicted_sentiment"] = test["tweet_text"].apply(lambda x: pipe.predict(x)[0]["label"])
        # Convert predicted sentiment to lowercase
        test["predicted_sentiment"] = test["predicted_sentiment"].str.lower()
        # Convert to categorical AFTER converting to lowercase AND adding the category
        test["predicted_sentiment"] = pd.Categorical(
            test["predicted_sentiment"], categories=["negative", "neutral", "positive"]
        )

        logger.info(f1_score(test["sentiment"], test["predicted_sentiment"], average="macro"))

    else:
        # Convert predicted sentiment to lowercase
        val["predicted_sentiment"] = val["predicted_sentiment"].str.lower()
        # Convert to categorical AFTER converting to lowercase AND adding the category
        val["predicted_sentiment"] = pd.Categorical(
            val["predicted_sentiment"], categories=["negative", "neutral", "positive"]
        )
        f1 = f1_score(val["sentiment"], val["predicted_sentiment"], average="macro")
        logger.info(f1)
        # save to cloud 
        # (local_path: Path, bucket_name: str, destination_blob_name: str):
        bucket_name = "sentiment-output-dtu"
        destination_blob_name = "evaluation/evaluate-output.txt"
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)
        blob.upload_from_string("F1-Score: "+str(f1))



# def upload_to_gcs(local_path: Path, bucket_name: str, destination_blob_name: str):
#     """Uploads a file or directory to a GCP bucket."""
#     client = storage.Client()
#     bucket = client.bucket(bucket_name)

#     if local_path.is_dir():
#         for file_path in local_path.glob("**/*"):
#             if file_path.is_file():
#                 relative_path = file_path.relative_to(local_path)
#                 blob = bucket.blob(f"{destination_blob_name}/{relative_path}")
#                 blob.upload_from_filename(str(file_path))
#                 logger.info(f"Uploaded {file_path} to gs://{bucket_name}/{destination_blob_name}/{relative_path}")
#     else:
#         blob = bucket.blob(destination_blob_name)
#         blob.upload_from_filename(str(local_path))
#         logger.info(f"Uploaded {local_path} to gs://{bucket_name}/{destination_blob_name}")


if __name__ == "__main__":
    typer.run(evaluate)
