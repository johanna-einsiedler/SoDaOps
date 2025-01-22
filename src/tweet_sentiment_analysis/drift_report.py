import os
import sys
from pathlib import Path

import anyio
import pandas as pd
from evidently.metric_preset import TargetDriftPreset, TextEvals
from evidently.report import Report
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from google.cloud import storage
from src.tweet_sentiment_analysis.data import load_data
from src.tweet_sentiment_analysis.model import SentimentModel


from loguru import logger

BUCKET_NAME = "mlops_monitoring"
logger.remove()
logger.add(sys.stdout, level="DEBUG")


def run_analysis(reference_data: pd.DataFrame, current_data: pd.DataFrame) -> None:
    """Run the analysis and return the report."""
    text_overview_report = Report(metrics=[TargetDriftPreset(columns=["target"]), TextEvals(column_name="content")])
    text_overview_report.run(reference_data=reference_data, current_data=current_data)
    report_path = Path("reports/monitoring.html")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Saving HTML report")
    text_overview_report.save_html(str(report_path))


def lifespan(app: FastAPI):
    """Load the data and class names before the application starts."""
    global train, class_names
    train, test, val = load_data()
    train["content"] = train["tweet_text"]
    pipe = SentimentModel()
    def analyze_sentiment(text):
        try:
            text = str(text)
            result = pipe.predict(text[:512])[0]  # Truncate to 512 tokens
            return result["label"], result["score"]
        except Exception:
            return "ERROR", 0
    train["target"] = train["tweet_text"].apply(lambda x: analyze_sentiment(x)[0])
    train["target"] = train["target"].str.lower()
        # Convert to categorical AFTER converting to lowercase AND adding the category
    train["target"] = pd.Categorical(
            train["target"], categories=["negative", "neutral", "positive"]
        )
    #train["target"] = train["sentiment_encoded"]  # in case we wish to look at ground truth drift
    train = train[["content", "target"]]
    logger.debug(train.head())
    class_names = ["negative", "neutral", "positive"]

    yield

    del train, class_names


app = FastAPI(lifespan=lifespan)


def load_latest_files(directory: Path, n: int) -> pd.DataFrame:
    """Load the N latest prediction files from the directory."""
    # Download the latest prediction files from the GCP bucket
    download_files(n=n)
    
    logger.info("Download complete.")

    # Get all prediction files in the directory
    files = directory.glob("predictions*.parquet")

    # Sort files based on when they were created
    files = sorted(files, key=os.path.getmtime)

    # Get the N latest files
    latest_files = files[-n:]

    dataframe = pd.DataFrame()
    for file in latest_files:
        tmp = pd.read_parquet(file)
        dataframe = pd.concat([dataframe, tmp], axis=0)

    dataframe["content"] = dataframe["tweet_text"]
    pipe = SentimentModel()
    def analyze_sentiment(text):
        try:
            text = str(text)
            result = pipe.predict(text[:512])[0]  # Truncate to 512 tokens
            return result["label"], result["score"]
        except Exception:
            return "ERROR", 0
    dataframe["target"] = dataframe["tweet_text"].apply(lambda x: analyze_sentiment(x)[0])
    dataframe["target"] = dataframe["target"].str.lower()
        # Convert to categorical AFTER converting to lowercase AND adding the category
    dataframe["target"] = pd.Categorical(
            dataframe["target"], categories=["negative", "neutral", "positive"]
        )
    #dataframe["target"] = dataframe["sentiment_encoded"] # in case we want to look at ground truth drift
    dataframe = dataframe[["content", "target"]]
    logger.debug(dataframe.head())
    return dataframe


def download_files(n: int = 5) -> None:
    """Download the N latest prediction files from the GCP bucket."""
    logger.info("Downloading latest predictions.")
    bucket = storage.Client().bucket(BUCKET_NAME)
    blobs = list(bucket.list_blobs(prefix="predictions"))
    blobs.sort(key=lambda x: x.updated, reverse=True)
    latest_blobs = blobs[:n]

    processed_dir = Path("data/processed/")
    processed_dir.mkdir(parents=True, exist_ok=True)

    for blob in latest_blobs:
        blob.download_to_filename(processed_dir / blob.name.split("/")[-1])


@app.get("/report", response_class=HTMLResponse)
async def get_report(n: int = 5):
    """Generate and return the report."""
    prediction_data = load_latest_files(Path("data/processed/"), n=n)
    run_analysis(train, prediction_data)

    async with await anyio.open_file("reports/monitoring.html", encoding="utf-8") as f:
        html_content = await f.read()

    return HTMLResponse(content=html_content, status_code=200)
