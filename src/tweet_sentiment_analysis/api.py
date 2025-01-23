import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from loguru import logger
from pydantic import BaseModel

from tweet_sentiment_analysis.model import SentimentPipeline
from tweet_sentiment_analysis.utils import upload_to_gcs


def make_filter(name):
    def filter(record):
        return record["extra"].get("name") == name

    return filter


logger.add("logs/api_logs.log", rotation="10 MB", retention="10 days")
logger.add(
    "logs/csv_logs.csv",
    format="{time:YYYY-MM-DD HH:mm:ss},{message}",
    level="INFO",
    rotation="10 MB",
    filter=make_filter("csv_logs"),
)
csv_logger = logger.bind(name="csv_logs")


class ReviewInput(BaseModel):
    """Define input data structure for the endpoint."""

    review: str


class PredictionOutput(BaseModel):
    """Define output data structure for the endpoint."""

    label: str
    score: float


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model and tokenizer when the app starts and clean up when the app stops."""
    logger.info("Attempting to load model and tokenizer")
    global pipeline
    pipeline = SentimentPipeline()

    logger.info("Model and tokenizer loaded successfully")

    yield

    bucket_name = "sentiment-output-dtu"
    try:
        upload_to_gcs(Path("logs/csv_logs.csv"), bucket_name, f"logs/csv_logs_{time.strftime('%Y%m%d-%H%M%S')}.csv")
        upload_to_gcs(Path("logs/api_logs.log"), bucket_name, f"logs/api_logs_{time.strftime('%Y%m%d-%H%M%S')}.log")
    except Exception as e:
        logger.error(f"Failed to upload logs to GCS: {e}")


# Initialize FastAPI app
app = FastAPI(lifespan=lifespan)


@app.get("/")
def read_root():
    return "This is a tweet sentiment classification API designer for political tweets. Go to /docs to test out online."


# Prediction endpoint
@app.post("/predict", response_model=PredictionOutput)
async def get_predict_sentiment(review_input: ReviewInput):
    """Predict sentiment of the input text."""
    try:
        result = pipeline.predict(review_input.review)[0]
        label, score = result["label"], result["score"]
        csv_logger.info(f"{label},{score},{review_input.review}")
        return PredictionOutput(label=label, score=score)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


# Add a GET endpoint for browser-based queries
@app.get("/predict", response_model=PredictionOutput)
async def get_predict_sentiment_browser(review: str):
    """
    Predict sentiment of the input text provided via a query parameter.
    This allows for direct browser-based access.
    Args:
        review (str): The review text passed as a query parameter.
    """
    try:
        result = pipeline.predict(review)[0]
        label, score = result["label"], result["score"]

        csv_logger.info(f"{label},{score},{review}")
        return PredictionOutput(label=label, score=score)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
