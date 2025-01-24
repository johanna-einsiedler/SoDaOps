import time
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, HTTPException
from loguru import logger
from pydantic import BaseModel

from tweet_sentiment_analysis.model import SentimentPipeline
from tweet_sentiment_analysis.utils import upload_to_gcs

# FastAPI application setup
app = FastAPI()

# Sentiment pipeline initialization
pipeline = SentimentPipeline()


# Function to create log file name based on startup time
def generate_log_filename():
    # Get the current timestamp and format it
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"logs/api_logs_{timestamp}.log"


# Generate dynamic log filename at startup
log_filename = generate_log_filename()

# Initialize the logger with the dynamic filename
logger.add(log_filename, rotation="10 MB", retention="10 days")
csv_logger = logger.bind(name="csv_logs")
csv_logger.add("logs/csv_logs.csv", format="{time:YYYY-MM-DD HH:mm:ss},{message}", level="INFO", rotation="10 MB")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model and tokenizer when the app starts and clean up when the app stops."""
    logger.info("Attempting to load model and tokenizer")
    global pipeline
    pipeline = SentimentPipeline()

    logger.info("Model and tokenizer loaded successfully")

    yield


# Initialize FastAPI app
app = FastAPI(lifespan=lifespan)


# Background task function to upload logs to GCS
def upload_logs_to_gcs(bucket_name: str):
    try:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        upload_to_gcs(Path("logs/csv_logs.csv"), bucket_name, f"logs/csv_logs_{timestamp}.csv")
        upload_to_gcs(Path(log_filename), bucket_name, f"logs/api_logs_{timestamp}.log")
    except Exception as e:
        logger.error(f"Failed to upload logs to GCS: {e}")


# Review input structure
class ReviewInput(BaseModel):
    review: str


# Prediction output structure
class PredictionOutput(BaseModel):
    label: str
    score: float


# Prediction endpoint
@app.post("/predict", response_model=PredictionOutput)
async def get_predict_sentiment(review_input: ReviewInput, background_tasks: BackgroundTasks):
    """Predict sentiment of the input text and upload logs in the background."""
    try:
        result = pipeline.predict(review_input.review)[0]
        label, score = result["label"], result["score"]

        # Log the result
        csv_logger.info(f"{label},{score},{review_input.review}")
        logger.info(f"Predicted sentiment: {label} with score: {score}")
        # Add the log upload task to background
        bucket_name = "sentiment-output-dtu"  # GCS bucket name
        background_tasks.add_task(upload_logs_to_gcs, bucket_name)

        return PredictionOutput(label=label, score=score)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


# Add a GET endpoint for browser-based queries
@app.get("/predict", response_model=PredictionOutput)
async def get_predict_sentiment_browser(review: str, background_tasks: BackgroundTasks):
    """Predict sentiment of the input text provided via a query parameter."""
    try:
        result = pipeline.predict(review)[0]
        label, score = result["label"], result["score"]

        # Log the result
        csv_logger.info(f"{label},{score},{review}")
        logger.info(f"Predicted sentiment: {label} with score: {score}")

        # Add the log upload task to background
        bucket_name = "sentiment-output-dtu"  # GCS bucket name
        background_tasks.add_task(upload_logs_to_gcs, bucket_name)

        return PredictionOutput(label=label, score=score)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
