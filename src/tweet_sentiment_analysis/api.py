from contextlib import asynccontextmanager

import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BertModel, BertTokenizer, pipeline

from tweet_sentiment_analysis.model import SentimentPipeline


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
    global pipeline
    pipeline = SentimentPipeline()

    print("Model and tokenizer loaded successfully")

    yield

    # del model, tokenizer


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
        return PredictionOutput(label=label, score=score)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
