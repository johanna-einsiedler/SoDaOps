from pathlib import Path

import pandas as pd
import typer
from loguru import logger
from model import SentimentModel
from sklearn.metrics import f1_score

from data import load_data, preprocess


def evaluate(use_test_set: bool = False) -> None:
    "Evaluating model performance"
    train, test, val = load_data()
    pipe = SentimentModel()

    text_input = train["tweet_text"].iloc[0]
    if not isinstance(text_input, str):
        text_input = str(text_input)

    result = pipe.predict(text_input)
    logger.debug(f"Check if model can produce results: {result}")

    # Function to perform sentiment analysis using the pipeline
    def analyze_sentiment(text):
        try:
            text = str(text)
            result = pipe.predict(text[:512])[0]  # Truncate to 512 tokens
            return result["label"], result["score"]
        except Exception:
            return "ERROR", 0

    val["predicted_sentiment"] = val["tweet_text"].apply(lambda x: analyze_sentiment(x)[0])
    if use_test_set:
        test["predicted_sentiment"] = test["tweet_text"].apply(lambda x: analyze_sentiment(x)[0])
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

        logger.info(f1_score(val["sentiment"], val["predicted_sentiment"], average="macro"))


if __name__ == "__main__":
    typer.run(evaluate)
