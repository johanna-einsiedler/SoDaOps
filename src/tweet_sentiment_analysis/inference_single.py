import sys

import wandb
from loguru import logger
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline


def get_best_model_artifact(entity_name: str, project_name: str, sweep_name: str):
    """Fetches the best model artifact based on the lowest eval_loss."""
    logger.info("Fetching the best model artifact")
    api = wandb.Api()
    sweep = api.sweep(f"{entity_name}/{project_name}/{sweep_name}")

    # Filter out only successful runs
    successful_runs = [run for run in sweep.runs if run.state == "finished"]
    if not successful_runs:
        logger.error("No successful runs found for the specified sweep")
        return None

    sorted_runs = sorted(successful_runs, key=lambda run: run.summary.get("eval_loss", float("inf")))
    best_run = sorted_runs[0]
    logger.info(f"Best run selected: {best_run.name} with eval_loss={best_run.summary.get('eval_loss')}")

    for artifact in best_run.logged_artifacts():
        artifact_path = artifact.download()
        logger.info(f"Downloaded artifact to: {artifact_path}")

    return artifact_path


def load_sentiment_pipeline():
    """Loads the model and tokenizer from the best artifact."""
    logger.info("Loading sentiment analysis pipeline")

    # TODO: Avoid hardcoding of these variables. Save in train.py scipt as env variable?
    # TODO: Add loading of local artifact if already loaded to speed up inference
    model_dir = get_best_model_artifact(
        entity_name="advanced-deep-learning-course", project_name="sentiment-analysis", sweep_name="n29e92vy"
    )
    if not model_dir:
        logger.error("Failed to load model artifact")
        sys.exit(1)

    # Load both the model and tokenizer from the artifact directory
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    logger.info("Model and tokenizer loaded successfully")

    # Initialize the sentiment analysis pipeline
    sentiment_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
    logger.info("Sentiment analysis pipeline initialized")
    return sentiment_pipeline


def analyze_sentiment(text: str):
    """Analyzes the sentiment of the given text."""
    logger.info(f"Analyzing sentiment for input text: {text[:30]}...")  # Log truncated version of the input
    sentiment_pipeline = load_sentiment_pipeline()

    # TODO: Convert this into using the same preprocess function that training uses to ensure consistent data preprocessing
    result = sentiment_pipeline(text[:16])[0]
    label, score = result["label"], result["score"]
    logger.info(f"Sentiment analysis result: label={label}, score={score:.4f}")
    return label, score


if __name__ == "__main__":
    # Set up logging
    logger.remove()
    LOG_LEVEL = "INFO"
    logger.add(sys.stderr, level=LOG_LEVEL)
    logger.add("logs/inference_logs.log", level=LOG_LEVEL, rotation="10 MB", retention="10 days")

    # Analyze sample text
    text_to_analyze = "This movie was fantastic! I loved it!"
    label, score = analyze_sentiment(text_to_analyze)
    print(f"Sentiment: {label}, Score: {score:.4f}")
