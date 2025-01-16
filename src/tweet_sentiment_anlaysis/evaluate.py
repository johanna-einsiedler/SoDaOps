from loguru import logger
import pandas as pd
from data import preprocess
from model import SentimentModel
from sklearn.metrics import f1_score
import typer
from pathlib import Path


def evaluate() -> None:
    "Evaluating model performance"
    process_path = Path("data/processed/")
    train_path = process_path / "train.csv"
    test_path = process_path / "test.csv"
    val_path = process_path / "val.csv"

    # Check if files exist; if not, preprocess them
    if not train_path.exists() or not test_path.exists() or not val_path.exists():
        logger.info("Files not found. Preprocessing dataset.")
        preprocess()
    
    train = pd.read_csv("data/processed/train.csv")
    test = pd.read_csv("data/processed/test.csv")
    val = pd.read_csv("data/processed/val.csv")
    pipe= SentimentModel()

    text_input = train['tweet_text'].iloc[0]
    if not isinstance(text_input, str):
        text_input = str(text_input)

    result = pipe.predict(text_input)
    logger.debug(f'Check if model can produce results: {result}')

    # Function to perform sentiment analysis using the pipeline
    def analyze_sentiment(text):
        try:
            text = str(text)
            result = pipe.predict(text[:512])[0]  # Truncate to 512 tokens
            return result['label'], result['score']
        except Exception as e:
            return 'ERROR', 0

    val['predicted_sentiment'] = val['tweet_text'].apply(lambda x: analyze_sentiment(x)[0])
    # Convert predicted sentiment to lowercase
    val['predicted_sentiment'] = val['predicted_sentiment'].str.lower()
    # Convert to categorical AFTER converting to lowercase AND adding the category
    val['predicted_sentiment'] = pd.Categorical(val['predicted_sentiment'], categories=['negative', 'neutral', 'positive'])

    logger.info(f1_score(val['sentiment'], val['predicted_sentiment'], average="macro"))
    
if __name__ == "__main__":
    typer.run(evaluate)
    