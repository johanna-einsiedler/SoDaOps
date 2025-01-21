import sys

from loguru import logger
from transformers import pipeline

logger.remove()
logger.add(sys.stdout, level="DEBUG")


class SentimentModel:
    # def __init__(self, model_path="cardiffnlp/twitter-roberta-base-sentiment-latest"):
    def __init__(self, model_path="distilbert-base-uncased-finetuned-sst-2-english"):
        self.model_path = model_path
        self.pipe = pipeline("sentiment-analysis", model=self.model_path, tokenizer=self.model_path)
        logger.info(f"Model initialized with path: {self.model_path}")

    def predict(self, text):
        return self.pipe(text)


if __name__ == "__main__":
    model = SentimentModel()
