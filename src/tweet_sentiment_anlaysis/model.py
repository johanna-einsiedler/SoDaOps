from transformers import pipeline
from loguru import logger
import sys

logger.remove()
logger.add(sys.stdout, level="DEBUG")

class SentimentModel:
    def __init__(self, model_path="cardiffnlp/twitter-roberta-base-sentiment-latest"):
        self.model_path = model_path
        self.pipe = pipeline("sentiment-analysis", model=self.model_path, tokenizer=self.model_path)
        logger.info(f"Model initialized with path: {self.model_path}")

if __name__ == "__main__":
    model = SentimentModel()
