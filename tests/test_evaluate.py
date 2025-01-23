from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from tweet_sentiment_analysis.evaluate import evaluate  # Replace with the actual script/module name
from tweet_sentiment_analysis.model import SentimentPipeline


def test_evaluate():
    train_data = pd.DataFrame({"tweet_text": ["I love coding!", "I hate bugs!"], "sentiment": ["positive", "negative"]})
    val_data = pd.DataFrame({"tweet_text": ["Coding is fun.", "Debugging is annoying."], "sentiment": ["positive", "negative"]})
    test_data = pd.DataFrame({"tweet_text": ["Great job!", "Terrible experience."], "sentiment": ["positive", "negative"]})
    pipe = SentimentPipeline()
    text_input = train["tweet_text"].iloc[0]
    if not isinstance(text_input, str):
        text_input = str(text_input)

    result = pipe.predict(text_input)
