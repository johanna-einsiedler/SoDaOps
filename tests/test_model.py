import os
from unittest.mock import MagicMock, patch

from tweet_sentiment_analysis.model import SentimentPipeline


# Test for `SentimentPipeline` class
@patch("tweet_sentiment_analysis.model.pipeline")
@patch("tweet_sentiment_analysis.model.load_sentiment_pipeline")
def test_sentiment_pipeline_default(mock_load_sentiment_pipeline, mock_pipeline):
    mock_pipeline.return_value = "default_pipeline_instance"
    mock_load_sentiment_pipeline.side_effect = Exception("Mocked exception")


# Test `SentimentPipeline.predict`
@patch("tweet_sentiment_analysis.model.SentimentPipeline")
def test_sentiment_pipeline_predict(mock_pipe):
    mock_pipe.return_value = [{"label": "positive", "score": 0.95}]
    sentiment_pipeline = SentimentPipeline()
    sentiment_pipeline.pipe = mock_pipe

    result = sentiment_pipeline.predict("This is a great day!")
    assert result == [{"label": "positive", "score": 0.95}]
    mock_pipe.assert_called_once_with("This is a great day!")
