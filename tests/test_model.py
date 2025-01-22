import pytest
from unittest.mock import patch, MagicMock
from tweet_sentiment_analysis.model import SentimentModel


@patch("tweet_sentiment_analysis.model.pipeline")  # Correct the path to the `pipeline` function
def test_model_initialization(mock_pipeline):
    """Test initialization of SentimentModel."""
    # Mock the pipeline
    mock_pipeline.return_value = MagicMock()

    # Instantiate the SentimentModel
    model = SentimentModel()

    # Assert that the pipeline was called with the expected arguments
    mock_pipeline.assert_called_once_with(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        tokenizer="distilbert-base-uncased-finetuned-sst-2-english",
    )

    # Additional assertions
    assert model.model_path == "distilbert-base-uncased-finetuned-sst-2-english"


@patch("tweet_sentiment_analysis.model.pipeline")  # Mock the `pipeline` function
def test_prediction_with_multiple_inputs(mock_pipeline):
    """Test the predict method with multiple inputs."""
    # Mock the pipeline
    mock_pipe = MagicMock()
    mock_pipe.return_value = [
        {"label": "POSITIVE", "score": 0.98},
        {"label": "NEGATIVE", "score": 0.85},
    ]
    mock_pipeline.return_value = mock_pipe

    # Instantiate the SentimentModel
    model = SentimentModel()

    # Test with multiple inputs
    inputs = ["I love this!", "I hate this!"]
    results = model.predict(inputs)

    # Assert the pipeline was called with the expected inputs
    mock_pipe.assert_called_once_with(inputs)

    # Assert the results are as expected
    assert results == [
        {"label": "POSITIVE", "score": 0.98},
        {"label": "NEGATIVE", "score": 0.85},
    ]


