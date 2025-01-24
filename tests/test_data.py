import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from tweet_sentiment_analysis.data import (
    clean_text,
    load_data,
    preprocess,
)

# Mock paths for testing
RAW_DATA_PATH = Path("data/raw/")
PROCESSED_DATA_PATH = Path("data/processed/")

# Sample test data
sample_data = pd.DataFrame(
    {
        "tweet_text": [
            "This is a sample tweet! #test http://example.com",
            "Another example tweet, with @mentions and #hashtags.",
        ],
        "party": ["Democrat", "Republican"],
        "sentiment": ["positive", "neutral"],
    }
)


@pytest.fixture
def mock_raw_data_path(tmp_path):
    """Mock the raw data directory with sample parquet files."""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    train_file = raw_dir / "train.parquet"
    test_file = raw_dir / "test.parquet"
    val_file = raw_dir / "val.parquet"

    # Write the sample data to the mock files
    sample_data.to_parquet(train_file, index=False)
    sample_data.to_parquet(test_file, index=False)
    sample_data.to_parquet(val_file, index=False)

    return raw_dir


def test_clean_text():
    """Test the clean_text function."""
    raw_text = "Check this out! #Test http://example.com @user123"
    expected_output = "check this out"
    assert clean_text(raw_text) == expected_output
