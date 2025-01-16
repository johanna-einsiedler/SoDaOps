# from sklearn.model_selection import train_test_split
import re
import shutil
import sys
from pathlib import Path

import pandas as pd
import typer
from kagglehub import dataset_download
from loguru import logger
from sklearn.preprocessing import LabelEncoder

logger.remove()
logger.add(sys.stdout, level="DEBUG")


def download():
    # Define the target directory for the raw data
    raw_data_path = Path("data/raw/")
    # Download the dataset
    path = dataset_download("emirhanai/2024-u-s-election-sentiment-on-x", force_download=True)
    path = Path(path)
    print(path)
    for file in path.iterdir():
        if file.is_file():
            shutil.move(str(file), raw_data_path / file.name)


def clean_text(text):
    text = text.lower()  # Lowercase
    text = re.sub(r"http\S+", "", text)  # Remove URLs
    text = re.sub(r"@\w+", "", text)  # Remove mentions
    text = re.sub(r"#\w+", "", text)  # Remove hashtags
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # Remove non-alphabetic characters
    return text


def preprocess():
    logger.info("Preprocessing data")
    raw_data_path = Path("data/raw/")
    train_path = raw_data_path / "train.csv"
    test_path = raw_data_path / "test.csv"
    val_path = raw_data_path / "val.csv"

    # Check if files exist; if not, download them
    if not train_path.exists() or not test_path.exists() or not val_path.exists():
        logger.info("Files not found. Downloading dataset.")
        download()

    train = pd.read_csv("data/raw/train.csv")
    train = train.head(50)
    test = pd.read_csv("data/raw/test.csv")
    val = pd.read_csv("data/raw/val.csv")
    # recombination, resplitting commented out as to keep original splits.
    # data = pd.concat([train.reset_index(), test.reset_index(), val.reset_index()], ignore_index=True)
    # train, test = train_test_split(data, test_size=0.3, random_state=111)
    # train, val = train_test_split(data, test_size=0.2, random_state=111)

    # List of datasets
    datasets = [train, test, val]

    # Apply cleaning and encoding to each dataset
    for dataset in datasets:
        # Apply cleaning
        dataset["clean_text"] = dataset["tweet_text"].apply(clean_text)

        # Encode 'party' using Label Encoding
        le_party = LabelEncoder()
        dataset["party_encoded"] = le_party.fit_transform(dataset["party"])

        # Map sentiment to numerical values for evaluation
        sentiment_mapping = {"negative": 0, "neutral": 1, "positive": 2}
        dataset["sentiment_encoded"] = dataset["sentiment"].map(sentiment_mapping)

    logger.debug(f"Train-size: {train.shape}")
    logger.debug(f"Validation-size: {val.shape}")
    logger.debug(f"Test-size: {test.shape}")
    train.to_csv("data/processed/train.csv")
    val.to_csv("data/processed/val.csv")
    test.to_csv("data/processed/test.csv")


if __name__ == "__main__":
    typer.run(preprocess)
