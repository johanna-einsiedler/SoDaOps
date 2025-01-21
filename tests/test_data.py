import os

import pytest
from datasets import load_dataset
from torch.utils.data import Dataset

from tests import _PATH_DATA
from tweet_sentiment_analysis.data import *

data_dir = _PATH_DATA + "/processed"


@pytest.mark.skipif(not os.path.exists(data_dir), reason="Data files not found")
def test_my_dataset():
    data_files = {
        "train": os.path.join(data_dir, "train.csv"),
        "test": os.path.join(data_dir, "test.csv"),
        "val": os.path.join(data_dir, "val.csv"),
    }
    dataset = load_dataset("csv", data_files=data_files)
    assert dataset.shape["train"][0] == dataset.shape["val"][0] == dataset.shape["test"][0], (
        "Train,test and validation data don't have same shape"
    )
