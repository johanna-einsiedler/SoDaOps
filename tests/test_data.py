from torch.utils.data import Dataset
from datasets import load_dataset
from tweet_sentiment_analysis.data import *
import os
from tests import _PATH_DATA


def test_my_dataset():
    csv_dir = _PATH_DATA + '/processed'
    data_files = {"train": os.path.join(csv_dir, "train.csv"),  "test": os.path.join(csv_dir, "test.csv"), "val": os.path.join(csv_dir, "val.csv")}
    dataset = load_dataset("csv", data_files=data_files)
    assert dataset.shape['train'][0] == dataset.shape['val'][0] == dataset.shape['test'][0], "Train,test and validation data don't have same shape"

