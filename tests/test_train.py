from unittest.mock import patch

from datasets import Dataset, DatasetDict

from tweet_sentiment_analysis.train import load_parquet_data


@patch("tweet_sentiment_analysis.train.load_dataset")
def test_load_parquet_data(mock_load_dataset):
    # Create mock datasets with necessary keys
    mock_train_data = Dataset.from_dict({"clean_text": ["mock text"], "sentiment_encoded": [1]})
    mock_val_data = Dataset.from_dict({"clean_text": ["mock text"], "sentiment_encoded": [1]})
    mock_dataset_dict = DatasetDict({"train": mock_train_data, "val": mock_val_data})
    mock_load_dataset.return_value = mock_dataset_dict

    dataset = load_parquet_data("mock/path")

    # Assert that `load_dataset` was called with correct arguments
    mock_load_dataset.assert_called_once_with(
        "parquet", data_files={"train": "mock/path/train.parquet", "val": "mock/path/val.parquet"}
    )

    # Check that dataset is of type DatasetDict
    assert isinstance(dataset, DatasetDict)

    # Check the dataset structure
    assert isinstance(dataset["train"], Dataset)
    assert isinstance(dataset["val"], Dataset)
    assert len(dataset["train"]) > 0  # Verify non-empty dataset
    assert len(dataset["val"]) > 0

    # Optionally, check the sample content
    assert dataset["train"][0]["clean_text"] == "mock text"
    assert dataset["train"][0]["sentiment_encoded"] == 1
    assert dataset["val"][0]["clean_text"] == "mock text"
    assert dataset["val"][0]["sentiment_encoded"] == 1


# @patch('wandb.login')
# @patch('wandb.init')
# @patch('wandb.Artifact')
# @patch('datasets.load_dataset')
# @patch('transformers.AutoTokenizer.from_pretrained')
# @patch('transformers.AutoModelForSequenceClassification.from_pretrained')
# @patch('builtins.open', new_callable=MagicMock)  # Mock for dotenv
# def test_finetune(mock_open, mock_model_class, mock_tokenizer_class, mock_load_dataset, mock_artifact, mock_wandb_init, mock_wandb_login):
#     """
#     Test finetune function.
#     """

#     # Mock environment variables
#     mock_open.return_value = MagicMock(read=lambda: "WANDB_PROJECT=mock_project\nWANDB_ENTITY=mock_entity\nWANDB_API_KEY=mock_key")
#     mock_wandb_login.return_value = None

#     # Mock dataset loading
#     mock_load_dataset.return_value = {"train": MagicMock(), "val": MagicMock()}

#     # Mock model and tokenizer loading
#     mock_tokenizer_class.return_value = MagicMock()
#     mock_model_class.return_value = MagicMock()

#     # Mock WandB Artifact
#     mock_artifact.return_value = MagicMock()

#     # Call finetune function
#     finetune()

#     # Assertions
#     mock_open.assert_called_once_with('.env', 'r')  # dotenv file opened
#     mock_wandb_login.assert_called_once_with(key='mock_key')  # WandB login
#     mock_wandb_init.assert_called_once_with(project='mock_project', entity='mock_entity')  # WandB init
#     mock_load_dataset.assert_called_once_with('parquet', data_files={'train': 'data/processed/train.parquet', 'val': 'data/processed/val.parquet'})  # dataset loading
#     mock_tokenizer_class.assert_called_once_with('cardiffnlp/twitter-roberta-base-sentiment-latest')  # tokenizer loading
#     mock_model_class.assert_called_once_with('cardiffnlp/twitter-roberta-base-sentiment-latest')  # model loading
#     mock_artifact.assert_called_once()  # Artifact created
