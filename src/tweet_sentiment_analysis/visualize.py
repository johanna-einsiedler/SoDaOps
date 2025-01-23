import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import typer
from google.cloud import storage
from loguru import logger
from sklearn.metrics import auc, classification_report, confusion_matrix, roc_curve
from sklearn.preprocessing import label_binarize

from data import preprocess
from tweet_sentiment_analysis.model import SentimentPipeline

logger.remove()
logger.add(sys.stdout, level="DEBUG")


def visualize(use_test_set: bool = False) -> None:
    "Visualizing model performance"
    process_path = Path("data/processed/")
    train_path = process_path / "train.parquet"
    test_path = process_path / "test.parquet"
    val_path = process_path / "val.parquet"
    plot_name ='plot'
    fig_path ='../../reports/figures/'
    # upload to gcp
     # Initialize the Google Cloud Storage client
    client = storage.Client()

    bucket_name = "sentiment-output-dtu"
    destination_blob_name = "visualisations/"
    # Get the bucket
    bucket = client.bucket(bucket_name)

    # train = pd.read_parquet("../../data/processed/train.parquet")
    # test = pd.read_parquet("../../data/processed/test.parquet")
    # val = pd.read_parquet("../../data/processed/val.parquet")
    # Check if files exist; if not, preprocess them
    if not train_path.exists() or not test_path.exists() or not val_path.exists():
       logger.info("Files not found. Preprocessing dataset.")
       preprocess()

    train = pd.read_parquet(train_path)
    test = pd.read_parquet(test_path)
    val = pd.read_parquet(val_path)
    pipe = SentimentPipeline()

    text_input = train["tweet_text"].iloc[0]
    if not isinstance(text_input, str):
        text_input = str(text_input)

    # Use the analyze_sentiment method to perform sentiment analysis
    result = pipe.predict(text_input)
    logger.debug(result)

    # Apply sentiment analysis to training set
    train[["predicted_sentiment", "confidence"]] = pd.DataFrame(list(train["tweet_text"].apply(lambda x: pipe.predict(x)[0])))
    logger.debug(f"Sentiment train head: {train[['tweet_text', 'predicted_sentiment', 'confidence']].head()}")

    # Apply sentiment analysis to validation set
    val[["predicted_sentiment", "confidence"]] = pd.DataFrame(list(val["tweet_text"].apply(lambda x: pipe.predict(x)[0])))
    logger.debug(f"Sentiment validation head: {val[['tweet_text', 'predicted_sentiment', 'confidence']].head()}")

    # Apply sentiment analysis to test set
    # test[['predicted_sentiment', 'confidence']] = test['tweet_text'].apply(lambda x: pd.Series(analyze_sentiment(x)))
    val["predicted_sentiment"] = val["tweet_text"].apply(lambda x: pipe.predict(x)[0]["label"])

    # Convert predicted sentiment to lowercase
    val["predicted_sentiment"] = val["predicted_sentiment"].str.lower()

    # Convert to categorical AFTER converting to lowercase AND adding the category
    val["predicted_sentiment"] = pd.Categorical(
        val["predicted_sentiment"], categories=["negative", "neutral", "positive"]
    )

    # Generate classification report
    logger.info(
        classification_report(
            val["sentiment"], val["predicted_sentiment"], target_names=["negative", "neutral", "positive"]
        )
    )  # Explicitly stating categories here

    # Generate confusion matrix using lowercase labels
    cm = confusion_matrix(
        val["sentiment"], val["predicted_sentiment"], labels=["negative", "neutral", "positive"]
    )  # Use lowercase labels

    # Generate probabilities for each class
    # Note: The pipeline provides only the top prediction, so for multi-class ROC, a different approach or model might be needed.
    # Here, we demonstrate ROC for POSITIVE class only as an example.
    y_prob = val["confidence"]  # Confidence scores for the predicted label
    y_true = (val["sentiment"] == "POSITIVE").astype(int)
    # # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["negative", "neutral", "positive"],
        yticklabels=["negative", "neutral", "positive"],
    )  # Lowercase labels for ticks
    plt.title("Confusion Matrix for Validation Set")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig(fig_path+"confusion_matrix_"+plot_name)
    blob_img = bucket.blob(destination_blob_name+"confusion_matrix_"+plot_name+'.png')
    blob_img.upload_from_filename(fig_path+"confusion_matrix_"+plot_name+'.png')
    # Binarize the output for ROC curve (One-vs-Rest)
    # y_val_binarized = label_binarize(val['sentiment'], classes=['NEGATIVE', 'NEUTRAL', 'POSITIVE'])
    # n_classes = y_val_binarized.shape[1]

    # # Compute ROC curve and AUC for POSITIVE class
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    # # Plot ROC Curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label="ROC curve (area = %0.2f)" % roc_auc)
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC)")
    plt.legend(loc="lower right")
    plt.savefig(fig_path+"roc_curve_"+plot_name)
    blob_img = bucket.blob(destination_blob_name+"roc_curve_"+plot_name+'.png')
    blob_img.upload_from_filename(fig_path+"roc_curve_"+plot_name+'.png')

    # # Plot confidence distribution by sentiment
    plt.figure(figsize=(10, 6))
    sns.boxplot(x="predicted_sentiment", y="confidence", data=train, palette="coolwarm")
    plt.title("Confidence Scores by Predicted Sentiment")
    plt.xlabel("Predicted Sentiment")
    plt.ylabel("Confidence Score")
    plt.savefig(fig_path+"confidence_distribution_"+plot_name)
    blob_img = bucket.blob(destination_blob_name+"confidence_distribution_"+plot_name+'.png')
    blob_img.upload_from_filename(fig_path+"confidence_distribution_"+plot_name+'.png')

    # Average confidence score by party and sentiment
    plt.figure(figsize=(14, 8))
    sns.barplot(x="party", y="confidence", hue="predicted_sentiment", data=train, palette="coolwarm")
    plt.title("Average Confidence Score by Party and Sentiment")
    plt.xlabel("Political Party")
    plt.ylabel("Average Confidence Score")
    plt.xticks(rotation=45)
    plt.legend(title="Predicted Sentiment")
    plt.savefig(fig_path+"confidence_by_party_and_sentiment_"+plot_name)
    blob_img = bucket.blob(destination_blob_name+"confidence_by_party_and_sentiment_"+plot_name+'.png')
    blob_img.upload_from_filename(fig_path+"confidence_by_party_and_sentiment_"+plot_name+'.png')




if __name__ == "__main__":
    typer.run(visualize)
