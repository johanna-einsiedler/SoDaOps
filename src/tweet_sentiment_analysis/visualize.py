import matplotlib.pyplot as plt
from loguru import logger
import seaborn as sns
import pandas as pd
from data import load_data
from model import SentimentModel
import sys
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import typer

logger.remove()
logger.add(sys.stdout, level="DEBUG")

def visualize(use_test_set: bool=False) -> None:
    "Visualizing model performance"
    train, test, val=load_data()
    pipe= SentimentModel()

    text_input = train['tweet_text'].iloc[0]
    if not isinstance(text_input, str):
        text_input = str(text_input)

    # Use the analyze_sentiment method to perform sentiment analysis
    result = pipe.predict(text_input)
    logger.debug(result)

    # Function to perform sentiment analysis using the pipeline
    def make_predictions(text):
        try:
            text = str(text)
            result = pipe.predict(text[:512])[0]  # Truncate to 512 tokens
            return result['label'], result['score']
        except Exception:
            return 'ERROR', 0

    # Apply sentiment analysis to training set
    train[['predicted_sentiment', 'confidence']] = train['tweet_text'].apply(lambda x: pd.Series(make_predictions(x)))
    logger.debug(f"Sentiment train head: {train[['tweet_text','predicted_sentiment','confidence']].head()}")
    
    if use_test_set:
        plot_name="test_set"
        # Apply sentiment analysis to validation set
        test[['predicted_sentiment', 'confidence']] = test['tweet_text'].apply(lambda x: pd.Series(make_predictions(x)))
        logger.debug(f"Sentiment validation head: {test[['tweet_text', 'predicted_sentiment', 'confidence']].head()}")

        # Apply sentiment analysis to test set
        # test[['predicted_sentiment', 'confidence']] = test['tweet_text'].apply(lambda x: pd.Series(make_predictions(x)))
        test['predicted_sentiment'] = test['tweet_text'].apply(lambda x: make_predictions(x)[0])

        # Convert predicted sentiment to lowercase
        test['predicted_sentiment'] = test['predicted_sentiment'].str.lower()

        # Convert to categorical AFTER converting to lowercase AND adding the category
        test['predicted_sentiment'] = pd.Categorical(test['predicted_sentiment'], categories=['negative', 'neutral', 'positive'])

        # Generate classification report
        logger.info(classification_report(test['sentiment'], test['predicted_sentiment'], target_names=['negative', 'neutral', 'positive'])) #Explicitly stating categories here

        # Generate confusion matrix using lowercase labels
        cm = confusion_matrix(test['sentiment'], test['predicted_sentiment'], labels=['negative', 'neutral', 'positive'])  # Use lowercase labels

        # Generate probabilities for each class
        # Note: The pipeline provides only the top prediction, so for multi-class ROC, a different approach or model might be needed.
        # Here, we demonstrate ROC for POSITIVE class only as an example.
        y_prob = test['confidence']  # Confidence scores for the predicted label
        y_true = (test['sentiment'] == 'POSITIVE').astype(int)
    else:
        plot_name="val_set"
        # Apply sentiment analysis to validation set
        val[['predicted_sentiment', 'confidence']] = val['tweet_text'].apply(lambda x: pd.Series(make_predictions(x)))
        logger.debug(f"Sentiment validation head: {val[['tweet_text', 'predicted_sentiment', 'confidence']].head()}")

        # Apply sentiment analysis to test set
        # test[['predicted_sentiment', 'confidence']] = test['tweet_text'].apply(lambda x: pd.Series(make_predictions(x)))
        val['predicted_sentiment'] = val['tweet_text'].apply(lambda x: make_predictions(x)[0])

        # Convert predicted sentiment to lowercase
        val['predicted_sentiment'] = val['predicted_sentiment'].str.lower()

        # Convert to categorical AFTER converting to lowercase AND adding the category
        val['predicted_sentiment'] = pd.Categorical(val['predicted_sentiment'], categories=['negative', 'neutral', 'positive'])

        # Generate classification report
        logger.info(classification_report(val['sentiment'], val['predicted_sentiment'], target_names=['negative', 'neutral', 'positive'])) #Explicitly stating categories here

        # Generate confusion matrix using lowercase labels
        cm = confusion_matrix(val['sentiment'], val['predicted_sentiment'], labels=['negative', 'neutral', 'positive'])  # Use lowercase labels
        
        # Generate probabilities for each class
        # Note: The pipeline provides only the top prediction, so for multi-class ROC, a different approach or model might be needed.
        # Here, we demonstrate ROC for POSITIVE class only as an example.
        y_prob = val['confidence']  # Confidence scores for the predicted label
        y_true = (val['sentiment'] == 'POSITIVE').astype(int)
    # # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['negative', 'neutral', 'positive'], yticklabels=['negative', 'neutral', 'positive'])  # Lowercase labels for ticks
    plt.title('Confusion Matrix for Validation Set')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(f"reports/figures/confusion_matrix_{plot_name}")

    # Binarize the output for ROC curve (One-vs-Rest)
    #y_val_binarized = label_binarize(val['sentiment'], classes=['NEGATIVE', 'NEUTRAL', 'POSITIVE'])
    #n_classes = y_val_binarized.shape[1]

    # # Compute ROC curve and AUC for POSITIVE class
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    # # Plot ROC Curve
    plt.figure(figsize=(8,6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc='lower right')
    plt.savefig(f"reports/figures/roc_curve_{plot_name}")

    # # Plot confidence distribution by sentiment
    plt.figure(figsize=(10,6))
    sns.boxplot(x='predicted_sentiment', y='confidence', data=train, palette='coolwarm')
    plt.title('Confidence Scores by Predicted Sentiment')
    plt.xlabel('Predicted Sentiment')
    plt.ylabel('Confidence Score')
    plt.savefig(f"reports/figures/confidence_distribution_{plot_name}")

    # Average confidence score by party and sentiment
    plt.figure(figsize=(14,8))
    sns.barplot(x='party', y='confidence', hue='predicted_sentiment', data=train, palette='coolwarm')
    plt.title('Average Confidence Score by Party and Sentiment')
    plt.xlabel('Political Party')
    plt.ylabel('Average Confidence Score')
    plt.xticks(rotation=45)
    plt.legend(title='Predicted Sentiment')
    plt.savefig(f"reports/figures/confidence_by_party_and_sentiment_{plot_name}")


if __name__ == "__main__":
    typer.run(visualize)
