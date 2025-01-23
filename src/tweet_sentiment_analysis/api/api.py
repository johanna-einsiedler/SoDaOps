from contextlib import asynccontextmanager
import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import BertModel, BertTokenizer
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
from tweet_sentiment_analysis.model import SentimentPipeline
import wandb
# Define model and device configuration
#MODEL_NAME = "bert-base-cased"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ReviewInput(BaseModel):
    """Define input data structure for the endpoint."""

    review: str


class PredictionOutput(BaseModel):
    """Define output data structure for the endpoint."""

    label: str
    score: float


# class SentimentClassifier(nn.Module):
#     """Sentiment Classifier class. Combines BERT model with a dropout and linear layer."""

#     def __init__(self, n_classes, model_name=MODEL_NAME):
#         super().__init__()
#         self.bert = BertModel.from_pretrained(model_name)
#         self.drop = nn.Dropout(p=0.3)
#         self.out = nn.Linear(self.bert.config.hidden_size, n_classes)

#     def forward(self, input_ids, attention_mask):
#         """Forward pass of the model."""
#         output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
#         output = self.drop(output[1])
#         return self.out(output)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model and tokenizer when the app starts and clean up when the app stops."""
    global pipeline
    #model = SentimentClassifier(n_classes=3)
    #model.load_state_dict(torch.load("bert_sentiment_model.pt", map_location=device))
    #model = model.to(device)
    #model.eval()    
    #tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    pipeline = SentimentPipeline()



    # wandb.init(project="application_test", entity="jae__-none")
    # artifact = wandb.use_artifact("magnus-nielsen/secret-project/magic-sweep-1_finetuned_model:v0", type="model")
    # #https://wandb.ai/magnus-nielsen/secret-project/runs/1jwfww8r?nw=nwusermagnusnielsen
    # artifact_dir = artifact.download()


    #  # Load both the model and tokenizer from the artifact directory
    # model = AutoModelForSequenceClassification.from_pretrained(artifact_dir)
    # tokenizer = AutoTokenizer.from_pretrained(artifact_dir)

    # # Initialize the sentiment analysis pipeline
    # #sentiment_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
    # sentiment_pipeline = load_sentiment_pipeline()
    print("Model and tokenizer loaded successfully")

    yield

    # del model, tokenizer


# Initialize FastAPI app
app = FastAPI(lifespan=lifespan)


# Prediction endpoint
@app.post("/predict", response_model=PredictionOutput)
async def predict_sentiment(review_input: ReviewInput):
    """Predict sentiment of the input text."""
    try:
    #     # Encode input text
    #     encoding = tokenizer.encode_plus(
    #         review_input.review,
    #         add_special_tokens=True,
    #         max_length=160,
    #         return_token_type_ids=False,
    #         padding="max_length",
    #         return_attention_mask=True,
    #         return_tensors="pt",
    #     )



    #     input_ids = encoding["input_ids"].to(device)
    #     attention_mask = encoding["attention_mask"].to(device)

    #     # Model prediction
    #     with torch.no_grad():
    #         outputs = model(input_ids, attention_mask)
    #         _, prediction = torch.max(outputs, dim=1)
    #         sentiment = class_names[prediction]

    #     return PredictionOutput(sentiment=sentiment)
    
        # TODO: Convert this into using the same preprocess function that training uses to ensure consistent data preprocessing
        result =pipeline.predict(review_input.review[:16])[0]
        label, score = result["label"], result["score"]
        #logger.info(f"Sentiment analysis result: label={label}, score={score:.4f}")
        return PredictionOutput(label=label, score=score)
    


    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e