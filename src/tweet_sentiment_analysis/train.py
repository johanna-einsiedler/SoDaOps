import os
import shutil
import sys

from datasets import load_dataset
from dotenv import load_dotenv  # For loading .env variables
from loguru import logger
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

import wandb


def load_csv_data(csv_dir: str):
    """Load train and validation CSV files into Hugging Face Dataset."""
    data_files = {"train": os.path.join(csv_dir, "train.parquet"), "val": os.path.join(csv_dir, "val.parquet")}
    dataset = load_dataset("parquet", data_files=data_files)

    # Select only the relevant columns
    dataset = dataset.map(lambda x: {"clean_text": x["clean_text"], "label": x["sentiment_encoded"]})
    return dataset


def clear_output_dir(output_dir: str):
    """
    Deletes all subfolders and files inside the output directory but keeps the directory itself.
    Ignores '.gitkeep' files.
    """
    for item in os.listdir(output_dir):
        item_path = os.path.join(output_dir, item)
        if os.path.isfile(item_path) and item != ".gitkeep":
            os.remove(item_path)  # Remove individual files
        elif os.path.isdir(item_path):
            shutil.rmtree(item_path)  # Remove subfolders


def finetune():
    # Load environment variables from .env
    load_dotenv()
    wandb_project = os.getenv("WANDB_PROJECT")
    wandb_entity = os.getenv("WANDB_ENTITY")
    wandb_api_key = os.getenv("WANDB_API_KEY")

    # Ensure required W&B variables are available
    if not all([wandb_project, wandb_entity, wandb_api_key]):
        raise EnvironmentError("Missing W&B configuration in the environment!")

    # Login to W&B using the API key
    wandb.login(key=wandb_api_key)

    # Initialize wandb
    wandb.init(project=wandb_project, entity=wandb_entity)
    run = wandb.run
    config = wandb.config
    lr = config.lr
    weight_decay = config.weight_decay

    # Set up logging
    logger.remove()
    LOG_LEVEL = "INFO"
    logger.add(sys.stderr, level=LOG_LEVEL)
    logger.add("logs/train_logs.log", level=LOG_LEVEL, rotation="10 MB", retention="10 days")
    logger.info(f"Starting run with project={run.project}, entity={run.entity}, run={run.name}")

    # Load model
    logger.info("Loading model")
    model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    output_dir = "./models"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    # Load data
    logger.info("Loading data")
    csv_dir = "data/processed"
    dataset = load_csv_data(csv_dir)
    train_data = dataset["train"]
    val_data = dataset["val"]

    def preprocess_function(examples):
        return tokenizer(examples["clean_text"], truncation=True, padding="max_length", max_length=16)

    train_data = train_data.map(preprocess_function, batched=True)
    val_data = val_data.map(preprocess_function, batched=True)

    # Set up trainer
    logger.info("Setting up training args and trainer")
    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        learning_rate=lr,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=1,
        weight_decay=weight_decay,
        logging_steps=10,
        save_total_limit=1,
        report_to="wandb",
        load_best_model_at_end=True,  # Ensure best model is loaded if overfitting
        metric_for_best_model="eval_loss",
        save_strategy="epoch",
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        processing_class=tokenizer,
    )

    # Train and save the best model
    logger.info("Training")
    trainer.train()

    # Remove local
    logger.info(f"Clearing local directory of checkpoints: {output_dir}")
    clear_output_dir(output_dir)
    logger.info("Local directory cleared.")

    # Save best model
    best_model_dir = os.path.join(output_dir, "best_model")
    best_model_dir = str(best_model_dir)
    logger.info("Saving best model locally")
    model.save_pretrained(best_model_dir)
    logger.info("Saving tokenizer locally")
    tokenizer.save_pretrained(best_model_dir)

    # Log model as a wandb artifact
    logger.info("Saving model to wandb")
    artifact = wandb.Artifact(f"{run.name}_finetuned_model", type="model")
    artifact.add_dir(best_model_dir)
    logger.info("Logging artifact")
    wandb.log_artifact(artifact)
    logger.info("Finishing WandB")
    wandb.finish()

    # Remove local
    logger.info(f"Clearing local directory of best model: {output_dir}")
    clear_output_dir(output_dir)
    logger.info("Local directory cleared.")


if __name__ == "__main__":
    finetune()
