#!/bin/bash
# Ensure logs and artifacts directories exist
mkdir -p logs artifacts data/processed
# Load environment variables
source /etc/environment
# Pull data
dvc pull
# Convert raw to processed data
python src/tweet_sentiment_analysis/data.py
# Create the W&B sweep and capture its output
echo "Logging in to W&B"
login_output=$(wandb login 2>&1)
# Print the output for debugging purposes
echo "$login_output"
echo "Creating W&B sweep..."
sweep_output=$(wandb sweep configs/sweep.yaml 2>&1)
# Print the output for debugging purposes
echo "$sweep_output"
# Extract the sweep ID from the output
sweep_id=$(echo "$sweep_output" | awk '/Run sweep agent with:/ {print $NF}')
# Check if the sweep ID was successfully extracted
if [ -z "$sweep_id" ]; then
    echo "Failed to extract the sweep ID. Full sweep output:"
    echo "$sweep_output"
    exit 1
fi
# Log the extracted sweep ID
echo "Sweep created with ID: $sweep_id"
export WANDB_SWEEP_NAME=$(echo "$sweep_id" | awk -F'/' '{print $NF}')
echo "Sweep name: $WANDB_SWEEP_NAME"
# Start the W&B agent
echo "Starting W&B agent for sweep: $WANDB_SWEEP_NAME"
wandb agent --count "$((WANDB_RUN_COUNT))" "$WANDB_ENTITY/$WANDB_PROJECT/$WANDB_SWEEP_NAME"
echo "Testing fetching of best model"
python src/tweet_sentiment_analysis/fetch_and_upload_model.py
echo "Testing inference with single tweet"
python src/tweet_sentiment_analysis/model.py
echo "Testing evaluation of f1 score"
python src/tweet_sentiment_analysis/evaluate.py