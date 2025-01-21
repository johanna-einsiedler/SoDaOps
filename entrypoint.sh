#!/bin/bash
# Ensure logs and artifacts directories exist
mkdir -p logs artifacts
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
echo "Starting W&B agent for sweep: $sweep_id"
# Start the W&B agent
wandb agent "$sweep_id"