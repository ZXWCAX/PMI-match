#!/bin/bash

# Set Python path (adjust according to your actual environment)
export PYTHONPATH="../../:$PYTHONPATH"

echo "🚀 Starting DeepLabV3+ training task (Total: 100 Epochs, auto-restarting every 2 Epochs)"

# Infinite loop to keep pulling up the Python script
while true; do
    python train.py

    # Capture the exit status code of the Python script
    EXIT_CODE=$?

    if [ $EXIT_CODE -eq 0 ]; then
        # Check if the 100 epochs training is fully completed
        echo "🔄 Python script exited normally. Preparing for the next restart..."
        sleep 2
    else
        echo "❌ Training interrupted due to an error! Exiting loop."
        break
    fi
done




