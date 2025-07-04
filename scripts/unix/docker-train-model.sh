#!/bin/bash
# Script for training models using Docker containers

# Parse command line arguments
MODEL_TYPE=${1:-electricity}
METER_ID=${2:-meter_123}

# Display information about the operation
echo "Training $MODEL_TYPE model for meter $METER_ID..."
echo "Using the dockerized Energy Forecasting service"
echo

# Call the API to start model training
RESPONSE=$(curl -s -X POST "http://localhost:8000/api/v1/training/$MODEL_TYPE/$METER_ID" \
  -H "Content-Type: application/json" \
  -d '{}')

# Extract task ID from response
TASK_ID=$(echo $RESPONSE | grep -o '"task_id":"[^"]*"' | cut -d'"' -f4)

if [ -z "$TASK_ID" ]; then
  echo "Failed to start training task. Response:"
  echo $RESPONSE
  exit 1
fi

echo "Training started with task ID: $TASK_ID"
echo
echo "To check training status:"
echo "curl http://localhost:8000/api/v1/training/status/$TASK_ID"
echo
echo "When complete, you can get a forecast using:"
echo "./scripts/docker-get-forecast.sh $MODEL_TYPE $METER_ID" 