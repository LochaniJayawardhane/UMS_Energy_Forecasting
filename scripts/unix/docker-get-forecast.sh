#!/bin/bash
# Script for getting forecasts from Docker containers

# Parse command line arguments
MODEL_TYPE=${1:-electricity}
METER_ID=${2:-meter_123}
DAYS=${3:-7}

# Display information about the operation
echo "Getting $DAYS day forecast for $MODEL_TYPE meter $METER_ID..."
echo "Using the dockerized Energy Forecasting service"
echo

# Call the API to get the forecast
curl -s -X GET "http://localhost:8000/api/v1/forecast/$MODEL_TYPE/$METER_ID?days=$DAYS" \
  -H "Content-Type: application/json" | json_pp

# Add helpful message
echo
echo "Request complete"
echo "For different forecast periods, specify days as the third argument:"
echo "./scripts/docker-get-forecast.sh $MODEL_TYPE $METER_ID <days>" 