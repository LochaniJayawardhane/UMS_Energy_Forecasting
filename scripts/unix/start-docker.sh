#!/bin/bash
# Script for starting Energy Forecasting with Docker (Linux/Mac)

echo "Starting Energy Forecasting with Docker..."
echo

# Check Docker installation
if ! command -v docker &> /dev/null; then
    echo "Error: Docker is not installed or not in your PATH."
    echo "Please install Docker from https://docs.docker.com/get-docker/"
    echo
    exit 1
fi

# Check Docker Compose
if ! command -v docker-compose &> /dev/null; then
    echo "Error: Docker Compose is not installed or not in your PATH."
    echo "It should be included with Docker installation."
    echo
    exit 1
fi

# Check .env file
if [ ! -f .env ]; then
    echo ".env file not found."
    echo "Creating from template..."
    cp env.example .env
    echo "Please edit the .env file with your actual values."
    echo
    read -p "Press Enter to continue..."
fi

echo "Building and starting Docker containers..."
docker-compose up -d --build

echo
echo "Containers are starting up. This may take a minute."
echo
echo "You can access the API at: http://localhost:8000"
echo "API documentation is available at: http://localhost:8000/docs"
echo
echo "Use these scripts to interact with the system:"
echo "- bash scripts/docker-train-model.sh [model_type] [meter_id]"
echo "- bash scripts/docker-get-forecast.sh [model_type] [meter_id] [days]"
echo
echo "To view logs:"
echo "docker-compose logs -f api"
echo 