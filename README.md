# Energy Forecasting System

A system for forecasting energy consumption (electricity and water) using machine learning models, weather data, and background task processing.

## Overview

This platform uses XGBoost machine learning models to predict energy consumption based on historical data and weather patterns. The system features:

- **Advanced Forecasting**: Separate models for electricity and water consumption
- **API-First Design**: RESTful API with FastAPI for integration with other systems
- **Background Processing**: Task queue system with Dramatiq for handling long-running operations
- **Real-time Data**: Integration with weather services for temperature data
- **Interactive Dashboard**: Web interface for visualizing forecasts and model performance

## Project Structure

```
Energy_Forecasting/
  - config/                # Configuration files (sensitive data excluded from git)
  - data/
    - initial_datasets/    # Base datasets for training and testing
    - Merged_*.csv         # Preprocessed datasets
  - examples/              # Example implementations and integrations
    - scripts/             # Utility and example scripts
    - web_interface/       # Web UI examples including SSE streaming
  - logs/                  # Application logs
  - models/                # Trained ML models
    - electricity/         # Electricity consumption models (.h5)
    - water/               # Water consumption models (.h5)
  - scripts/               # Utility scripts
    - start_api.bat        # Script to start the API server
    - start_worker.bat     # Script to start the background worker
  - src/                   # Main application code
    - api/                 # API endpoints and routes
      - routes/            # API route modules
    - schemas/             # Data validation models
    - services/            # Business logic services
    - dramatiq_*.py        # Background task processing
    - main.py              # FastAPI application
  - weather_utils/         # Weather data integration utilities
```

## Setup

### Prerequisites

- Python 3.8 or higher
- Redis (for background task processing)
- Docker (optional, for containerization)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Energy_Forecasting.git
cd Energy_Forecasting
```

2. Create and activate a virtual environment (optional but recommended):
```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up Redis (required for background tasks):
```bash
# Using Docker
docker run -d -p 6379:6379 --name redis redis
```

5. Configure environment variables (create a `.env` file in the project root):
```
DEBUG=false
REDIS_URL=redis://localhost:6379/0
```

## Running the Application

### Starting the API Server

Run the API server using the provided script:

```bash
scripts\start_api.bat
```

This will start the FastAPI server on http://localhost:8000.

### Starting the Background Worker

Start the background task processing worker:

```bash
scripts\start_worker.bat
```

This worker processes background tasks like model training and evaluation.

## Key Features

### Energy Consumption Forecasting

The system provides forecasting for two types of energy consumption:

- **Electricity**: Forecasts electricity consumption usage based on historical consumption patterns and temperature
- **Water**: Forecasts water consumption usage based on historical consumption patterns and temperature

### Model Training and Evaluation

- Train models for specific meters with:
  - Automatic hyperparameter optimization using Optuna
  - Feature engineering tailored to each consumption type
  - Advanced metrics for model evaluation

### Weather Data Integration

- Automatically retrieves temperature data for forecasting
- Correlates energy consumption with weather patterns

## Deployment

### Production Deployment

For production deployment:

1. Configure appropriate environment variables:
   - Set `DEBUG=false` for production mode
   - Configure `REDIS_URL` to your production Redis instance

2. Use a production ASGI server:
```bash
uvicorn src.main:app --host 0.0.0.0 --port 8000 --workers 4
```

3. For background workers in production:
```bash
python -m src.dramatiq_worker
```

### Docker Deployment (Optional)

1. Build Docker image:
```bash
docker build -t energy-forecasting:latest .
```

2. Run containers:
```bash
# Run Redis
docker run -d -p 6379:6379 --name redis redis

# Run API
docker run -d -p 8000:8000 --name energy-api --link redis:redis energy-forecasting:latest python -m src.main

# Run Worker
docker run -d --name energy-worker --link redis:redis energy-forecasting:latest python -m src.dramatiq_worker
```

## Development

### Debug Mode

Enable debug mode for enhanced logging and auto-reloading:

```bash
set DEBUG=true
scripts\start_api.bat
```

### Testing

Run tests to ensure everything is working correctly:

```bash
pytest
``` 