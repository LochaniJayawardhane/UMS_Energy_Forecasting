# Energy Forecasting System

A system for forecasting energy consumption (electricity and water) using machine learning models, weather data, and background task processing.

## Project Structure

```
Energy_Focasting/
  - config/               # Configuration files
  - data/                 # Data files
  - docs/                 # Documentation
  - examples/             # Example scripts and web interfaces
    - scripts/            # Example scripts
    - web_interface/      # Example web interfaces
  - logs/                 # Log files
  - models/               # Trained models
  - scripts/              # Utility scripts
    - start_api.bat       # Script to start the API server
    - start_worker.bat    # Script to start the background worker
  - src/                  # Source code
    - dramatiq_broker.py  # Dramatiq broker configuration
    - dramatiq_worker.py  # Dramatiq worker
    - influx_client.py    # InfluxDB client
    - logger_config.py    # Logging configuration
    - main.py             # FastAPI application
    - task_system.py      # Background task system
    - utils.py            # Utility functions
  - tests/                # Tests
  - weather_utils/        # Weather utilities
```

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Start Redis (required for background tasks):
```bash
docker run -d -p 6379:6379 --name redis redis
```

## Running the Application

1. Start the API server:
```bash
scripts\start_api.bat
```

2. Start the background worker:
```bash
scripts\start_worker.bat
```

## API Documentation

- API Documentation: http://localhost:8000/docs
- Health Check: http://localhost:8000/
- Worker Health: http://localhost:8000/health/worker

## Documentation

See the `docs/` directory for detailed documentation:

- [System Documentation](docs/SYSTEM_DOCUMENTATION.md)
- [Background Processing](docs/BACKGROUND_PROCESSING.md)
- [Dramatiq README](docs/README_DRAMATIQ.md)
- [Server-Sent Events](docs/README_SSE.md)
- [Weather API](docs/README_WEATHER_API.md) 