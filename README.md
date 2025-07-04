# Energy Forecasting System

A system for forecasting energy consumption (electricity and water) using machine learning models, weather data, and background task processing.

## Overview

This platform uses XGBoost machine learning models to predict energy consumption based on historical data and weather patterns. The system features:

- **Advanced Forecasting**: Separate models for electricity and water consumption
- **API-First Design**: RESTful API with FastAPI for integration with other systems
- **Background Processing**: Task queue system with Dramatiq for handling long-running operations
- **Real-time Data**: Integration with weather services for temperature data
- **Interactive Dashboard**: Web interface for visualizing forecasts and model performance
- **Docker Support**: Fully containerized for easy deployment and development

## Project Structure

```
Energy_Forecasting/
├── config/                 # Configuration files
│   ├── influxdb_config.py  # InfluxDB connection settings
│   ├── model_config.py     # Model configuration
│   └── weather_config.py   # Weather API settings
├── data/                   # Data storage
│   └── Merged_*.csv        # Preprocessed datasets
├── examples/               # Example implementations
│   ├── scripts/            # Utility scripts
│   │   └── load_electricity_data.py  # Data loading script
│   └── web_interface/      # Web UI examples
├── logs/                   # Application logs
├── models/                 # Trained ML models
│   ├── electricity/        # Electricity models (.h5)
│   └── water/              # Water models (.h5)
├── scripts/                # Utility scripts
│   ├── windows/            # Windows batch scripts
│   │   ├── start-docker.bat
│   │   ├── docker-train-model.bat
│   │   └── docker-get-forecast.bat
│   ├── unix/               # Unix/Linux shell scripts
│   │   ├── start-docker.sh
│   │   ├── docker-train-model.sh
│   │   └── docker-get-forecast.sh
│   ├── start_api.bat       # Legacy API startup
│   └── start_worker.bat    # Legacy worker startup
├── src/                    # Main application code
│   ├── api/                # API endpoints and routes
│   │   └── routes/         # API route modules
│   ├── schemas/            # Data validation models
│   ├── services/           # Business logic services
│   ├── dramatiq_broker.py  # Task queue configuration
│   ├── dramatiq_worker.py  # Background worker
│   ├── influx_client.py    # InfluxDB client
│   ├── main.py             # FastAPI application
│   └── task_system.py      # Background task definitions
├── weather_utils/          # Weather data integration
├── .dockerignore           # Docker ignore file
├── .env.example            # Environment variables template
├── docker-compose.yml      # Docker services configuration
├── Dockerfile              # Docker image definition
└── requirements.txt        # Python dependencies
```

## Quick Start with Docker (Recommended)

### Prerequisites

- Docker and Docker Compose installed
- Git for cloning the repository

### 1. Clone and Setup

```bash
git clone <repository-url>
cd Energy_Forecasting
```

### 2. Start the System

**Windows:**
```bash
scripts\windows\start-docker.bat
```

**Linux/Mac:**
```bash
bash scripts/unix/start-docker.sh
```

This will:
- Check Docker installation
- Create `.env` file from template if needed
- Build and start all containers (API, Worker, Redis, InfluxDB)
- Provide access URLs and usage instructions

### 3. Load Sample Data

After containers are running, load your data:

```bash
# Copy your data file to the container
docker cp data/Merged_Electricity_Temperature.csv energy_focasting-api-1:/app/data/

# Run the data loading script
docker exec -it energy_focasting-api-1 bash -c "cd /app && python -m examples.scripts.load_electricity_data"
```

### 4. Train Models

**Windows:**
```bash
scripts\windows\docker-train-model.bat electricity meter_test_001
```

**Linux/Mac:**
```bash
bash scripts/unix/docker-train-model.sh electricity meter_test_001
```

### 5. Get Forecasts

**Windows:**
```bash
scripts\windows\docker-get-forecast.bat electricity meter_test_001 7
```

**Linux/Mac:**
```bash
bash scripts/unix/docker-get-forecast.sh electricity meter_test_001 7
```

## Docker Services

The system runs four main services:

- **API** (Port 8000): FastAPI server for REST endpoints
- **Worker**: Background task processor for model training
- **Redis** (Port 6379): Task queue and results storage
- **InfluxDB** (Port 8086): Time series database for energy data

### Container Management

```bash
# View running containers
docker-compose ps

# View logs
docker-compose logs -f api
docker-compose logs -f worker

# Stop all services
docker-compose down

# Restart services
docker-compose restart

# Rebuild after code changes
docker-compose up -d --build
```

## Environment Configuration

Create a `.env` file based on `env.example`:



# Redis Configuration
REDIS_URL=redis://redis:6379/0

# Debug mode
DEBUG=false
```

## API Endpoints

Once running, access the API at `http://localhost:8000`:

- **API Documentation**: `http://localhost:8000/docs`
- **Health Check**: `http://localhost:8000/`
- **Training**: `POST /api/v1/training/{meter_type}/{meter_id}`
- **Forecasting**: `GET /api/v1/forecast/{meter_type}/{meter_id}?days={days}`
- **Training Status**: `GET /api/v1/training/status/{task_id}`

## Data Persistence

All data is persisted through Docker volumes:
- `./models`: Trained ML models
- `./data`: Input/output data files  
- `./logs`: Application logs
- `redis-data`: Redis queue data
- `influxdb-data`: InfluxDB time series data

## Development Setup (Non-Docker)

### Prerequisites

- Python 3.8 or higher
- Redis server
- InfluxDB server

### Installation

1. Create virtual environment:
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Start Redis:
```bash
docker run -d -p 6379:6379 --name redis redis
```

4. Configure environment variables in `.env` file

5. Start services:
```bash
# Start API
scripts\start_api.bat

# Start Worker (in separate terminal)
scripts\start_worker.bat
```

## Key Features

### Energy Consumption Forecasting

- **Electricity**: Forecasts based on historical consumption and temperature
- **Water**: Forecasts based on historical consumption and temperature
- **Multi-meter Support**: Train models for different meters
- **Configurable Forecast Periods**: 1-30 days ahead

### Model Training and Evaluation

- **Automatic Hyperparameter Optimization**: Using Optuna
- **Feature Engineering**: Tailored to each consumption type
- **Advanced Metrics**: RMSE, MAE, MAPE for model evaluation
- **Background Processing**: Non-blocking model training

### Weather Data Integration

- **Automatic Temperature Retrieval**: For forecasting periods
- **Weather Correlation**: Energy consumption vs weather patterns
- **Configurable Locations**: Support for multiple geographic areas

### Real-time Monitoring

- **Task Progress Tracking**: Real-time training status updates
- **Server-Sent Events**: Live progress streaming
- **Comprehensive Logging**: Structured logging with different levels

## Troubleshooting

### Common Issues

1. **Port Conflicts**: If ports 8000, 6379, or 8086 are in use, stop existing services
2. **Permission Errors**: Ensure Docker has proper permissions on mounted volumes
3. **Data Loading Issues**: Verify CSV file format and InfluxDB connection
4. **Model Training Failures**: Check data availability and InfluxDB configuration

### Debug Mode

Enable debug mode for enhanced logging:
```bash
# In .env file
DEBUG=true

# Restart containers
docker-compose restart
```

### Viewing Logs

```bash
# API logs
docker-compose logs -f api

# Worker logs  
docker-compose logs -f worker

# All logs
docker-compose logs -f
```

## Production Deployment

For production deployment:

1. **Security**: Update default passwords and tokens
2. **Environment**: Set `DEBUG=false`
3. **Resources**: Configure appropriate memory and CPU limits
4. **Monitoring**: Set up log aggregation and monitoring
5. **Backup**: Configure regular backups of volumes

```bash
set DEBUG=true
scripts\start_api.bat
```

### Testing

Run tests to ensure everything is working correctly:

```bash
pytest
``` 