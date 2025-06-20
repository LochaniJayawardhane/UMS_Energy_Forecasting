# Energy Forecasting System - Complete Technical Documentation

## 🏗️ System Architecture

We've built a **comprehensive energy consumption forecasting system** that combines time-series data processing, machine learning, and background task processing for automated model training.

### High-Level Architecture
```
External APIs (Weather) → FastAPI → Dramatiq Workers → Redis → InfluxDB → ML Models
                                ↓
                          Background Tasks
                          Progress Tracking
                          Result Storage
```

---

## 🛠️ Technologies Stack

### 1. Core Framework
- **FastAPI** - Modern, fast web framework for building APIs
- **Python 3.x** - Main programming language
- **Uvicorn** - ASGI server for running FastAPI

### 2. Database & Storage
- **InfluxDB** - Time-series database for storing consumption and temperature data
- **Redis** - In-memory data store for task queues, progress tracking, and caching

### 3. Machine Learning
- **XGBoost** - Gradient boosting framework for regression models
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computing
- **Scikit-learn** - Machine learning utilities and metrics

### 4. Background Task Processing
- **Dramatiq** - Fast, reliable task queue system (replaced Celery)
- **Redis Backend** - Storage for task queues and results

### 5. Logging & Monitoring
- **Structlog** - Structured logging
- **Colorlog** - Colored console output
- **File-based logging** with rotation

### 6. Weather Integration
- **OpenWeatherMap API** - Historical and forecasted weather data
- **Custom weather utilities** - Location management and temperature forecasting

---

## 📊 Data Flow & Storage

### 1. Data Sources
```
📡 InfluxDB (Primary)
├── Electricity consumption data
├── Water consumption data  
└── Temperature readings

🌤️ OpenWeatherMap API
├── Historical temperature data
├── Weather forecasts (5-day)
└── Location-based weather
```

### 2. Data Storage Locations

#### InfluxDB (Time-Series Data)
```
Database: energy_forecasting
├── Measurements:
│   ├── electricity_consumption
│   ├── water_consumption
│   └── temperature_data
└── Fields: DateTime, Consumption, Temperature, meter_id
```

#### Local File System
```
models/
├── electricity/
│   └── {meter_id}.h5     # XGBoost models
└── water/
    └── {meter_id}.h5     # XGBoost models

data/
├── initial_datasets/
│   ├── Electricity_Consumption_data.csv
│   ├── Water_Consumption.csv
│   └── Temperature_Data.csv
├── Merged_Electricity_Temperature.csv
└── Merged_Water_Temperature.csv

logs/
└── energy_forecasting_{timestamp}.log
```

---

## 🔴 Redis Configuration & Usage

### 1. Redis Setup
```bash
# Docker Redis Container
docker run -d -p 6379:6379 --name redis redis

# Connection Details
Host: localhost
Port: 6379
Database: 0 (default)
URL: redis://localhost:6379/0
```

### 2. Redis Usage in Our System

#### A. Task Queue Management
```python
# Dramatiq uses Redis as message broker
REDIS_URL = "redis://localhost:6379/0"

# Queue Structure:
dramatiq:default.DQ    # Default queue for tasks
dramatiq:default.XQ    # Dead letter queue for failed tasks
```

#### B. Progress Tracking
```python
# Progress tracking keys in Redis:
task_progress:{task_id} = {
    "task_id": "uuid",
    "state": "TRAINING_MODEL", 
    "progress": 75,
    "message": "Training model...",
    "details": {...},
    "started_at": "2025-06-19T13:00:00",
    "updated_at": "2025-06-19T13:15:00"
}
```

#### C. Task Results Storage
```python
# Results storage:
dramatiq-result:{task_id} = {
    "status": "success",
    "message": "Model trained successfully",
    "details": {
        "model_path": "models/electricity/meter_001.h5",
        "training_time": 512.36,
        "data_points": 26388
    }
}
```

#### D. Error Tracking
```python
# Error information:
task_error:{task_id} = {
    "error": "Connection timeout",
    "error_type": "CONNECTION_ERROR",
    "traceback": "...",
    "timestamp": "2025-06-19T13:30:00"
}
```

### 3. Redis Configuration in Code
```python
# dramatiq_broker.py
broker = dramatiq.broker.redis.RedisBroker(
    host="localhost",
    port=6379,
    db=0,
    connection_pool_kwargs={
        'max_connections': 20,
        'retry_on_timeout': True,
        'socket_keepalive': True,
        'socket_keepalive_options': {},
        'socket_connect_timeout': 10,
        'socket_timeout': 10
    }
)

# Middleware Stack:
├── TimeLimit (task timeouts)
├── Callbacks (success/failure hooks)  
├── Retries (automatic retry logic)
├── Results (result storage)
└── AgeLimit (message expiration)
```

---

## ⚙️ Background Task System (Dramatiq)

### 1. Why We Switched from Celery to Dramatiq
```
Celery Issues:
❌ Complex configuration
❌ State corruption errors
❌ Slower performance (11-17s for 20k tasks)
❌ Memory leaks
❌ Difficult debugging

Dramatiq Benefits:  
✅ Simple configuration
✅ Reliable state management
✅ 3x faster performance (4-5s for 20k tasks)
✅ Better error handling
✅ Enhanced debugging
```

### 2. Task Processing Flow
```
1. API Request → /trainmodel/
2. Submit to Dramatiq → train_model_task.send()
3. Redis Queue → Task queued
4. Worker Pickup → Dramatiq worker processes
5. Progress Updates → Redis progress tracking
6. Model Training → XGBoost training
7. Result Storage → Redis + File system
8. API Response → /trainmodel/status/{task_id}
```

### 3. 8-Stage Progress Tracking
```python
TaskState.PENDING           # 0%   - Task submitted
TaskState.STARTED           # 5%   - Worker picked up task
TaskState.CONNECTING        # 10%  - Connecting to InfluxDB
TaskState.FETCHING_DATA     # 15%  - Fetching training data
TaskState.VALIDATING_DATA   # 35%  - Data validation
TaskState.TRAINING_MODEL    # 50%  - ML model training
TaskState.SAVING_MODEL      # 90%  - Saving trained model
TaskState.SUCCESS           # 100% - Task completed
```

---

## 🧠 Machine Learning Pipeline

### 1. Feature Engineering
```python
def create_features(df, meter_id, meter_type):
    features = []
    
    # Time-based features
    df['hour'] = df['DateTime'].dt.hour
    df['day_of_week'] = df['DateTime'].dt.dayofweek
    df['month'] = df['DateTime'].dt.month
    df['is_weekend'] = df['day_of_week'].isin([5, 6])
    
    # Temperature features
    df['temp_rolling_mean_24h'] = df['Temperature'].rolling(24).mean()
    
    # Consumption lag features
    df['consumption_lag_24'] = df['Consumption'].shift(24)
    df['consumption_rolling_mean_24h'] = df['Consumption'].rolling(24).mean()
    
    return features
```

### 2. Model Training Process
```python
# Data preprocessing
df = influx_client.get_meter_data(meter_id, meter_type)
features = create_features(df, meter_id, meter_type)

# Model configuration (differs by meter type)
if meter_type == "electricity":
    model = XGBRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42
    )
    scaler = RobustScaler()
else:  # water
    model = XGBRegressor(
        n_estimators=150,
        max_depth=8,
        learning_rate=0.08,
        random_state=42
    )
    scaler = StandardScaler()

# Training and saving
model.fit(X_scaled, y)
model.save_model(f"models/{meter_type}/{meter_id}.h5")
```

### 3. Model Evaluation Metrics
- **R² Score** - Coefficient of determination
- **RMSE** - Root Mean Square Error
- **MAE** - Mean Absolute Error
- **MAPE** - Mean Absolute Percentage Error
- **Accuracy Percentage** - Custom accuracy metric

---

## 🌐 API Endpoints & Functionality

### 1. Core Training API
```http
POST /trainmodel/
Content-Type: application/json

{
  "meter_id": "meter_001",
  "meter_type": "electricity"
}

Response:
{
  "task_id": "uuid-string",
  "status": "PENDING",
  "progress": 0
}
```

```http
GET /trainmodel/status/{task_id}

Response:
{
  "task_id": "uuid-string",
  "status": "SUCCESS",
  "progress": 100,
  "result": {
    "status": "success",
    "message": "Model trained successfully",
    "details": {
      "meter_id": "meter_001",
      "meter_type": "electricity",
      "data_points": 26388,
      "model_path": "models/electricity/meter_001.h5",
      "training_completed": "2025-06-19T13:31:38.466633",
      "training_time_seconds": 512.36
    }
  }
}
```

### 2. Forecasting API
```http
POST /forecast/
Content-Type: application/json

{
  "meter_id": "meter_001",
  "meter_type": "electricity",
  "start_date": "2025-06-20",
  "end_date": "2025-06-25"
}

Response:
{
  "forecast_data": [
    {
      "datetime": "2025-06-20",
      "value": 45.2,
      "type": "historical"
    },
    {
      "datetime": "2025-06-21",
      "value": 48.7,
      "type": "forecast",
      "temperature": 22.5
    }
  ]
}
```

### 3. Testing & Validation API
```http
POST /test_model/
Content-Type: application/json

{
  "meter_id": "meter_001",
  "meter_type": "electricity",
  "test_size": 0.2
}

Response:
{
  "test_results": {
    "meter_type": "electricity",
    "test_samples": 5000,
    "metrics": {
      "r2_score": 0.8954,
      "rmse": 12.45,
      "mae": 8.32,
      "mean_accuracy_percent": 92.1,
      "mape_percent": 7.9
    },
    "performance_summary": {
      "r2_category": "good"
    }
  }
}
```

### 4. Health & Monitoring
```http
GET /health/

Response:
{
  "status": "healthy",
  "timestamp": "2025-06-19T13:30:00",
  "services": {
    "api": "running",
    "redis": "connected",
    "task_system": "dramatiq"
  }
}
```

```http
GET /health/worker

Response:
{
  "worker_status": "running",
  "redis_clients": 3,
  "task_system": "dramatiq",
  "timestamp": "2025-06-19T13:30:00"
}
```

### 5. Location & Weather API
```http
POST /location/
Content-Type: application/json

{
  "latitude": 40.7128,
  "longitude": -74.0060,
  "city": "New York"
}
```

```http
GET /temperature/validate/?days=30
```

```http
POST /temperature/series/
Content-Type: application/json

{
  "start_date": "2025-06-01",
  "end_date": "2025-06-30"
}
```

---

## 🔧 System Configuration & Setup

### 1. Environment Setup
```bash
# Required environment variables
REDIS_URL=redis://localhost:6379/0
DEBUG=false  # Set to true for development

# Docker services
docker run -d -p 6379:6379 --name redis redis
docker run -d -p 8086:8086 --name influxdb influxdb:2.7.3
```

### 2. Installation & Dependencies
```bash
# Install Python dependencies
pip install -r requirements.txt

# Key dependencies:
dramatiq[redis,watch]>=1.17.0
fastapi>=0.104.1
uvicorn[standard]>=0.24.0
redis>=5.0.0
xgboost>=2.0.0
pandas>=2.1.0
numpy>=1.24.0
scikit-learn>=1.3.0
structlog>=23.2.0
colorlog>=6.8.0
```

### 3. Startup Process
```bash
# 1. Start Redis & InfluxDB
docker start redis influxdb

# 2. Start Dramatiq worker
start_dramatiq_worker.bat

# 3. Start FastAPI server  
start_dramatiq_api.bat
```

### 4. Configuration Files

#### `config/influxdb_config.json`
```json
{
  "url": "http://localhost:8086",
  "token": "your-influxdb-token",
  "org": "your-org",
  "bucket": "energy_forecasting"
}
```

#### `config/weather_config.json`
```json
{
  "api_key": "your-openweather-api-key",
  "base_url": "http://api.openweathermap.org/data/2.5"
}
```

### 5. Logging Configuration
```python
# Enhanced logging system
├── Console logging (colored, structured)
├── File logging (detailed, with rotation)
├── Component-specific log levels
├── Debug mode support
└── Task-specific logging with context

# Log levels:
- dramatiq: INFO
- redis: WARNING  
- influxdb_client: INFO
- xgboost: WARNING
- urllib3/requests: WARNING
```

---

## 📈 Performance & Reliability

### 1. Performance Improvements
- **3x faster** task processing (Dramatiq vs Celery)
- **Automatic retries** with exponential backoff
- **Connection pooling** for Redis
- **Efficient progress tracking**

### 2. Reliability Features
- **Dead letter queues** for failed tasks
- **Task timeouts** and limits
- **Graceful error handling**
- **Comprehensive logging**
- **Health monitoring**

### 3. Scalability
- **Horizontal worker scaling** (multiple worker processes)
- **Redis clustering** support
- **Load balancing** across workers
- **Resource monitoring**

### 4. Error Handling
```python
# Error Classification:
CONNECTION_ERROR    # Database/API connection issues
MEMORY_ERROR       # Out of memory during processing
TIMEOUT_ERROR      # Task execution timeout
TRAINING_ERROR     # Model training failures
NO_DATA_FOUND      # No training data available
INSUFFICIENT_DATA  # Not enough data for training
```

---

## 🚀 Deployment & Operations

### 1. Development Mode
```bash
# Enable debug mode
set DEBUG=true

# Start with auto-reload
start_dramatiq_worker.bat  # Worker with auto-reload
start_dramatiq_api.bat     # API with auto-reload
```

### 2. Production Deployment
```bash
# Production environment variables
set DEBUG=false
set REDIS_URL=redis://production-redis:6379/0

# Multiple workers for scalability
start /B python dramatiq_worker.py
start /B python dramatiq_worker.py
start /B python dramatiq_worker.py

# Production API server
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

### 3. Monitoring & Maintenance
- **Health check endpoints** for service monitoring
- **Log rotation** for disk space management
- **Redis memory monitoring**
- **Model performance tracking**
- **Task queue monitoring**

### 4. Backup & Recovery
```bash
# Model backups
models/
├── electricity/
└── water/

# Configuration backups
config/
├── influxdb_config.json
└── weather_config.json

# Log retention (7 days)
logs/energy_forecasting_*.log
```

---

## 🎯 Use Cases & Applications

### 1. Utility Companies
- **Peak demand forecasting** - Predict electricity demand peaks
- **Load balancing optimization** - Optimize grid load distribution
- **Infrastructure planning** - Plan capacity expansions

### 2. Building Management
- **Energy consumption optimization** - Reduce building energy costs
- **HVAC scheduling** - Optimize heating/cooling schedules
- **Cost prediction** - Predict monthly energy expenses

### 3. Smart Cities
- **Grid management** - Intelligent power grid control
- **Renewable energy integration** - Optimize solar/wind integration
- **Demand response programs** - Coordinate city-wide energy usage

### 4. Industrial Facilities
- **Production planning** - Schedule energy-intensive operations
- **Energy cost optimization** - Minimize peak demand charges
- **Predictive maintenance** - Predict equipment energy efficiency

---

## 🔍 Troubleshooting Guide

### 1. Common Issues

#### Redis Connection Problems
```bash
# Check Redis status
docker ps | grep redis

# Restart Redis if needed
docker restart redis

# Test connection
python -c "import redis; r=redis.Redis(); print(r.ping())"
```

#### InfluxDB Connection Issues
```bash
# Check InfluxDB status
docker ps | grep influxdb

# Verify configuration
cat config/influxdb_config.json
```

#### Worker Not Processing Tasks
```bash
# Check worker logs
tail -f logs/energy_forecasting_*.log

# Restart worker
start_dramatiq_worker.bat
```

### 2. Performance Optimization
- **Increase worker processes** for parallel processing
- **Optimize Redis memory** settings
- **Monitor task queue** lengths
- **Use connection pooling** for databases

### 3. Debug Mode
```bash
# Enable detailed logging
set DEBUG=true

# Check task progress
GET /trainmodel/status/{task_id}

# Monitor Redis queues
redis-cli monitor
```

---

## 📚 API Documentation

The complete API documentation is available at:
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

---

## 🔐 Security Considerations

### 1. Authentication
- Implement API key authentication for production
- Use HTTPS for secure communication
- Secure Redis with authentication

### 2. Data Protection
- Encrypt sensitive configuration data
- Use environment variables for secrets
- Implement data access controls

### 3. Network Security
- Use firewalls to restrict access
- Implement rate limiting
- Monitor for suspicious activity

---

## 📞 Support & Maintenance

### 1. Monitoring
- **Health checks**: `/health/` and `/health/worker`
- **Log monitoring**: Check logs directory
- **Performance metrics**: Task processing times

### 2. Maintenance Tasks
- **Log rotation**: Automatic with timestamp-based files
- **Model retraining**: Periodic model updates
- **Database cleanup**: Archive old data

### 3. Upgrades
- **Dependency updates**: Regular package updates
- **Feature additions**: New forecasting capabilities
- **Performance improvements**: Optimization and scaling

---

**System Status**: ✅ Production Ready  
**Last Updated**: June 19, 2025  
**Version**: 2.0 (Dramatiq Implementation) 