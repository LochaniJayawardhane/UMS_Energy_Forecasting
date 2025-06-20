# Background Processing for Energy Forecasting

This document explains how to use the background processing system for training energy forecasting models.

## Overview

The system uses Celery for background task processing and InfluxDB for storing and retrieving time-series data. This approach eliminates the need to send large datasets through API calls, improving performance and scalability.

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Up Redis

Redis is used as the message broker for Celery. You can run it using Docker:

```bash
docker run -d -p 6379:6379 --name redis redis
```

Or install and run it directly on your system.

### 3. Configure InfluxDB

Set the following environment variables for InfluxDB connection:

```bash
export INFLUX_URL=http://localhost:8086
export INFLUX_TOKEN=your_influxdb_token
export INFLUX_ORG=your_organization
export INFLUX_BUCKET=energy_data
```

## Running the System

### 1. Start Redis (if not already running)

```bash
docker start redis
```

### 2. Start Celery Worker

```bash
celery -A celery_worker worker --loglevel=info
```

### 3. Start FastAPI Application

```bash
uvicorn main:app --reload
```

## API Usage

### Start a Training Job

```bash
curl -X POST "http://localhost:8000/trainmodel/v2/" \
     -H "Content-Type: application/json" \
     -d '{
           "meter_id": "meter1",
           "meter_type": "electricity",
           "start_date": "2023-01-01T00:00:00Z",
           "end_date": "2023-12-31T23:59:59Z"
         }'
```

Response:
```json
{
  "task_id": "a1b2c3d4-e5f6-7890-abcd-1234567890ab",
  "status": "PENDING",
  "progress": 0
}
```

### Check Training Status

```bash
curl -X GET "http://localhost:8000/trainmodel/status/a1b2c3d4-e5f6-7890-abcd-1234567890ab"
```

Response (in progress):
```json
{
  "task_id": "a1b2c3d4-e5f6-7890-abcd-1234567890ab",
  "status": "PROCESSING_DATA",
  "progress": 30
}
```

Response (completed):
```json
{
  "task_id": "a1b2c3d4-e5f6-7890-abcd-1234567890ab",
  "status": "SUCCESS",
  "progress": 100,
  "result": {
    "status": "success",
    "message": "Model trained and saved at models/electricity/meter1.h5",
    "details": {
      "meter_id": "meter1",
      "meter_type": "electricity",
      "data_points": 8760,
      "model_path": "models/electricity/meter1.h5",
      "training_completed": "2023-06-01T12:34:56.789012"
    }
  }
}
```

## InfluxDB Data Format

The system expects data in InfluxDB to be structured as follows:

- Measurement: `electricity_consumption` or `water_consumption`
- Tags:
  - `meter_id`: Identifier for the meter
- Fields:
  - `consumption`: Consumption value
  - `temperature`: Temperature value (if available)

Example InfluxDB query:
```
from(bucket: "energy_data")
  |> range(start: 2023-01-01T00:00:00Z, stop: 2023-12-31T23:59:59Z)
  |> filter(fn: (r) => r["_measurement"] == "electricity_consumption")
  |> filter(fn: (r) => r["meter_id"] == "meter1")
  |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
```

## Monitoring

You can monitor Celery tasks using Flower:

```bash
pip install flower
celery -A celery_worker flower
```

Then open http://localhost:5555 in your browser.

## Troubleshooting

1. **Task stuck in PENDING state**: Check if Celery worker is running
2. **InfluxDB connection errors**: Verify environment variables and InfluxDB server status
3. **Redis connection errors**: Ensure Redis is running and accessible 