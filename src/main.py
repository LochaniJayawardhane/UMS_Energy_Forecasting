from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import pandas as pd
import numpy as np
import xgboost as xgb
import os
import json
import asyncio
from datetime import datetime, timedelta
import sys
import os
import threading
import subprocess
import psutil
import time

# Add parent directory to path to import modules from project root
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Import from src package
from src.utils import create_features, train_electricity_model, train_water_model
from src.dramatiq_broker import broker
from src.task_system import train_model_task
from src.influx_client import InfluxClient
from src.logger_config import setup_logging, get_logger, debug_mode_from_env
# Task cleanup imports removed - using singleton behavior instead

# Import from project root directories
from weather_utils.location_manager import set_location, get_location
from weather_utils.weather import get_temperature_forecast, validate_temperature_forecast_accuracy, get_temperature_series

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

# Setup logging
debug_mode = debug_mode_from_env()
setup_logging(debug=debug_mode)
logger = get_logger("energy_forecasting.main")

# Helper function to get project root directory
def get_project_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

app = FastAPI(title="UMS Forecasting Service (Dramatiq)")

# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Ensure model directories exist
def ensure_model_dirs():
    project_root = get_project_root()
    os.makedirs(os.path.join(project_root, 'models/electricity'), exist_ok=True)
    os.makedirs(os.path.join(project_root, 'models/water'), exist_ok=True)
    os.makedirs(os.path.join(project_root, 'config'), exist_ok=True)

ensure_model_dirs()

class TrainModelRequest(BaseModel):
    meter_id: str
    meter_type: str  # 'electricity' or 'water'

class TaskStatus(BaseModel):
    task_id: str
    status: str
    progress: Optional[int] = None
    result: Optional[Dict] = None
    error: Optional[str] = None
    error_details: Optional[Dict] = None

class ForecastRequest(BaseModel):
    meter_id: str
    meter_type: str  # 'electricity' or 'water'
    start_date: str  # Start date in YYYY-MM-DD format
    end_date: str    # End date in YYYY-MM-DD format

class ForecastResponse(BaseModel):
    forecast_data: List[Dict]  # Each dict: {datetime, value, type, temperature (optional)}

class LocationRequest(BaseModel):
    latitude: float
    longitude: float
    city: str

class ValidationResponse(BaseModel):
    error_metrics: Dict



class ModelTestRequest(BaseModel):
    meter_id: str
    meter_type: str  # 'electricity' or 'water'
    test_size: float = 0.2  # Proportion of data to use for testing (default 20%)

class ModelTestResponse(BaseModel):
    test_results: Dict

@app.post("/location/", status_code=201)
def set_global_location(request: LocationRequest):
    """
    Set the global location for temperature forecasting.
    This location will be used for all meters.
    """
    success = set_location(
        request.latitude,
        request.longitude,
        request.city
    )
    
    if success:
        return {"message": "Global location set successfully"}
    else:
        raise HTTPException(status_code=500, detail="Failed to set global location")

@app.get("/location/")
def get_global_location():
    """
    Get the global location used for temperature forecasting
    """
    location = get_location()
    return location

@app.get("/temperature/validate/")
def validate_temperature_accuracy(days: int = 30):
    """
    Validate the accuracy of temperature forecasting
    """
    error_metrics = validate_temperature_forecast_accuracy(test_period_days=days)
    
    return {
        "error_metrics": error_metrics
    }

@app.post("/trainmodel/", response_model=TaskStatus)
def train_model(request: TrainModelRequest):
    """
    Train a model for a specific meter by fetching all historical data from InfluxDB.
    
    STRICT SINGLETON BEHAVIOR: This endpoint IMMEDIATELY cancels ALL existing training tasks
    and their SSE streams before starting a new one. Only ONE training task can exist at a time.
    
    Returns a task ID that can be used to check the status.
    """
    meter_id = request.meter_id
    meter_type = request.meter_type.lower()
    
    # Validate meter type
    if meter_type not in ["electricity", "water"]:
        raise HTTPException(status_code=400, detail="meter_type must be 'electricity' or 'water'")
    
    import redis
    from src.task_system import task_tracker, TaskState
    
    try:
        logger.info("Training model request received", meter_id=meter_id, meter_type=meter_type)
        
        redis_client = redis.from_url(os.getenv('REDIS_URL', 'redis://localhost:6379/0'))
        active_task_key = "active_training_task"
        
        # STEP 1: IMMEDIATELY CANCEL ALL EXISTING TASKS
        try:
            old_active_task = redis_client.get(active_task_key)
            if old_active_task:
                old_task_id = old_active_task.decode()
                logger.info("CANCELLING previous task immediately", old_task_id=old_task_id)
                
                # Cancel the old task immediately in TaskTracker
                task_tracker.update_progress(old_task_id, 
                    TaskState.CANCELLED, 0, 
                    "Cancelled - New training task started", 
                    {"cancelled_by": "new_task", "reason": "singleton_enforcement"},
                    error="Cancelled by new task", error_type="CANCELLED")
                
                # Broadcast cancellation to all SSE streams for this task
                cancel_message = {
                    "task_id": old_task_id,
                    "status": "CANCELLED", 
                    "progress": 0,
                    "message": "Cancelled - New training task started",
                    "error": "Cancelled by new task",
                    "error_type": "CANCELLED",
                    "updated_at": datetime.now().isoformat()
                }
                redis_client.publish(f"task_updates:{old_task_id}", json.dumps(cancel_message))
                
                logger.info("Previous task cancelled and SSE notified", old_task_id=old_task_id)
                
            # Clear any stale active task marker
            redis_client.delete(active_task_key)
            
        except Exception as e:
            logger.warning("Could not cancel previous tasks", error=str(e))
        
        # STEP 2: Submit the new task
        message = train_model_task.send(meter_id, meter_type)
        task_id = message.message_id
        
        # STEP 3: Set this as the ONLY active task
        try:
            redis_client.setex(active_task_key, 3600, task_id)  # 1 hour expiration
            logger.info("New task set as ONLY active training task", 
                       task_id=task_id, meter_id=meter_id, meter_type=meter_type)
                
        except Exception as e:
            logger.warning("Could not set new active task", task_id=task_id, error=str(e))
        
        # STEP 4: Initialize progress tracking for new task
        task_tracker.update_progress(task_id, 
            TaskState.PENDING, 0, 
            "Task submitted - waiting to start", 
            {"meter_id": meter_id, "meter_type": meter_type})
        
        logger.info("NEW training task submitted and active", task_id=task_id, meter_id=meter_id, meter_type=meter_type)
        
        return TaskStatus(
            task_id=task_id,
            status="PENDING",
            progress=0
        )
        
    except Exception as e:
        logger.error("Failed to submit training task", error=str(e), meter_id=meter_id, meter_type=meter_type)
        raise HTTPException(status_code=500, detail=f"Failed to submit training task: {str(e)}")

@app.get("/trainmodel/status/{task_id}", response_model=TaskStatus)
def get_task_status(task_id: str):
    """
    Get the status of a training task.
    Uses only TaskTracker to avoid triggering task execution.
    """
    try:
        from src.task_system import task_tracker
        
        # Get task status from TaskTracker only (no Dramatiq backend interaction)
        result_data = task_tracker.get_progress(task_id)
        
        if not result_data:
            # Task not found in progress tracking
            return TaskStatus(
                task_id=task_id,
                status="NOT_FOUND",
                progress=0,
                error="Task not found or expired"
            )
        
        # Convert our format to API format
        api_status = result_data.get("state", "PENDING")
        if api_status == "FAILED":
            api_status = "FAILURE"
        
        return TaskStatus(
            task_id=task_id,
            status=api_status,
            progress=result_data.get("progress", 0),
            result=result_data.get("result"),
            error=result_data.get("error"),
            error_details=result_data.get("details")
        )
            
    except Exception as e:
        logger.error("Failed to get task status", task_id=task_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get task status: {str(e)}")

@app.get("/trainmodel/stream/{task_id}")
async def stream_task_status(task_id: str):
    """
    Stream real-time task status updates using Server-Sent Events (SSE).
    
    This endpoint provides real-time updates for training tasks, eliminating
    the need for polling the status endpoint.
    
    Usage:
    - JavaScript: new EventSource('/trainmodel/stream/{task_id}')
    - Python: requests.get('/trainmodel/stream/{task_id}', stream=True)
    """
    try:
        return StreamingResponse(
            generate_task_updates(task_id),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",  # Disable nginx buffering for real-time streaming
            }
        )
    except Exception as e:
        logger.error("Failed to start SSE stream", task_id=task_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to start status stream: {str(e)}")

async def generate_task_updates(task_id: str):
    """
    Generate Server-Sent Events for task status updates.
    
    This function:
    1. Subscribes to Redis pub/sub for real-time updates
    2. Sends initial status if available
    3. Streams updates as they come from Redis
    4. Closes connection when task completes or fails
    """
    import redis
    
    # Initialize Redis client for pub/sub
    try:
        redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
        redis_client = redis.from_url(redis_url)
        pubsub = redis_client.pubsub()
        
        # Subscribe to task-specific channel
        channel = f"task_updates:{task_id}"
        pubsub.subscribe(channel)
        
        logger.info("SSE stream started", task_id=task_id, channel=channel)
        
        # Send initial status if available (only from TaskTracker, not Dramatiq backend)
        try:
            from src.task_system import task_tracker
            initial_status = task_tracker.get_progress(task_id)
            if initial_status:
                # Convert to API format
                api_status = initial_status.get("state", "PENDING")
                if api_status == "FAILED":
                    api_status = "FAILURE"
                
                event_data = {
                    "task_id": task_id,
                    "status": api_status,
                    "progress": initial_status.get("progress", 0),
                    "message": initial_status.get("message", ""),
                    "updated_at": initial_status.get("updated_at"),
                    "estimated_completion": initial_status.get("estimated_completion"),
                    "result": initial_status.get("result"),
                    "error": initial_status.get("error"),
                    "error_details": initial_status.get("details")
                }
                
                yield f"data: {json.dumps(event_data)}\n\n"
                
                # If task is already completed, close the stream
                if api_status in ["SUCCESS", "FAILURE", "CANCELLED"]:
                    logger.info("SSE stream closed - task already completed/cancelled", task_id=task_id, status=api_status)
                    return
            else:
                # If no progress found, check if this task is still active
                redis_client_check = redis.from_url(os.getenv('REDIS_URL', 'redis://localhost:6379/0'))
                current_active_task = redis_client_check.get("active_training_task")
                
                if current_active_task and current_active_task.decode() != task_id:
                    # This task is not active anymore, it was cancelled
                    cancelled_event = {
                        "task_id": task_id,
                        "status": "CANCELLED",
                        "progress": 0,
                        "message": "Task was cancelled by a newer training request",
                        "error": "Cancelled by newer task",
                        "error_type": "CANCELLED",
                        "updated_at": datetime.now().isoformat()
                    }
                    yield f"data: {json.dumps(cancelled_event)}\n\n"
                    logger.info("SSE stream closed - task was cancelled", task_id=task_id)
                    return
                else:
                    # If no progress found, send a simple pending status
                    event_data = {
                        "task_id": task_id,
                        "status": "PENDING",
                        "progress": 0,
                        "message": "Waiting for task to start...",
                        "updated_at": datetime.now().isoformat()
                    }
                    yield f"data: {json.dumps(event_data)}\n\n"
                    
        except Exception as e:
            logger.warning("Failed to send initial status", task_id=task_id, error=str(e))
        
        # Listen for real-time updates
        timeout_seconds = 3600  # 1 hour timeout
        start_time = time.time()
        
        while time.time() - start_time < timeout_seconds:
            try:
                # Check for new messages with timeout
                message = pubsub.get_message(timeout=5.0)
                
                if message and message['type'] == 'message':
                    try:
                        # Parse the update from Redis
                        update_data = json.loads(message['data'])
                        
                        # Convert to API format - handle both "state" and "status" fields
                        api_status = update_data.get("state") or update_data.get("status", "PENDING")
                        if api_status == "FAILED":
                            api_status = "FAILURE"
                        
                        event_data = {
                            "task_id": task_id,
                            "status": api_status,
                            "progress": update_data.get("progress", 0),
                            "message": update_data.get("message", ""),
                            "updated_at": update_data.get("updated_at"),
                            "estimated_completion": update_data.get("estimated_completion"),
                            "result": update_data.get("result"),
                            "error": update_data.get("error"),
                            "error_details": update_data.get("details")
                        }
                        
                        # Send the update
                        yield f"data: {json.dumps(event_data)}\n\n"
                        
                        # Close stream if task completed
                        if api_status in ["SUCCESS", "FAILURE", "CANCELLED"]:
                            logger.info("SSE stream closed - task completed/cancelled", task_id=task_id, status=api_status)
                            break
                            
                    except json.JSONDecodeError as e:
                        logger.warning("Failed to parse Redis message", task_id=task_id, error=str(e))
                        continue
                        
                elif message is None:
                    # Timeout occurred, send heartbeat
                    heartbeat = {
                        "task_id": task_id,
                        "type": "heartbeat",
                        "timestamp": datetime.now().isoformat()
                    }
                    yield f"data: {json.dumps(heartbeat)}\n\n"
                    
                # Small delay to prevent excessive CPU usage
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.error("Error in SSE stream", task_id=task_id, error=str(e))
                # Send error event
                error_event = {
                    "task_id": task_id,
                    "type": "error",
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                }
                yield f"data: {json.dumps(error_event)}\n\n"
                break
        
        # Cleanup
        try:
            pubsub.unsubscribe(channel)
            pubsub.close()
            redis_client.close()
        except Exception as e:
            logger.warning("Error during SSE cleanup", task_id=task_id, error=str(e))
            
        logger.info("SSE stream ended", task_id=task_id)
        
    except Exception as e:
        logger.error("Failed to initialize SSE stream", task_id=task_id, error=str(e))
        # Send error and close
        error_event = {
            "task_id": task_id,
            "type": "error", 
            "error": f"Failed to initialize stream: {str(e)}",
            "timestamp": datetime.now().isoformat()
        }
        yield f"data: {json.dumps(error_event)}\n\n"

def get_last_recorded_date(meter_id: str, meter_type: str) -> pd.Timestamp:
    """
    Get the last recorded date from InfluxDB for a specific meter.
    
    Args:
        meter_id: ID of the meter
        meter_type: Type of meter (electricity/water)
        
    Returns:
        Last recorded date as pandas Timestamp
        
    Raises:
        Exception: If data cannot be found or loaded
    """
    try:
        # Initialize InfluxDB client
        influx_client = InfluxClient()
        
        # Get the last record for this meter
        last_record = influx_client.get_last_record(meter_id, meter_type)
        
        # Close the InfluxDB client
        influx_client.close()
        
        if last_record is None or last_record.empty:
            raise Exception(f"No data found for meter {meter_id}")
        
        # Get the last datetime
        last_date = last_record['DateTime'].iloc[0]
        
        print(f"Last recorded date for {meter_type} meter {meter_id}: {last_date.date()}")
        return last_date
        
    except Exception as e:
        raise Exception(f"Error reading data from InfluxDB: {str(e)}")

def get_historical_consumption_data(meter_id: str, meter_type: str, dates: pd.DatetimeIndex) -> Dict[str, float]:
    """
    Retrieve historical consumption data from InfluxDB for specified dates.
    
    Args:
        meter_id: ID of the meter
        meter_type: Type of meter (electricity/water)
        dates: List of dates to retrieve data for
        
    Returns:
        Dictionary mapping date strings to consumption values
    """
    try:
        # Initialize InfluxDB client
        influx_client = InfluxClient()
        
        # Get start and end dates from the date range
        start_date = min(dates).strftime('%Y-%m-%d')
        end_date = max(dates).strftime('%Y-%m-%d')
        
        # Fetch data from InfluxDB for the date range
        data = influx_client.get_meter_data(
            meter_id=meter_id,
            meter_type=meter_type,
            start_date=start_date,
            end_date=end_date
        )
        
        # Close the InfluxDB client
        influx_client.close()
        
        if data.empty:
            print(f"No data found for meter {meter_id} in the specified date range")
            return {}
        
        # Group records by date and calculate daily averages
        data['date'] = data['DateTime'].dt.date
        daily_data = data.groupby('date')['Consumption'].mean().to_dict()
        
        # Convert date objects to strings
        historical_data = {str(date): value for date, value in daily_data.items()}
        
        # Filter for requested dates only
        result = {}
        for date in dates:
            date_str = str(date.date())
            if date_str in historical_data:
                result[date_str] = historical_data[date_str]
        
        print(f"Retrieved {len(result)} historical records for meter {meter_id} (daily averages)")
        return result
        
    except Exception as e:
        print(f"Error reading data from InfluxDB: {str(e)}")
        return {}

@app.post("/forecast/", response_model=ForecastResponse)
def forecast(request: ForecastRequest):
    meter_id = request.meter_id
    meter_type = request.meter_type.lower()
    
    # Parse and validate dates
    try:
        start_date = pd.to_datetime(request.start_date)
        end_date = pd.to_datetime(request.end_date)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid date format. Use YYYY-MM-DD format. Error: {str(e)}")
    
    if start_date > end_date:
        raise HTTPException(status_code=400, detail="start_date must be before or equal to end_date")
    
    # Generate date range (daily frequency)
    datetimes = pd.date_range(start=start_date, end=end_date, freq='D')
    
    if meter_type not in ["electricity", "water"]:
        raise HTTPException(status_code=400, detail="meter_type must be 'electricity' or 'water'")
    
    model_path = f"models/{meter_type}/{meter_id}.h5"
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail=f"Model for meter {meter_id} not found.")
    
    # Get the last recorded date from InfluxDB
    try:
        last_recorded_date = get_last_recorded_date(meter_id, meter_type)
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Cannot determine last recorded date: {str(e)}")
    
    # Ensure timezone consistency for datetime comparison
    if last_recorded_date.tz is not None:
        last_recorded_date = last_recorded_date.tz_localize(None)

    # Separate historical and forecast dates based on last recorded date
    historical_dates = datetimes[datetimes <= last_recorded_date]
    forecast_dates = datetimes[datetimes > last_recorded_date]
    
    forecast_data = []
    
    # Handle historical dates - return actual data from InfluxDB
    if len(historical_dates) > 0:
        try:
            # Get historical data from InfluxDB
            historical_data = get_historical_consumption_data(meter_id, meter_type, historical_dates)
             
            for dt in historical_dates:
                dt_str = str(dt.date())  # Format as YYYY-MM-DD
                if dt_str in historical_data:
                    forecast_data.append({
                        "datetime": dt_str,
                        "value": float(historical_data[dt_str]),
                        "type": "historical"
                        # No temperature for historical data
                    })
                else:
                    # This shouldn't happen if the date is within available data range
                    print(f"Warning: No data found for {dt_str} despite being within historical range")
                    forecast_data.append({
                        "datetime": dt_str,
                        "value": 0.0,  # Or some default value
                        "type": "missing"
                    })
        except Exception as e:
            print(f"Error getting historical data: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error retrieving historical data: {str(e)}")
    
    # Handle forecast dates - return forecasted data with temperature
    if len(forecast_dates) > 0:
        try:
            # Load model
            model = xgb.XGBRegressor()
            model.load_model(model_path)
            
            # Get temperature forecasts for future dates only
            temperatures = get_temperature_forecast(forecast_dates)
            
            # Create dataframe with temperature values
            df_future = pd.DataFrame({
                "DateTime": forecast_dates,
                "Temperature": temperatures,
                "Consumption": [np.nan]*len(forecast_dates)
            })
            
            # Create features and generate predictions
            features = create_features(df_future, meter_id, meter_type)
            preds = model.predict(features)
            
            # Add future forecasts to response
            for dt, pred, temp in zip(forecast_dates, preds, temperatures):
                forecast_data.append({
                    "datetime": str(dt.date()),  # Format as YYYY-MM-DD
                    "value": float(pred),
                    "type": "forecast",
                    "temperature": float(temp)  # Include temperature only for future forecasts
                })
                
        except Exception as e:
            print(f"Error generating forecasts: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error generating forecasts: {str(e)}")
    
    # Sort by datetime to maintain chronological order
    forecast_data.sort(key=lambda x: x["datetime"])
    
    return ForecastResponse(forecast_data=forecast_data)



def evaluate_model_accuracy(y_true, y_pred, meter_type: str) -> Dict:
    """
    Calculate comprehensive evaluation metrics for model predictions.
    Similar to the evaluate_xgboost function but adapted for API usage.
    """
    # Calculate standard metrics
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    # Calculate accuracy as percentage with handling for zero values
    epsilon = 1e-10  # Small constant to avoid division by zero
    relative_error = np.abs((y_true - y_pred) / (y_true + epsilon))
    accuracy = (1 - relative_error) * 100
    mean_accuracy = np.mean(accuracy)

    # Alternative accuracy calculation (based on relative difference)
    accuracy_alt = (1 - rmse/np.mean(y_true)) * 100 if np.mean(y_true) != 0 else 0

    # Calculate MAPE (Mean Absolute Percentage Error)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100

    # Calculate additional statistics
    prediction_std = np.std(y_pred)
    actual_std = np.std(y_true)
    prediction_mean = np.mean(y_pred)
    actual_mean = np.mean(y_true)

    return {
        "meter_type": meter_type,
        "test_samples": len(y_true),
        "metrics": {
            "mse": float(mse),
            "rmse": float(rmse),
            "mae": float(mae),
            "r2_score": float(r2),
            "mean_accuracy_percent": float(mean_accuracy),
            "alternative_accuracy_percent": float(accuracy_alt),
            "mape_percent": float(mape)
        },
        "statistics": {
            "actual_mean": float(actual_mean),
            "predicted_mean": float(prediction_mean),
            "actual_std": float(actual_std),
            "predicted_std": float(prediction_std),
            "mean_difference": float(prediction_mean - actual_mean),
            "std_difference": float(prediction_std - actual_std)
        },
        "performance_summary": {
            "excellent": float(r2) > 0.9,
            "good": 0.7 < float(r2) <= 0.9,
            "fair": 0.5 < float(r2) <= 0.7,
            "poor": float(r2) <= 0.5,
            "r2_category": "excellent" if float(r2) > 0.9 else "good" if 0.7 < float(r2) <= 0.9 else "fair" if 0.5 < float(r2) <= 0.7 else "poor"
        }
    }

@app.post("/test_model/", response_model=ModelTestResponse)
def test_model_accuracy(request: ModelTestRequest):
    """
    Test the accuracy of a trained model using a portion of the data from InfluxDB.
    This function loads the data from InfluxDB, splits it into train/test sets,
    and evaluates the model's performance on the test set.
    """
    print(f"🔍 Starting test_model_accuracy function")
    print(f"Request: meter_id={request.meter_id}, meter_type={request.meter_type}, test_size={request.test_size}")
    
    meter_id = request.meter_id
    meter_type = request.meter_type.lower()
    test_size = request.test_size

    print(f"Processed inputs: meter_id={meter_id}, meter_type={meter_type}, test_size={test_size}")

    # Validate inputs
    if meter_type not in ["electricity", "water"]:
        raise HTTPException(status_code=400, detail="meter_type must be 'electricity' or 'water'")
    
    if not (0.1 <= test_size <= 0.5):
        raise HTTPException(status_code=400, detail="test_size must be between 0.1 and 0.5")

    print("✅ Input validation passed")

    # Check if model exists
    project_root = get_project_root()
    model_path = os.path.join(project_root, f"models/{meter_type}/{meter_id}.h5")
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail=f"Model for {meter_type} meter {meter_id} not found. Please train the model first.")

    print(f"✅ Model found at: {model_path}")

    try:
        print(f"Loading data from InfluxDB for {meter_type} meter {meter_id}...")
        
        # Initialize InfluxDB client
        influx_client = InfluxClient()
        
        # Fetch all data for this meter from InfluxDB
        df = influx_client.get_meter_data(
            meter_id=meter_id,
            meter_type=meter_type
            # No date parameters - will fetch all available data
        )
        
        # Close the InfluxDB client
        influx_client.close()
        
        if df.empty:
            raise HTTPException(status_code=404, detail=f"No data found for meter {meter_id}")
        
        print(f"Loaded {len(df)} records for {meter_type} meter {meter_id}")
        print(f"Testing model accuracy with {test_size*100}% of data as test set")
        print(f"DataFrame columns: {list(df.columns)}")
        print(f"DataFrame shape: {df.shape}")
        
        print("Starting preprocessing...")
        # Preprocessing - same as in training
        df = df.sort_values('DateTime')
        df = df.set_index('DateTime')
        
        print("Handling temperature values...")
        # Handle missing or zero temperature values
        df['Temperature'] = df['Temperature'].replace(0, np.nan)
        df['Temperature'] = df['Temperature'].interpolate(method='time')
        
        # Reset index to keep DateTime as a column for feature engineering
        df = df.reset_index()
        
        print("Loading model...")
        # Load the trained model first to check what features it expects
        model = xgb.XGBRegressor()
        model.load_model(model_path)
        
        print("Creating features...")
        # Feature engineering - same as in training
        try:
            features = create_features(df, meter_id=meter_id, meter_type=meter_type)
            print("✅ Feature creation successful")
        except Exception as e:
            print(f"❌ Error in feature creation: {e}")
            raise e
        
        print(f"Generated features: {list(features.columns)}")
        print(f"Features shape: {features.shape}")
        
        print("Checking model feature requirements...")
        # Check what features the model actually expects
        try:
            model_feature_names = model.feature_names_in_ if hasattr(model, 'feature_names_in_') else None
            model_n_features = model.n_features_in_ if hasattr(model, 'n_features_in_') else len(features.columns)
            print(f"Model expects features: {model_feature_names}")
            print(f"Model n_features: {model_n_features}")
        except Exception as e:
            print(f"Error getting model feature info: {e}")
            model_feature_names = None
            model_n_features = len(features.columns)
        
        print("Adapting features...")
        # Adapt features to match what the model expects
        if model_feature_names is not None:
            try:
                # If model has feature names, ensure our features match exactly
                expected_features = list(model_feature_names)
                current_features = list(features.columns)
                
                print(f"Expected features: {expected_features}")
                print(f"Current features: {current_features}")
                
                # Check if we have all expected features
                missing_features = set(expected_features) - set(current_features)
                extra_features = set(current_features) - set(expected_features)
                
                if missing_features:
                    print(f"Warning: Missing features that model expects: {missing_features}")
                    # Add missing features with default values
                    for feat in missing_features:
                        if feat == 'consumption_rolling_mean_24h':
                            # Use the same default as consumption_lag_24
                            if 'consumption_lag_24' in features.columns:
                                features[feat] = features['consumption_lag_24'].copy()
                            else:
                                features[feat] = 0.01
                        else:
                            features[feat] = 0.0
                
                if extra_features:
                    print(f"Warning: Extra features that model doesn't expect: {extra_features}")
                    # Remove extra features
                    features = features.drop(columns=list(extra_features))
                
                # Reorder features to match model expectations
                features = features[expected_features]
                
            except Exception as e:
                print(f"Error in feature adaptation: {e}")
                print(f"Using features as-is: {list(features.columns)}")
        else:
            print("Model has no feature names - using features as generated")
        
        print("Applying scaling...")
        # Apply the same scaling as used in training
        if meter_type == "electricity":
            from sklearn.preprocessing import RobustScaler
            scaler = RobustScaler()
            numerical_cols = ['temp_rolling_mean_24h', 'consumption_lag_24', 'consumption_rolling_mean_24h']
        else:  # water
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            numerical_cols = ['temp_rolling_mean_24h', 'consumption_lag_24', 'consumption_rolling_mean_24h']
        
        # Apply scaling only to existing columns
        existing_numerical_cols = [col for col in numerical_cols if col in features.columns]
        if existing_numerical_cols:
            print(f"Scaling features: {existing_numerical_cols}")
            features[existing_numerical_cols] = scaler.fit_transform(features[existing_numerical_cols])
            print(f"✅ Scaling completed")
        
        X = features
        y = df['Consumption']
        
        print("Splitting data...")
        # Split the data using the same random state as training for consistency
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        
        print(f"Final test features provided: {list(X_test.columns)}")
        print(f"Final test features shape: {X_test.shape}")
        
        print("Making predictions...")
        # Make predictions on test set
        y_pred = model.predict(X_test)
        
        # Calculate evaluation metrics
        test_results = evaluate_model_accuracy(y_test, y_pred, meter_type)
        
        # Add additional information
        test_results["model_info"] = {
            "meter_id": meter_id,
            "meter_type": meter_type,
            "model_path": model_path,
            "total_training_samples": len(df),
            "training_samples_used": len(X_train),
            "test_samples": len(X_test),
            "test_size_ratio": test_size
        }
        
        print(f"\n{meter_type.title()} Model Test Results for Meter {meter_id}:")
        print(f"R² Score: {test_results['metrics']['r2_score']:.4f}")
        print(f"RMSE: {test_results['metrics']['rmse']:.4f}")
        print(f"Mean Accuracy: {test_results['metrics']['mean_accuracy_percent']:.2f}%")
        print(f"Performance Category: {test_results['performance_summary']['r2_category']}")
        
        return ModelTestResponse(test_results=test_results)
        
    except Exception as e:
        print(f"Error during model testing: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error during model testing: {str(e)}")

@app.post("/test_all_models/")
def test_all_available_models(test_size: float = 0.2):
    """
    Test accuracy for all available trained models.
    Returns a comprehensive report of all models' performance.
    """
    if not (0.1 <= test_size <= 0.5):
        raise HTTPException(status_code=400, detail="test_size must be between 0.1 and 0.5")
    
    results = {}
    
    # Find all electricity models
    project_root = get_project_root()
    electricity_models_dir = os.path.join(project_root, "models/electricity")
    if os.path.exists(electricity_models_dir):
        for model_file in os.listdir(electricity_models_dir):
            if model_file.endswith('.h5'):
                meter_id = model_file.replace('.h5', '')
                try:
                    request = ModelTestRequest(
                        meter_id=meter_id,
                        meter_type="electricity",
                        test_size=test_size
                    )
                    test_result = test_model_accuracy(request)
                    results[f"electricity_{meter_id}"] = test_result.test_results
                except Exception as e:
                    results[f"electricity_{meter_id}"] = {"error": str(e)}
    
    # Find all water models
    water_models_dir = os.path.join(project_root, "models/water")
    if os.path.exists(water_models_dir):
        for model_file in os.listdir(water_models_dir):
            if model_file.endswith('.h5'):
                meter_id = model_file.replace('.h5', '')
                try:
                    request = ModelTestRequest(
                        meter_id=meter_id,
                        meter_type="water",
                        test_size=test_size
                    )
                    test_result = test_model_accuracy(request)
                    results[f"water_{meter_id}"] = test_result.test_results
                except Exception as e:
                    results[f"water_{meter_id}"] = {"error": str(e)}
    
    if not results:
        raise HTTPException(status_code=404, detail="No trained models found to test")
    
    # Generate summary statistics
    successful_tests = [r for r in results.values() if "error" not in r]
    if successful_tests:
        avg_r2 = np.mean([r["metrics"]["r2_score"] for r in successful_tests])
        avg_accuracy = np.mean([r["metrics"]["mean_accuracy_percent"] for r in successful_tests])
        
        summary = {
            "total_models_tested": len(results),
            "successful_tests": len(successful_tests),
            "failed_tests": len(results) - len(successful_tests),
            "average_r2_score": float(avg_r2),
            "average_accuracy_percent": float(avg_accuracy),
            "test_size_used": test_size
        }
        
        results["summary"] = summary
    
    return {"model_test_results": results}

@app.get("/")
def health_check():
    """Simple health check endpoint"""
    return {
        "status": "healthy",
        "service": "UMS Forecasting Service",
        "timestamp": datetime.now().isoformat()
    }



@app.get("/trainmodel/active_task")
def get_active_task():
    """
    DEBUG ENDPOINT: Get the currently active training task.
    This helps verify singleton behavior is working correctly.
    """
    try:
        import redis
        redis_client = redis.from_url(os.getenv('REDIS_URL', 'redis://localhost:6379/0'))
        
        active_task_key = "active_training_task"
        active_task_id = redis_client.get(active_task_key)
        
        if active_task_id:
            task_id = active_task_id.decode()
            # Get task status
            from src.task_system import task_tracker
            status = task_tracker.get_progress(task_id)
            
            return {
                "active_task_id": task_id,
                "status": status,
                "message": "Active training task found"
            }
        else:
            return {
                "active_task_id": None,
                "status": None,
                "message": "No active training task"
            }
            
    except Exception as e:
        logger.error("Failed to get active task", error=str(e))
        return {
            "error": str(e),
            "message": "Failed to get active task info"
        }

