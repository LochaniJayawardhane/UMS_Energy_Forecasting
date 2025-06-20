from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Dict, Optional
import pandas as pd
import numpy as np
import xgboost as xgb
import os
import json
from datetime import datetime, timedelta
from utils import create_features, train_electricity_model, train_water_model
from weather_utils.location_manager import set_location, get_location
from weather_utils.weather import get_temperature_forecast, validate_temperature_forecast_accuracy, get_temperature_series
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
# Dramatiq imports instead of Celery
from dramatiq_broker import broker
from task_system import train_model_task

import threading
import subprocess
import sys
import psutil
import time

from influx_client import InfluxClient
from logger_config import setup_logging, get_logger, debug_mode_from_env

# Setup logging
debug_mode = debug_mode_from_env()
setup_logging(debug=debug_mode)
logger = get_logger("energy_forecasting.main")



app = FastAPI(title="UMS Forecasting Service (Dramatiq)")

# Ensure model directories exist
def ensure_model_dirs():
    os.makedirs('models/electricity', exist_ok=True)
    os.makedirs('models/water', exist_ok=True)
    os.makedirs('config', exist_ok=True)

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

class TemperatureSeriesRequest(BaseModel):
    start_date: str  # Start date in YYYY-MM-DD format
    end_date: str    # End date in YYYY-MM-DD format

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
    Returns a task ID that can be used to check the status.
    """
    meter_id = request.meter_id
    meter_type = request.meter_type.lower()
    
    # Validate meter type
    if meter_type not in ["electricity", "water"]:
        raise HTTPException(status_code=400, detail="meter_type must be 'electricity' or 'water'")
    
    try:
        logger.info("Training model request received", meter_id=meter_id, meter_type=meter_type)
        
        # Submit task to Dramatiq
        message = train_model_task.send(meter_id, meter_type)
        task_id = message.message_id
        
        logger.info("Training task submitted", task_id=task_id, meter_id=meter_id, meter_type=meter_type)
        
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
    """
    try:
        from task_system import get_task_result
        
        # Get task result using our comprehensive tracker
        result_data = get_task_result(task_id)
        
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

@app.post("/temperature/series/")
def get_temperature_data_series(request: TemperatureSeriesRequest):
    """
    Retrieve historical temperature data for a date range and return a complete series 
    of forecasted temperatures for each date within the range.
    
    For past dates: Returns actual historical temperatures
    For future dates: Returns forecasted temperatures based on 3-year historical averages
    """
    try:
        temperature_series = get_temperature_series(request.start_date, request.end_date)
        
        return {
            "status": "success",
            "start_date": request.start_date,
            "end_date": request.end_date,
            "total_days": len(temperature_series),
            "temperature_series": temperature_series
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving temperature series: {str(e)}")

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
    model_path = f"models/{meter_type}/{meter_id}.h5"
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
    electricity_models_dir = "models/electricity"
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
    water_models_dir = "models/water"
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

@app.get("/health/")
def health_check():
    """Health check endpoint"""
    try:
        # Test Redis connection
        import redis
        r = redis.Redis(host='localhost', port=6379, db=0)
        r.ping()
        
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "services": {
                "api": "running",
                "redis": "connected",
                "task_system": "dramatiq"
            }
        }
    except Exception as e:
        return {
            "status": "unhealthy", 
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

@app.get("/health/worker")
def check_worker_health():
    """Check if Dramatiq workers are running and healthy"""
    try:
        import redis
        r = redis.Redis(host='localhost', port=6379, db=0)
        
        # Check for active workers (simplified check)
        worker_info = r.info('clients')
        
        return {
            "worker_status": "running",
            "redis_clients": worker_info.get('connected_clients', 0),
            "task_system": "dramatiq",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error("Worker health check failed", error=str(e))
        raise HTTPException(status_code=503, detail=f"Worker health check failed: {str(e)}")

