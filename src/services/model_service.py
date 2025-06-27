"""
Model service for loading, training, testing and evaluating models.
"""
import os
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import xgboost as xgb
from datetime import timedelta

from src.utils import create_features, train_electricity_model, train_water_model
from src.logger_config import get_logger
from src.services.data_service import get_training_data, get_historical_consumption_data, get_last_recorded_date
from config.model_config import get_model_path

logger = get_logger("energy_forecasting.services.model")

def get_project_root():
    """Get the root directory of the project."""
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def ensure_model_dirs():
    """Ensure model directories exist."""
    project_root = get_project_root()
    base_model_path = get_model_path()
    
    # First ensure the base model directory exists
    base_dir = os.path.join(project_root, base_model_path)
    os.makedirs(base_dir, exist_ok=True)
    logger.info(f"Ensuring model base directory exists: {base_dir}")
    
    # Then create meter type subdirectories
    os.makedirs(os.path.join(base_dir, 'electricity'), exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'water'), exist_ok=True)
    logger.debug("Model directories verified")

def get_model_file_path(meter_id: str, meter_type: str) -> str:
    """
    Get the path to a model file.
    
    """
    ensure_model_dirs()
    base_model_path = get_model_path()
    return f"{base_model_path}/{meter_type}/{meter_id}.h5"

def load_model(meter_id: str, meter_type: str) -> Optional[xgb.XGBRegressor]:
    """
    Load a trained model for a specific meter.
    
    """
    model_path = get_model_file_path(meter_id, meter_type)
    
    if not os.path.exists(model_path):
        logger.warning(f"Model not found at {model_path}")
        return None
    
    try:
        model = xgb.XGBRegressor()
        model.load_model(model_path)
        logger.info(f"Model loaded from {model_path}")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return None

def train_model(meter_id: str, meter_type: str) -> Tuple[bool, str, Dict]:
    """
    Train a model for a specific meter using historical data.
    
    """
    try:
        # Get training data
        data = get_training_data(meter_id, meter_type)
        
        if data.empty:
            return False, "No training data available", {}
            
        # Train model based on meter type
        if meter_type == 'electricity':
            model = train_electricity_model(data)
        elif meter_type == 'water':
            model = train_water_model(data)
        else:
            return False, f"Unsupported meter type: {meter_type}", {}
            
        # Save model
        model_path = get_model_file_path(meter_id, meter_type)
        model.save_model(model_path)
        
        # Calculate training metrics
        X = create_features(data, meter_id, meter_type)
        y = data['Consumption']
        y_pred = model.predict(X)
        metrics = evaluate_model_accuracy(y, y_pred, meter_type)
        
        return True, f"Model trained and saved to {model_path}", {
            "metrics": metrics,
            "data_points": len(data),
            "model_path": model_path
        }
        
    except Exception as e:
        logger.error(f"Error training model: {str(e)}")
        return False, f"Error training model: {str(e)}", {}

def evaluate_model_accuracy(y_true, y_pred, meter_type: str) -> Dict:
    """
    Calculate comprehensive evaluation metrics for model predictions.
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
        meter_type: Type of meter (electricity/water)
        
    Returns:
        Dictionary of evaluation metrics
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

def get_available_models() -> Dict[str, List[str]]:
    """
    Get a list of available trained models.
    
    Returns:
        Dictionary mapping meter types to lists of meter IDs with trained models
    """
    # Always ensure directories exist first
    ensure_model_dirs()
    
    project_root = get_project_root()
    base_model_path = get_model_path()
    base_dir = os.path.join(project_root, base_model_path)
    result = {'electricity': [], 'water': []}
    
    # Safely check both directories
    try:
        # Check electricity models
        electricity_dir = os.path.join(base_dir, 'electricity')
        if os.path.exists(electricity_dir) and os.path.isdir(electricity_dir):
            for file in os.listdir(electricity_dir):
                if file.endswith('.h5'):
                    meter_id = file.replace('.h5', '')
                    result['electricity'].append(meter_id)
        else:
            logger.warning(f"Electricity models directory not found at: {electricity_dir}")
                    
        # Check water models
        water_dir = os.path.join(base_dir, 'water')
        if os.path.exists(water_dir) and os.path.isdir(water_dir):
            for file in os.listdir(water_dir):
                if file.endswith('.h5'):
                    meter_id = file.replace('.h5', '')
                    result['water'].append(meter_id)
        else:
            logger.warning(f"Water models directory not found at: {water_dir}")
    except Exception as e:
        logger.error(f"Error accessing model directories: {str(e)}")
                
    return result

def test_model(meter_id: str, meter_type: str, test_size: float = 0.2) -> Tuple[Dict, Optional[str]]:
    """
    Test a trained model using historical data.
    
    Args:
        meter_id: ID of the meter
        meter_type: Type of meter (electricity/water)
        test_size: Proportion of data to use for testing (0-1)
        
    Returns:
        Tuple of (test_results, error_message)
    """
    try:
        # Check if model exists
        model_path = get_model_file_path(meter_id, meter_type)
        if not os.path.exists(model_path):
            return {}, f"No trained model found for {meter_type} meter {meter_id}. Please train a model first."
        
        # Get last recorded date
        last_date = get_last_recorded_date(meter_id, meter_type)
        
        # Generate date range for historical data (90 days)
        end_date = last_date
        start_date = end_date - timedelta(days=90)
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Get historical consumption data
        historical_data = get_historical_consumption_data(meter_id, meter_type, date_range)
        
        if not historical_data:
            return {}, f"No historical data found for {meter_type} meter {meter_id}"
        
        # Import temperature data function
        from weather_utils.weather import get_temperature_series
        
        # Get temperature data for the date range
        temperature_series = get_temperature_series(start_date, end_date)
        
        # Create temperature dictionary for easier lookup
        temperature_dict = {item['date']: item['temperature'] for item in temperature_series}
        
        # Create DataFrame with DateTime, Consumption, and Temperature
        df = pd.DataFrame({
            'DateTime': date_range,
            'Temperature': [temperature_dict.get(str(date.date()), 20.0) for date in date_range],  # Default to 20°C if missing
            'Consumption': [0] * len(date_range)  # Placeholder values
        })
        
        # Fill in actual consumption values
        for i, date in enumerate(date_range):
            date_str = str(date.date())
            if date_str in historical_data:
                df.loc[i, 'Consumption'] = historical_data[date_str]
        
        # Create features
        X = create_features(df, meter_id, meter_type)
        
        # Create target variable
        y = df['Consumption'].values
        
        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        
        # Load model
        model = load_model(meter_id, meter_type)
        if model is None:
            return {}, f"Failed to load model for {meter_type} meter {meter_id}"
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Evaluate model
        metrics = evaluate_model_accuracy(y_test, y_pred, meter_type)
        
        # Add test details
        test_results = {
            "metrics": metrics,
            "test_details": {
                "test_size": test_size,
                "test_samples": len(y_test),
                "date_range": {
                    "start": start_date.strftime("%Y-%m-%d"),
                    "end": end_date.strftime("%Y-%m-%d")
                }
            }
        }
        
        return test_results, None
        
    except Exception as e:
        logger.error(f"Error testing model: {str(e)}")
        return {}, f"Failed to test model: {str(e)}"

def test_all_models(test_size: float = 0.2) -> Dict:
    """
    Test all available trained models and return their accuracy metrics.
    
    Args:
        test_size: Proportion of data to use for testing (0-1)
        
    Returns:
        Dictionary with test results for all models
    """
    try:
        # Get all available models
        available_models = get_available_models()
        
        results = {
            "electricity": [],
            "water": []
        }
        
        # Test electricity models
        for meter_id in available_models['electricity']:
            test_result, error = test_model(meter_id, 'electricity', test_size)
            if error is None:
                results['electricity'].append({
                    "meter_id": meter_id,
                    "results": test_result
                })
            else:
                results['electricity'].append({
                    "meter_id": meter_id,
                    "error": error
                })
                
        # Test water models
        for meter_id in available_models['water']:
            test_result, error = test_model(meter_id, 'water', test_size)
            if error is None:
                results['water'].append({
                    "meter_id": meter_id,
                    "results": test_result
                })
            else:
                results['water'].append({
                    "meter_id": meter_id,
                    "error": error
                })
                
        return {
            "test_size": test_size,
            "timestamp": pd.Timestamp.now().isoformat(),
            "results": results
        }
        
    except Exception as e:
        logger.error(f"Error testing all models: {str(e)}")
        return {
            "error": f"Failed to test all models: {str(e)}",
            "timestamp": pd.Timestamp.now().isoformat()
        } 