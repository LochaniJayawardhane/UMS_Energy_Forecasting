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

def test_model(meter_id: str, meter_type: str, test_size: float = 0.2, latitude: float = None, longitude: float = None, city: str = None) -> Tuple[Dict, Optional[str]]:
    """
    Test a trained model using historical data.
    
    Args:
        meter_id: ID of the meter
        meter_type: Type of meter (electricity/water)
        test_size: Proportion of data to use for testing (0-1)
        latitude: Location latitude (required for temperature data)
        longitude: Location longitude (required for temperature data)
        city: Location city name (required for temperature data)
        
    Returns:
        Tuple of (test_results, error_message)
    """
    try:
        # Validate location parameters
        if latitude is None or longitude is None or city is None:
            return {}, "Location is required. Please provide latitude, longitude, and city."
            
        # Check if model exists
        model_path = get_model_file_path(meter_id, meter_type)
        if not os.path.exists(model_path):
            return {}, f"No trained model found for {meter_type} meter {meter_id}. Please train a model first."
        
        logger.info(f"Testing model for {meter_type} meter {meter_id}")
        
        # Initialize InfluxDB client to fetch ALL available data
        from src.influx_client import InfluxClient
        influx_client = InfluxClient()
        
        # Fetch all data for this meter from InfluxDB (no date constraints)
        df = influx_client.get_meter_data(
            meter_id=meter_id,
            meter_type=meter_type
            # No date parameters - will fetch all available data
        )
        
        # Close the InfluxDB client
        influx_client.close()
        
        if df.empty:
            return {}, f"No data found for {meter_type} meter {meter_id}"
        
        logger.info(f"Loaded {len(df)} records for {meter_type} meter {meter_id}")
        
        # Preprocessing - same as in training
        df = df.sort_values('DateTime')
        
        # Handle missing or zero temperature values
        # Set DateTime as index for time-based interpolation
        df = df.set_index('DateTime')
        df['Temperature'] = df['Temperature'].replace(0, np.nan)
        df['Temperature'] = df['Temperature'].interpolate(method='time')
        # Reset index to keep DateTime as a column
        df = df.reset_index()
        
        # Load the trained model first to check what features it expects
        model = load_model(meter_id, meter_type)
        if model is None:
            return {}, f"Failed to load model for {meter_type} meter {meter_id}"
        
        # Feature engineering
        features = create_features(df, meter_id=meter_id, meter_type=meter_type)
        
        # Check what features the model actually expects
        model_feature_names = model.feature_names_in_ if hasattr(model, 'feature_names_in_') else None
        model_n_features = model.n_features_in_ if hasattr(model, 'n_features_in_') else len(features.columns)
        
        logger.info(f"Model expects {model_n_features} features")
        if model_feature_names is not None:
            logger.info(f"Model feature names: {model_feature_names}")
        
        # Adapt features to match what the model expects
        if model_feature_names is not None:
            # If model has feature names, ensure our features match exactly
            expected_features = list(model_feature_names)
            current_features = list(features.columns)
            
            # Check if we have all expected features
            missing_features = set(expected_features) - set(current_features)
            extra_features = set(current_features) - set(expected_features)
            
            if missing_features:
                logger.warning(f"Missing features that model expects: {missing_features}")
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
                logger.warning(f"Extra features that model doesn't expect: {extra_features}")
                # Remove extra features
                features = features.drop(columns=list(extra_features))
            
            # Reorder features to match model expectations
            features = features[expected_features]
        
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
            logger.info(f"Scaling features: {existing_numerical_cols}")
            features[existing_numerical_cols] = scaler.fit_transform(features[existing_numerical_cols])
        
        X = features
        y = df['Consumption']
        
        # Split the data using the same random state as training for consistency
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        
        logger.info(f"Split data: {len(X_train)} training samples, {len(X_test)} testing samples")
        
        # Make predictions on test set
        y_pred = model.predict(X_test)
        
        # Calculate evaluation metrics
        metrics = evaluate_model_accuracy(y_test, y_pred, meter_type)
        
        # Add test details
        test_results = {
            "metrics": metrics,
            "test_details": {
                "test_size": test_size,
                "test_samples": len(X_test),
                "total_samples": len(df),
                "model_info": {
                    "meter_id": meter_id,
                    "meter_type": meter_type,
                    "model_path": model_path,
                    "total_training_samples": len(df),
                    "training_samples_used": len(X_train),
                    "test_samples": len(X_test),
                    "test_size_ratio": test_size
                }
            }
        }
        
        return test_results, None
        
    except Exception as e:
        logger.error(f"Error testing model: {str(e)}")
        return {}, f"Failed to test model: {str(e)}"

def test_all_models(test_size: float = 0.2, latitude: float = None, longitude: float = None, city: str = None) -> Dict:
    """
    Test all available trained models and return their accuracy metrics.
    
    Args:
        test_size: Proportion of data to use for testing (0-1)
        latitude: Location latitude (required for temperature data)
        longitude: Location longitude (required for temperature data)
        city: Location city name (required for temperature data)
        
    Returns:
        Dictionary with test results for all models
    """
    try:
        # Validate location parameters
        if latitude is None or longitude is None or city is None:
            return {
                "error": "Location is required. Please provide latitude, longitude, and city.",
                "timestamp": pd.Timestamp.now().isoformat()
            }
            
        # Get all available models
        available_models = get_available_models()
        
        results = {
            "electricity": [],
            "water": []
        }
        
        # Test electricity models
        for meter_id in available_models['electricity']:
            test_result, error = test_model(meter_id, 'electricity', test_size, latitude, longitude, city)
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
            test_result, error = test_model(meter_id, 'water', test_size, latitude, longitude, city)
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