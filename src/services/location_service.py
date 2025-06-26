"""
Location service for handling location and weather operations.
"""
from typing import Dict, Optional
from datetime import datetime, timedelta

from weather_utils.location_manager import set_location, get_location
from weather_utils.weather import validate_temperature_forecast_accuracy, get_temperature_series
from src.logger_config import get_logger

logger = get_logger("energy_forecasting.services.location")

def set_global_location(latitude: float, longitude: float, city: str) -> bool:
    """
    Set the global location for temperature forecasting.
    
    """
    try:
        success = set_location(latitude, longitude, city)
        if success:
            logger.info(f"Global location set to {city} ({latitude}, {longitude})")
        else:
            logger.error(f"Failed to set global location to {city} ({latitude}, {longitude})")
        return success
    except Exception as e:
        logger.error(f"Error setting global location: {str(e)}")
        return False

def get_global_location() -> Dict:
    """
    Get the global location used for temperature forecasting.
    
    """
    try:
        location = get_location()
        return location
    except Exception as e:
        logger.error(f"Error getting global location: {str(e)}")
        return {"error": str(e)}

def validate_temperature_accuracy(days: int = 30, latitude: float = None, longitude: float = None, city: str = None) -> Dict:
    """
    Validate the accuracy of temperature forecasting.

    """
    try:
        if latitude is None or longitude is None or city is None:
            return {"error": "Location is required. Please provide latitude, longitude, and city."}
            
        location = {"lat": latitude, "lon": longitude, "city": city}
        error_metrics = validate_temperature_forecast_accuracy(location=location, test_period_days=days)
        return error_metrics
    except Exception as e:
        logger.error(f"Error validating temperature accuracy: {str(e)}")
        return {"error": str(e)}

def get_temperature_data(start_date: str, end_date: str, latitude: float = None, longitude: float = None, city: str = None) -> Dict:
    """
    Get temperature data for a date range.
    """
    try:
        if latitude is None or longitude is None or city is None:
            return {"error": "Location is required. Please provide latitude, longitude, and city."}
            
        # Convert string dates to datetime objects if needed
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, "%Y-%m-%d")
        if isinstance(end_date, str):
            end_date = datetime.strptime(end_date, "%Y-%m-%d")
            
        location = {"lat": latitude, "lon": longitude, "city": city}
        temperature_series = get_temperature_series(start_date, end_date, location)
        return {
            "temperature_data": temperature_series,
            "start_date": start_date.strftime("%Y-%m-%d"),
            "end_date": end_date.strftime("%Y-%m-%d"),
            "days": len(temperature_series)
        }
    except Exception as e:
        logger.error(f"Error getting temperature data: {str(e)}")
        return {"error": str(e)} 