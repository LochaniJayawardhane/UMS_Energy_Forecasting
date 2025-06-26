"""
Forecast service for generating consumption forecasts.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime

from src.utils import create_features
from src.logger_config import get_logger
from src.services.data_service import get_last_recorded_date, get_historical_consumption_data
from src.services.model_service import load_model
from weather_utils.weather import get_temperature_forecast

logger = get_logger("energy_forecasting.services.forecast")

def generate_forecast(
    meter_id: str, 
    meter_type: str, 
    start_date: str, 
    end_date: str,
    latitude: float,
    longitude: float,
    city: str
) -> Tuple[List[Dict], Optional[str]]:
    """
    Generate a consumption forecast for a specific meter.
    
    Args:
        meter_id: ID of the meter
        meter_type: Type of meter (electricity/water)
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        latitude: Location latitude
        longitude: Location longitude
        city: Location city name
        
    Returns:
        Tuple of (forecast_data, error_message)
        - forecast_data: List of dictionaries with forecast data
        - error_message: Error message if any, None if successful
    """
    try:
        # Create location dictionary
        location = {"lat": latitude, "lon": longitude, "city": city}
        
        # Parse dates
        start_date_dt = pd.to_datetime(start_date)
        end_date_dt = pd.to_datetime(end_date)
        
        # Generate date range (daily frequency)
        datetimes = pd.date_range(start=start_date_dt, end=end_date_dt, freq='D')
        
        # Load model
        model = load_model(meter_id, meter_type)
        if model is None:
            return [], f"Model for {meter_type} meter {meter_id} not found"
        
        # Get the last recorded date from InfluxDB
        try:
            last_recorded_date = get_last_recorded_date(meter_id, meter_type)
            
            # Ensure timezone consistency for datetime comparison
            if last_recorded_date.tz is not None:
                last_recorded_date = last_recorded_date.tz_localize(None)
        except Exception as e:
            return [], f"Cannot determine last recorded date: {str(e)}"
        
        # Separate historical and forecast dates based on last recorded date
        historical_dates = datetimes[datetimes <= last_recorded_date]
        forecast_dates = datetimes[datetimes > last_recorded_date]
        
        forecast_data = []
        
        # Handle historical dates - return actual data from InfluxDB
        if len(historical_dates) > 0:
            try:
                # Get historical data from InfluxDB
                historical_data = get_historical_consumption_data(meter_id, meter_type, historical_dates)
                
                # Check if no data was found for any of the requested historical dates
                if not historical_data or all(dt_str not in historical_data for dt_str in [str(dt.date()) for dt in historical_dates]):
                    return [], f"No historical data is available for {meter_type} meter {meter_id} between {historical_dates[0].date()} and {historical_dates[-1].date()}"
                
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
                        logger.warning(f"No data found for {dt_str} despite being within historical range")
                        forecast_data.append({
                            "datetime": dt_str,
                            "value": 0.0,  # Or some default value
                            "type": "missing"
                        })
            except Exception as e:
                logger.error(f"Error getting historical data: {str(e)}")
                return [], f"Error retrieving historical data: {str(e)}"
        
        # Handle forecast dates - return forecasted data with temperature
        if len(forecast_dates) > 0:
            try:
                # Get temperature forecasts for future dates only, passing location
                temperatures = get_temperature_forecast(forecast_dates, location)
                
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
                logger.error(f"Error generating forecasts: {str(e)}")
                return [], f"Error generating forecasts: {str(e)}"
        
        # Sort by datetime to maintain chronological order
        forecast_data.sort(key=lambda x: x["datetime"])
        
        return forecast_data, None
        
    except Exception as e:
        logger.error(f"Error in forecast generation: {str(e)}")
        return [], f"Failed to generate forecast: {str(e)}" 