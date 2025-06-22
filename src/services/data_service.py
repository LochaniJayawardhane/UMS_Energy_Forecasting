"""
Data service for InfluxDB operations and historical data retrieval.
"""
import pandas as pd
from typing import Dict, Optional, List
from datetime import datetime, timedelta

from src.influx_client import InfluxClient
from src.logger_config import get_logger

logger = get_logger("energy_forecasting.services.data")

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
        
        logger.info(f"Last recorded date for {meter_type} meter {meter_id}: {last_date.date()}")
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
            logger.warning(f"No data found for meter {meter_id} in the specified date range")
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
        
        logger.info(f"Retrieved {len(result)} historical records for meter {meter_id} (daily averages)")
        return result
        
    except Exception as e:
        logger.error(f"Error reading data from InfluxDB: {str(e)}")
        return {}

def get_training_data(meter_id: str, meter_type: str, days: int = 365) -> pd.DataFrame:
    """
    Get training data for a specific meter from InfluxDB.
    
    Args:
        meter_id: ID of the meter
        meter_type: Type of meter (electricity/water)
        days: Number of days of historical data to retrieve (default: 365)
        
    Returns:
        DataFrame with DateTime, Consumption, and Temperature columns
    """
    try:
        # Initialize InfluxDB client
        influx_client = InfluxClient()
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Format dates for query
        start_date_str = start_date.strftime('%Y-%m-%d')
        end_date_str = end_date.strftime('%Y-%m-%d')
        
        # Get data from InfluxDB
        data = influx_client.get_meter_data(
            meter_id=meter_id,
            meter_type=meter_type,
            start_date=start_date_str,
            end_date=end_date_str,
            include_temperature=True
        )
        
        # Close the InfluxDB client
        influx_client.close()
        
        if data.empty:
            logger.warning(f"No training data found for {meter_type} meter {meter_id}")
            return pd.DataFrame()
            
        logger.info(f"Retrieved {len(data)} training records for {meter_type} meter {meter_id}")
        return data
        
    except Exception as e:
        logger.error(f"Error retrieving training data: {str(e)}")
        return pd.DataFrame()

def get_available_meters(meter_type: Optional[str] = None) -> Dict[str, List[str]]:
    """
    Get a list of available meters from InfluxDB.
    
    Args:
        meter_type: Optional filter for meter type (electricity/water)
        
    Returns:
        Dictionary mapping meter types to lists of meter IDs
    """
    try:
        # Initialize InfluxDB client
        influx_client = InfluxClient()
        
        result = {}
        
        # Get electricity meters if requested or if no type specified
        if meter_type is None or meter_type.lower() == 'electricity':
            electricity_meters = influx_client.get_available_meters('electricity')
            result['electricity'] = electricity_meters
            
        # Get water meters if requested or if no type specified
        if meter_type is None or meter_type.lower() == 'water':
            water_meters = influx_client.get_available_meters('water')
            result['water'] = water_meters
            
        # Close the InfluxDB client
        influx_client.close()
        
        return result
        
    except Exception as e:
        logger.error(f"Error retrieving available meters: {str(e)}")
        return {'electricity': [], 'water': []} 