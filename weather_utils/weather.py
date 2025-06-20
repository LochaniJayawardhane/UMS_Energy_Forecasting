import os
import requests
import json
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from weather_utils.location_manager import get_location

# Load weather configuration
def load_weather_config():
    """Load weather API configuration from file"""
    config_path = 'config/weather_config.json'
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        raise Exception(f"Cannot load weather_config.json. Please ensure the file exists and is valid. Error: {str(e)}")

# Initialize configuration
WEATHER_CONFIG = load_weather_config()
VISUAL_CROSSING_API_KEY = os.getenv("VISUAL_CROSSING_API_KEY", WEATHER_CONFIG["visual_crossing"]["api_key"])
BASE_URL = WEATHER_CONFIG["visual_crossing"]["base_url"]

def get_temperature_series(start_date, end_date):
    """
    Retrieve historical temperature data for a date range and return a complete series 
    of forecasted temperatures for each date within the range.
    
    This is the main function for getting temperature data for date ranges.
    For past dates: Returns actual historical temperatures
    For future dates: Returns forecasted temperatures based on 3-year historical averages
    
    Args:
        start_date: Start date (string in YYYY-MM-DD format or datetime object)
        end_date: End date (string in YYYY-MM-DD format or datetime object)
        
    Returns:
        List of dictionaries with date and temperature for each day in the range:
        [
            {"date": "2025-06-01", "temperature": 24.5},
            {"date": "2025-06-02", "temperature": 25.1},
            ...
        ]
        
    Raises:
        Exception: If temperature data cannot be retrieved
    """
    # Convert string dates to datetime objects if needed
    if isinstance(start_date, str):
        start_date = datetime.strptime(start_date, "%Y-%m-%d")
    if isinstance(end_date, str):
        end_date = datetime.strptime(end_date, "%Y-%m-%d")
    
    # Generate complete date range
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    
    print(f"Retrieving temperature series for {len(date_range)} dates from {start_date.date()} to {end_date.date()}")
    
    # Get temperatures for the entire range
    temperatures = get_temperature_forecast(date_range)
    
    # Build result series
    temperature_series = []
    for i, date in enumerate(date_range):
        temperature_series.append({
            "date": date.strftime("%Y-%m-%d"),
            "temperature": temperatures[i]
        })
    
    print(f"Successfully generated temperature series with {len(temperature_series)} data points")
    return temperature_series

def get_temperature_forecast(dates):
    """
    Get temperature data for requested dates using Visual Crossing Weather API.
    
    For past dates: Returns actual historical temperatures
    For future dates: Calculates average temperatures from same calendar dates over past 3 years
    
    Args:
        dates: pandas DatetimeIndex, list of datetime objects, or single datetime
        
    Returns:
        List of temperature values (or single value if single date input)
        
    Raises:
        Exception: If temperature data cannot be retrieved
    """
    # Get global location
    location = get_location()
    
    # Validate API key
    if VISUAL_CROSSING_API_KEY == "your_visual_crossing_api_key_here":
        raise Exception("Visual Crossing API key not configured. Please set VISUAL_CROSSING_API_KEY environment variable or update config/weather_config.json")
    
    # Handle single date input
    single_date_input = False
    if isinstance(dates, (datetime, pd.Timestamp)):
        dates = [dates]
        single_date_input = True
    
    # Convert to pandas DatetimeIndex if not already
    if not isinstance(dates, pd.DatetimeIndex):
        dates = pd.DatetimeIndex(dates)
    
    if len(dates) == 0:
        return []
    
    print(f"Getting temperature data for {len(dates)} dates from {dates[0].date()} to {dates[-1].date()}")
    
    # Separate past and future dates
    current_date = datetime.now().date()
    past_dates = []
    future_dates = []
    
    for date in dates:
        if date.date() <= current_date:
            past_dates.append(date)
        else:
            future_dates.append(date)
    
    temperatures = {}
    
    # Handle past dates - get actual historical data
    if past_dates:
        print(f"Retrieving actual historical data for {len(past_dates)} past dates")
        past_temps = _get_historical_temperatures(past_dates, location)
        temperatures.update(past_temps)
    
    # Handle future dates - calculate averages from past 3 years
    if future_dates:
        print(f"Calculating forecasted temperatures for {len(future_dates)} future dates using 3-year historical averages")
        future_temps = _get_forecasted_temperatures(future_dates, location)
        temperatures.update(future_temps)
    
    # Build result list in original order
    result = []
    for date in dates:
        date_str = date.strftime("%Y-%m-%d")
        if date_str in temperatures:
            result.append(temperatures[date_str])
        else:
            # Fallback to climate normal
            fallback_temp = _get_climate_normal(date, location)
            result.append(fallback_temp)
            print(f"Using climate normal fallback for {date_str}: {fallback_temp:.2f}°C")
    
    return result[0] if single_date_input else result

def _get_historical_temperatures(dates, location):
    """
    Get actual historical temperatures for past dates.
    
    Args:
        dates: List of past dates
        location: Location dictionary
        
    Returns:
        Dictionary mapping date strings to temperatures
    """
    if not dates:
        return {}
    
    location_str = f"{location['lat']},{location['lon']}"
    temperatures = {}
    
    try:
        # Group consecutive dates for efficient API calls
        date_ranges = _group_consecutive_dates(dates)
        
        for start_date, end_date in date_ranges:
            start_str = start_date.strftime("%Y-%m-%d")
            end_str = end_date.strftime("%Y-%m-%d")
            
            if start_date == end_date:
                url = f"{BASE_URL}/{location_str}/{start_str}"
            else:
                url = f"{BASE_URL}/{location_str}/{start_str}/{end_str}"
            
            params = {
                "key": VISUAL_CROSSING_API_KEY,
                "unitGroup": WEATHER_CONFIG["visual_crossing"]["units"],
                "include": "days",
                "elements": "datetime,temp"
            }
            
            response = requests.get(url, params=params, timeout=30)
            range_temps = _parse_api_response(response, start_date, end_date)
            temperatures.update(range_temps)
            
    except Exception as e:
        print(f"Error getting historical temperatures: {str(e)}")
        # Fallback to individual date requests
        for date in dates:
            try:
                date_str = date.strftime("%Y-%m-%d")
                url = f"{BASE_URL}/{location_str}/{date_str}"
                
                params = {
                    "key": VISUAL_CROSSING_API_KEY,
                    "unitGroup": WEATHER_CONFIG["visual_crossing"]["units"],
                    "include": "days",
                    "elements": "datetime,temp"
                }
                
                response = requests.get(url, params=params, timeout=10)
                temp_data = _parse_api_response(response, date, date)
                temperatures.update(temp_data)
                
            except Exception as date_e:
                print(f"Failed to get temperature for {date.date()}: {str(date_e)}")
                continue
    
    return temperatures

def _get_forecasted_temperatures(future_dates, location):
    """
    Calculate forecasted temperatures for future dates using 3-year historical averages.
    
    For each future date, gets the same calendar date from the past 3 years,
    calculates the average, and uses that as the forecast.
    
    Optimized to make bulk API calls for date ranges instead of individual dates.
    
    Args:
        future_dates: List of future dates
        location: Location dictionary
        
    Returns:
        Dictionary mapping date strings to forecasted temperatures
    """
    if not future_dates:
        return {}
    
    location_str = f"{location['lat']},{location['lon']}"
    forecasted_temps = {}
    current_year = datetime.now().year
    
    # Sort future dates to ensure proper processing
    sorted_future_dates = sorted(future_dates)
    start_date = sorted_future_dates[0]
    end_date = sorted_future_dates[-1]
    
    print(f"Calculating forecasted temperatures for date range {start_date.date()} to {end_date.date()} using 3-year historical averages")
    
    # For each of the past 3 years, get the entire date range
    historical_data_by_year = {}
    years_with_data = []
    
    for year_offset in range(1, 4):  # Past 3 years
        historical_year = current_year - year_offset
        
        try:
            # Create historical date range for this year
            historical_start = start_date.replace(year=historical_year)
            historical_end = end_date.replace(year=historical_year)
            
            # Make sure these historical dates are actually in the past
            current_date = datetime.now().date()
            if historical_end.date() <= current_date:
                historical_start_str = historical_start.strftime("%Y-%m-%d")
                historical_end_str = historical_end.strftime("%Y-%m-%d")
                
                print(f"Retrieving historical data for {historical_year}: {historical_start_str} to {historical_end_str}")
                
                # Make bulk API call for the entire date range
                if historical_start == historical_end:
                    url = f"{BASE_URL}/{location_str}/{historical_start_str}"
                else:
                    url = f"{BASE_URL}/{location_str}/{historical_start_str}/{historical_end_str}"
                
                params = {
                    "key": VISUAL_CROSSING_API_KEY,
                    "unitGroup": WEATHER_CONFIG["visual_crossing"]["units"],
                    "include": "days",
                    "elements": "datetime,temp"
                }
                
                response = requests.get(url, params=params, timeout=30)
                
                if response.status_code == 200:
                    data = response.json()
                    if "days" in data:
                        year_data = {}
                        for day in data["days"]:
                            date_str = day.get("datetime")
                            temp = day.get("temp")
                            if date_str and temp is not None:
                                # Convert to month-day format for easier matching
                                date_obj = datetime.strptime(date_str, "%Y-%m-%d")
                                month_day = date_obj.strftime("%m-%d")
                                year_data[month_day] = temp
                        
                        if year_data:
                            historical_data_by_year[historical_year] = year_data
                            years_with_data.append(historical_year)
                            print(f"Retrieved {len(year_data)} temperature records for {historical_year}")
                        
                elif response.status_code == 401:
                    raise Exception("Invalid API key. Please check your Visual Crossing API key.")
                elif response.status_code == 429:
                    print(f"Rate limit exceeded for {historical_year}, skipping...")
                    continue
                else:
                    print(f"API error for {historical_year}: Status {response.status_code}")
                    continue
                    
        except ValueError as e:
            # Handle leap year issues (e.g., Feb 29)
            print(f"Date conversion error for {historical_year}: {str(e)}")
            continue
        except Exception as e:
            print(f"Failed to get data for {historical_year}: {str(e)}")
            continue
    
    print(f"Successfully retrieved historical data from {len(years_with_data)} years: {years_with_data}")
    
    # Now calculate forecasted temperatures for each future date
    for future_date in sorted_future_dates:
        future_date_str = future_date.strftime("%Y-%m-%d")
        month_day = future_date.strftime("%m-%d")
        
        # Collect temperatures for this calendar date from all available years
        temps_for_this_date = []
        
        for year, year_data in historical_data_by_year.items():
            if month_day in year_data:
                temps_for_this_date.append(year_data[month_day])
        
        # Calculate average or use fallback
        if temps_for_this_date:
            avg_temp = np.mean(temps_for_this_date)
            forecasted_temps[future_date_str] = avg_temp
            print(f"Forecasted temperature for {future_date_str}: {avg_temp:.2f}°C (avg of {len(temps_for_this_date)} years)")
        else:
            # No historical data available for this calendar date, use climate normal
            climate_temp = _get_climate_normal(future_date, location)
            forecasted_temps[future_date_str] = climate_temp
            print(f"Climate normal temperature for {future_date_str}: {climate_temp:.2f}°C (no historical data)")
    
    # Ensure we have forecasted temperatures for all requested dates
    missing_dates = []
    for future_date in future_dates:
        future_date_str = future_date.strftime("%Y-%m-%d")
        if future_date_str not in forecasted_temps:
            missing_dates.append(future_date)
    
    if missing_dates:
        print(f"Warning: Missing forecasted temperatures for {len(missing_dates)} dates, using climate normals")
        for missing_date in missing_dates:
            missing_date_str = missing_date.strftime("%Y-%m-%d")
            climate_temp = _get_climate_normal(missing_date, location)
            forecasted_temps[missing_date_str] = climate_temp
    
    print(f"Generated forecasted temperatures for {len(forecasted_temps)} dates")
    return forecasted_temps

def _group_consecutive_dates(dates):
    """
    Group consecutive dates into ranges for efficient API calls.
    
    Args:
        dates: List of dates
        
    Returns:
        List of (start_date, end_date) tuples
    """
    if not dates:
        return []
    
    # Sort dates
    sorted_dates = sorted(dates)
    ranges = []
    current_start = sorted_dates[0]
    current_end = sorted_dates[0]
    
    for i in range(1, len(sorted_dates)):
        if (sorted_dates[i] - sorted_dates[i-1]).days == 1:
            # Consecutive date
            current_end = sorted_dates[i]
        else:
            # Gap found
            ranges.append((current_start, current_end))
            current_start = sorted_dates[i]
            current_end = sorted_dates[i]
    
    # Add the last range
    ranges.append((current_start, current_end))
    return ranges

def _parse_api_response(response, start_date, end_date):
    """
    Parse Visual Crossing API response and extract temperatures.
    
    Args:
        response: requests.Response object
        start_date: Start date of the request
        end_date: End date of the request
        
    Returns:
        Dictionary mapping date strings to temperatures
    """
    if response.status_code == 401:
        raise Exception("Invalid API key. Please check your Visual Crossing API key.")
    elif response.status_code == 429:
        raise Exception("API rate limit exceeded. Please try again later.")
    elif response.status_code != 200:
        raise Exception(f"Weather API returned status {response.status_code}: {response.text}")
    
    try:
        data = response.json()
        
        if "days" not in data:
            raise Exception("No weather data returned from API")
        
        temperatures = {}
        
        # Generate date range for mapping when datetime is not provided
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, "%Y-%m-%d")
        if isinstance(end_date, str):
            end_date = datetime.strptime(end_date, "%Y-%m-%d")
            
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        
        for i, day in enumerate(data["days"]):
            temp = day.get("temp")
            if temp is not None:
                # Try to get datetime from response, otherwise use date range
                date_str = day.get("datetime")
                if not date_str and i < len(date_range):
                    date_str = date_range[i].strftime("%Y-%m-%d")
                
                if date_str:
                    temperatures[date_str] = temp
        
        return temperatures
        
    except json.JSONDecodeError:
        raise Exception("Invalid JSON response from weather API")
    except Exception as e:
        raise Exception(f"Error parsing API response: {str(e)}")

def _get_climate_normal(date, location):
    """
    Calculate climate normal temperature for a date and location.
    This is a simple fallback when API calls fail.
    """
    month = date.month
    lat = abs(location["lat"])
    
    # Base temperature decreases with latitude
    base_temp = 30 - (lat * 0.6)  # Rough approximation
    
    # Seasonal variation
    if month in [12, 1, 2]:  # Winter
        seasonal_adj = -10
    elif month in [3, 4, 5]:  # Spring
        seasonal_adj = 0
    elif month in [6, 7, 8]:  # Summer
        seasonal_adj = 5
    else:  # Fall
        seasonal_adj = -2
    
    # Adjust for Southern Hemisphere
    if location["lat"] < 0:
        seasonal_adj = -seasonal_adj
    
    return base_temp + seasonal_adj

def validate_temperature_forecast_accuracy(location=None, test_period_days=30):
    """
    Validate the Visual Crossing Weather API integration.
    Tests both historical data retrieval and future forecasting logic, including date range series.
    
    Args:
        location: Dict with lat/lon coordinates (optional, uses global location if None)
        test_period_days: Number of days to test (for testing purposes)
        
    Returns:
        Dict with validation results
    """
    # Use provided location or get global location
    if location is None:
        location = get_location()
        
    # Validate API key
    if VISUAL_CROSSING_API_KEY == "your_visual_crossing_api_key_here":
        raise Exception("Visual Crossing API key not configured. Please set VISUAL_CROSSING_API_KEY environment variable or update config/weather_config.json")
    
    try:
        # Test 1: Historical data retrieval (single date)
        historical_date = datetime.now() - timedelta(days=7)
        print(f"Testing historical data retrieval for: {historical_date.date()}")
        historical_temp = get_temperature_forecast(historical_date)
        
        # Test 2: Multiple historical dates
        historical_dates = pd.date_range(start=historical_date, periods=3, freq='D')
        historical_temps = get_temperature_forecast(historical_dates)
        
        # Test 3: Future date forecasting (should use 3-year average)
        future_date = datetime.now() + timedelta(days=30)
        print(f"Testing future forecasting for: {future_date.date()}")
        future_temp = get_temperature_forecast(future_date)
        
        # Test 4: Future date range forecasting
        future_dates = pd.date_range(start=future_date, periods=3, freq='D')
        future_temps = get_temperature_forecast(future_dates)
        
        # Test 5: Temperature series for date range (NEW TEST)
        series_start = future_date
        series_end = future_date + timedelta(days=6)  # 7-day series
        print(f"Testing temperature series for range: {series_start.date()} to {series_end.date()}")
        temperature_series = get_temperature_series(series_start, series_end)
        
        # Test 6: Mixed historical/future series
        mixed_start = datetime.now() - timedelta(days=3)
        mixed_end = datetime.now() + timedelta(days=3)
        print(f"Testing mixed historical/future series: {mixed_start.date()} to {mixed_end.date()}")
        mixed_series = get_temperature_series(mixed_start, mixed_end)
        
        return {
            "status": "success",
            "api_provider": "Visual Crossing Weather API",
            "location": f"{location['lat']}, {location['lon']}",
            "tests": {
                "historical_single": {
                    "date": historical_date.strftime("%Y-%m-%d"),
                    "temperature": historical_temp,
                    "type": "actual_historical_data"
                },
                "historical_multiple": {
                    "dates": [d.strftime("%Y-%m-%d") for d in historical_dates],
                    "temperatures": historical_temps,
                    "type": "actual_historical_data"
                },
                "future_single": {
                    "date": future_date.strftime("%Y-%m-%d"),
                    "temperature": future_temp,
                    "type": "3_year_average_forecast"
                },
                "future_multiple": {
                    "dates": [d.strftime("%Y-%m-%d") for d in future_dates],
                    "temperatures": future_temps,
                    "type": "3_year_average_forecast"
                },
                "temperature_series": {
                    "start_date": series_start.strftime("%Y-%m-%d"),
                    "end_date": series_end.strftime("%Y-%m-%d"),
                    "series_length": len(temperature_series),
                    "sample_data": temperature_series[:3],  # First 3 entries as sample
                    "type": "complete_date_range_series"
                },
                "mixed_series": {
                    "start_date": mixed_start.strftime("%Y-%m-%d"),
                    "end_date": mixed_end.strftime("%Y-%m-%d"),
                    "series_length": len(mixed_series),
                    "sample_data": mixed_series[:3],  # First 3 entries as sample
                    "type": "mixed_historical_and_forecast_series"
                }
            },
            "methodology": {
                "historical_dates": "Direct API call to Visual Crossing for actual recorded temperatures",
                "future_dates": "Calculate average from same calendar dates over past 3 years",
                "date_ranges": "Bulk API calls for efficient retrieval of temperature series",
                "fallback": "Climate normal calculation when API data unavailable"
            },
            "features": {
                "single_date_support": True,
                "multiple_date_support": True,
                "date_range_series": True,
                "mixed_historical_future": True,
                "bulk_api_optimization": True,
                "complete_series_guarantee": True
            }
        }
        
    except Exception as e:
        return {
            "status": "failed",
            "error": str(e),
            "api_provider": "Visual Crossing Weather API"
        } 