"""
Utility module for loading Weather API configuration
"""
import os
from pathlib import Path
from dotenv import load_dotenv

def load_weather_config():
    """
    Load Weather API configuration from environment variables.

    """
    # Load environment variables
    env_file = find_dotenv_file()
    load_dotenv(dotenv_path=env_file, encoding="latin-1")
    
    # Get configuration from environment variables
    config = {
        'visual_crossing': {
            'api_key': os.getenv('VISUAL_CROSSING_API_KEY'),
            'units': os.getenv('VISUAL_CROSSING_UNITS'),
            'base_url': os.getenv('VISUAL_CROSSING_BASE_URL')
        },
        'location': {
            'lat': float(os.getenv('LOCATION_LAT')),
            'lon': float(os.getenv('LOCATION_LON')),
            'city': os.getenv('LOCATION_CITY')
        }
    }
    
    # Validate required keys
    validate_config(config)
    
    return config

def validate_config(config):
    """
    Validate that all required configuration values are present.

    """
    # Check Visual Crossing API settings
    if not config['visual_crossing']['api_key']:
        raise ValueError("Missing required environment variable: VISUAL_CROSSING_API_KEY")
    if not config['visual_crossing']['units']:
        raise ValueError("Missing required environment variable: VISUAL_CROSSING_UNITS")
    if not config['visual_crossing']['base_url']:
        raise ValueError("Missing required environment variable: VISUAL_CROSSING_BASE_URL")
    
    # Check location settings
    if not config['location']['city']:
        raise ValueError("Missing required environment variable: LOCATION_CITY")
    
    # lat and lon should be numbers
    try:
        float(config['location']['lat'])
        float(config['location']['lon'])
    except (TypeError, ValueError):
        raise ValueError("LOCATION_LAT and LOCATION_LON must be valid numbers")

def get_location():
    """
    Get the location information from environment variables
    
    Returns:
        dict: Location dictionary with lat, lon, and city
    """
    config = load_weather_config()
    return config['location']

def set_location(lat, lon, city):
    """
    This is a placeholder function that would update the environment variables
    or .env file. Since directly modifying .env files is not recommended at runtime,
    this function logs a warning and doesn't actually update the file.
    
    In a real-world scenario, you would either:
    1. Use a database to store this information
    2. Implement a proper configuration management system
    3. Update a separate configuration file specifically for dynamic settings
    
    Returns:
        bool: Always returns False in this implementation
    """
    print(f"Warning: Setting location to lat={lat}, lon={lon}, city={city}")
    print("Note: Location updates via environment variables are not persisted.")
    print("To permanently update location, please modify your .env file manually.")
    
    return False

def find_dotenv_file():
    """
    Find the .env file in various possible locations.

    """
    # Try different possible locations
    possible_locations = [
        Path(".env"),                    # Current working directory
        Path.cwd() / ".env",             # Explicit current working directory
        Path(__file__).parent.parent / ".env",  # Project root (two directories up from this file)
    ]
    
    for location in possible_locations:
        if location.exists():
            return location
    
    # If no .env file is found, return the first location and let dotenv handle the absence
    return possible_locations[0] 