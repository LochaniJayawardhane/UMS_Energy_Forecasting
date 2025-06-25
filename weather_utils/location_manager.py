import os
import json
from config.weather_config import get_location as get_location_from_env
from config.weather_config import set_location as set_location_in_env
from config.weather_config import load_weather_config as load_config_from_env

def get_config_path():
    """Get the path to the weather config file"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    return os.path.join(project_root, 'config', 'weather_config.json')

def ensure_config_dir():
    """Ensure the config directory exists"""
    config_path = get_config_path()
    config_dir = os.path.dirname(config_path)
    os.makedirs(config_dir, exist_ok=True)

def load_weather_config():
    """
    Load weather configuration from environment variables
    """
    return load_config_from_env()

def save_weather_config(config):
    """Save weather configuration"""
    config_path = get_config_path()
    try:
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        return True
    except Exception as e:
        print(f"Error saving weather config: {str(e)}")
        return False

def get_location():
    """
    Get the location information from environment variables
    """
    try:
        location = get_location_from_env()
        return location
    except KeyError:
        raise Exception("Location not found in environment variables. Please set location using the /location/ endpoint.")
    except Exception as e:
        raise Exception(f"Failed to load location from environment variables. Error: {str(e)}")

def set_location(lat, lon, city):
    """
    Set the location in environment variables
    
    Note: This only modifies the in-memory values and will not persist across restarts.
    To make permanent changes, update your .env file manually.
    """
    try:
        return set_location_in_env(lat, lon, city)
    except Exception as e:
        raise Exception(f"Cannot update location. Error: {str(e)}") 