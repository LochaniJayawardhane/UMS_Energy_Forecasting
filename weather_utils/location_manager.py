import json
import os

def ensure_config_dir():
    """Ensure the config directory exists"""
    os.makedirs('config', exist_ok=True)

def load_weather_config():
    """Load weather configuration"""
    config_path = 'config/weather_config.json'
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        raise Exception(f"Weather config file not found or invalid: {str(e)}. Please ensure config/weather_config.json exists with proper location settings.")

def save_weather_config(config):
    """Save weather configuration"""
    config_path = 'config/weather_config.json'
    try:
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        return True
    except Exception as e:
        print(f"Error saving weather config: {str(e)}")
        return False

def get_location():
    """
    Get the location information from weather_config.json
    
    Returns:
        Dictionary with location data (lat, lon, city)
        
    Raises:
        Exception: If location data cannot be loaded
    """
    try:
        weather_config = load_weather_config()
        location = weather_config["location"]
        return location
    except KeyError:
        raise Exception("Location not found in weather_config.json. Please set location using the /location/ endpoint.")
    except Exception as e:
        raise Exception(f"Failed to load location from weather_config.json. Error: {str(e)}")

def set_location(lat, lon, city):
    """
    Set the location in weather_config.json
    
    Args:
        lat: Latitude
        lon: Longitude
        city: City name
        
    Returns:
        Boolean indicating success
    """
    ensure_config_dir()
    
    try:
        # Load existing config
        weather_config = load_weather_config()
    except Exception as e:
        raise Exception(f"Cannot load weather_config.json to update location. Please ensure the file exists and is valid. Error: {str(e)}")
    
    # Update location
    weather_config["location"] = {
        "lat": float(lat),
        "lon": float(lon),
        "city": city
    }
    
    # Save updated config
    return save_weather_config(weather_config) 