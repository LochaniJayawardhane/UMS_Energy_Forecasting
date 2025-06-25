"""
Utility module for loading InfluxDB configuration
"""
import os
from pathlib import Path
from dotenv import load_dotenv

def load_influxdb_config():
    """
    Load InfluxDB configuration from environment variables.
    
    """
    # Load environment variables
    env_file = find_dotenv_file()
    load_dotenv(dotenv_path=env_file, encoding="latin-1")
    
    # Get configuration from environment variables
    config = {
        'url': os.getenv('INFLUXDB_URL'),
        'token': os.getenv('INFLUXDB_TOKEN'),
        'org': os.getenv('INFLUXDB_ORG'),
        'bucket': os.getenv('INFLUXDB_BUCKET')
    }
    
    # Validate required keys
    required_keys = ['url', 'token', 'org', 'bucket']
    missing_keys = [key for key in required_keys if not config.get(key)]
    
    if missing_keys:
        raise ValueError(f"Missing required environment variables: {', '.join(['INFLUXDB_' + key.upper() for key in missing_keys])}")
    
    return config

def get_influxdb_config():
    """
    Get InfluxDB configuration as individual variables for backward compatibility.
    
    Returns:
        tuple: (url, token, org, bucket)
    """
    config = load_influxdb_config()
    return config['url'], config['token'], config['org'], config['bucket']

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