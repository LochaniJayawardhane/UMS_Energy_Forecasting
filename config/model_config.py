"""
Utility module for loading Model configuration
"""
import os
from pathlib import Path
from dotenv import load_dotenv

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

def get_model_path():
    """
    Get the base path for models from environment variables.
    Defaults to 'models' if not specified in .env
    """
    # Load environment variables
    env_file = find_dotenv_file()
    load_dotenv(dotenv_path=env_file, encoding="latin-1")
    
    # Get model path with default fallback
    model_path = os.getenv("MODEL_PATH", "models")
    
    return model_path 