"""
Main application module for the Energy Forecasting system.

"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file before importing other modules
dotenv_path = Path(__file__).parent.parent / '.env'
if dotenv_path.exists():
    load_dotenv(dotenv_path=dotenv_path, encoding="latin-1")
    print(f"Loaded environment variables from {dotenv_path}")
else:
    print(f"Warning: .env file not found at {dotenv_path}")

from src.api import api_router
from src.logger_config import setup_logging, get_logger, debug_mode_from_env
from src.services.health_service import get_system_health
from src.services.model_service import ensure_model_dirs

# Setup logging
debug_mode = debug_mode_from_env()
setup_logging(debug=debug_mode)
logger = get_logger("energy_forecasting.main")

# Log successful environment variable loading
env_vars_loaded = all([
    os.getenv('INFLUXDB_URL'), 
    os.getenv('INFLUXDB_ORG'),
    os.getenv('INFLUXDB_BUCKET')
])
logger.info("Environment configuration status", env_vars_loaded=env_vars_loaded)

# Create FastAPI app
app = FastAPI(title="UMS Forecasting Service")

# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Ensure model directories exist
ensure_model_dirs()

# Include all API routes
app.include_router(api_router)

# Add root health check endpoint directly in main.py for backward compatibility
@app.get("/")
def health_check():
    """
    Simple health check endpoint.
    
    Returns basic system information and API status.
    """
    health_info = get_system_health()
    return health_info 