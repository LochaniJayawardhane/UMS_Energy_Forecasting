"""
Main application module for the Energy Forecasting system.

This module initializes the FastAPI application, sets up middleware,
and includes all API routes from the src.api package.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os

from src.api import api_router
from src.logger_config import setup_logging, get_logger, debug_mode_from_env
from src.services.health_service import get_system_health
from src.services.model_service import ensure_model_dirs

# Setup logging
debug_mode = debug_mode_from_env()
setup_logging(debug=debug_mode)
logger = get_logger("energy_forecasting.main")

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
    # Instead of importing from health.py, use the service directly
    health_info = get_system_health()
    return health_info 