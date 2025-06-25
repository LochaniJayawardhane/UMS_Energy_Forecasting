#!/usr/bin/env python3
"""
Dramatiq Worker for Energy Forecasting System

This script starts Dramatiq workers to process background tasks for model training.
"""
import os
import sys
import signal
import time
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file before importing other modules
dotenv_path = Path(__file__).parent.parent / '.env'
if dotenv_path.exists():
    load_dotenv(dotenv_path=dotenv_path, encoding="latin-1")
    print(f"Loaded environment variables from {dotenv_path}")
else:
    print(f"Warning: .env file not found at {dotenv_path}")

from src.logger_config import setup_logging, get_logger, debug_mode_from_env
from src.dramatiq_broker import broker
import src.task_system as task_system  

debug_mode = debug_mode_from_env()
setup_logging(debug=debug_mode)
logger = get_logger("dramatiq.worker")

# Log environment variables for debugging (excluding sensitive ones)
safe_env_vars = {
    'INFLUXDB_URL': os.getenv('INFLUXDB_URL'),
    'INFLUXDB_ORG': os.getenv('INFLUXDB_ORG'),
    'INFLUXDB_BUCKET': os.getenv('INFLUXDB_BUCKET'),
    'VISUAL_CROSSING_BASE_URL': os.getenv('VISUAL_CROSSING_BASE_URL'),
    'LOCATION_CITY': os.getenv('LOCATION_CITY')
}
logger.info("Environment variables loaded", **{k: v for k, v in safe_env_vars.items() if v is not None})

def signal_handler(signum, frame):
    """Handle shutdown signals gracefully"""
    logger.info("Received shutdown signal", signal=signum)
    sys.exit(0)

def main():
    """Main worker function"""
    logger.info("Starting Dramatiq worker for Energy Forecasting")
    logger.info("Worker configuration", 
                debug_mode=debug_mode,
                redis_url=os.getenv('REDIS_URL', 'redis://localhost:6379/0'),
                working_directory=os.getcwd())
    
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Import dramatiq CLI and run worker
        from dramatiq.cli import main as dramatiq_main
        
        # Set up worker arguments
        worker_args = [
            "dramatiq",
            "src.task_system",  # Module containing our tasks
            "--processes", "2",  # Number of worker processes
            "--threads", "2",    # Number of threads per process
            "--verbose",         # Verbose logging
        ]
        
        if debug_mode:
            worker_args.append("--watch")  # Auto-reload on code changes in debug mode
        
        logger.info("Starting Dramatiq worker with arguments", args=worker_args[1:])
        
        # Override sys.argv and run dramatiq
        original_argv = sys.argv
        sys.argv = worker_args
        
        try:
            dramatiq_main()
        finally:
            sys.argv = original_argv
            
    except KeyboardInterrupt:
        logger.info("Worker stopped by user")
    except Exception as e:
        logger.error("Worker failed with exception", error=str(e))
        raise
    finally:
        logger.info("Dramatiq worker shutting down")

if __name__ == "__main__":
    main() 