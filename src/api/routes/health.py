from fastapi import APIRouter, HTTPException
from src.services.health_service import get_system_health, get_worker_health
from src.logger_config import get_logger

router = APIRouter(tags=["Health"])
logger = get_logger("energy_forecasting.api.health")

@router.get("/")
def health_check():
    """
    Simple health check endpoint.
    
    Returns basic system information and API status.
    """
    health_info = get_system_health()
    
    if health_info.get("status") != "healthy":
        raise HTTPException(status_code=500, detail="Health check failed")
        
    return health_info

@router.get("/worker")
def worker_health():
    """
    Check the health of the Dramatiq worker system.
    
    Verifies Redis connection and checks if workers are processing tasks.
    """
    worker_info = get_worker_health()
    
    return worker_info 