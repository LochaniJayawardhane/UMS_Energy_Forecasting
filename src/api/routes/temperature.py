from fastapi import APIRouter, HTTPException
from src.schemas.models import ValidationResponse
from src.services.location_service import validate_temperature_accuracy, get_temperature_data
from src.logger_config import get_logger

router = APIRouter(tags=["Temperature"])
logger = get_logger("energy_forecasting.api.temperature")

@router.get("/temperature/validate/", response_model=ValidationResponse)
def validate_temperature_accuracy_endpoint(days: int = 30):
    """
    Validate the accuracy of temperature forecasting
    """
    error_metrics = validate_temperature_accuracy(test_period_days=days)
    
    if "error" in error_metrics:
        raise HTTPException(status_code=500, detail=error_metrics["error"])
    
    return {
        "error_metrics": error_metrics
    }

@router.get("/temperature/data/")
def get_temperature_data_endpoint(start_date: str, end_date: str):
    """
    Get temperature data for a date range
    """
    temperature_data = get_temperature_data(start_date, end_date)
    
    if "error" in temperature_data:
        raise HTTPException(status_code=500, detail=temperature_data["error"])
        
    return temperature_data 