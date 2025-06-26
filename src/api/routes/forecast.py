from fastapi import APIRouter, HTTPException
import pandas as pd
from typing import Dict

from src.schemas.models import ForecastRequest, ForecastResponse
from src.logger_config import get_logger
from src.services.forecast_service import generate_forecast

router = APIRouter(tags=["Forecasting"])
logger = get_logger("energy_forecasting.api.forecast")

@router.post("/forecast/", response_model=ForecastResponse)
def forecast(request: ForecastRequest):
    meter_id = request.meter_id
    meter_type = request.meter_type.lower()
    
    # Validate meter type
    if meter_type not in ["electricity", "water"]:
        raise HTTPException(status_code=400, detail="meter_type must be 'electricity' or 'water'")
    
    # Parse and validate dates
    try:
        start_date = request.start_date
        end_date = request.end_date
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid date format. Use YYYY-MM-DD format. Error: {str(e)}")
    
    # Generate forecast using the forecast service
    forecast_data, error = generate_forecast(
        meter_id, 
        meter_type, 
        start_date, 
        end_date,
        request.latitude,
        request.longitude,
        request.city
    )
    
    if error:
        # Check if it's a no historical data error
        if "No historical data is available" in error:
            raise HTTPException(status_code=404, detail=error)
        # Other errors
        raise HTTPException(status_code=500, detail=error)
    
    if not forecast_data:
        raise HTTPException(status_code=404, detail="No forecast data available")
    
    return {
        "forecast_data": forecast_data,
        "meter_id": meter_id,
        "meter_type": meter_type
    }
