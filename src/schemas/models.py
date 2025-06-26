from pydantic import BaseModel
from typing import List, Dict, Optional

class TrainModelRequest(BaseModel):
    meter_id: str
    meter_type: str  # 'electricity' or 'water'

class TaskStatus(BaseModel):
    task_id: str
    status: str
    progress: Optional[int] = None
    result: Optional[Dict] = None
    error: Optional[str] = None
    error_details: Optional[Dict] = None

class ForecastRequest(BaseModel):
    meter_id: str
    meter_type: str  # 'electricity' or 'water'
    start_date: str  # Start date in YYYY-MM-DD format
    end_date: str    # End date in YYYY-MM-DD format
    latitude: float   # Location latitude
    longitude: float  # Location longitude
    city: str        # Location city name

class ForecastResponse(BaseModel):
    forecast_data: List[Dict]  # Each dict: {datetime, value, type, temperature (optional)}
    meter_id: Optional[str] = None
    meter_type: Optional[str] = None

class LocationRequest(BaseModel):
    latitude: float
    longitude: float
    city: str

class ValidationResponse(BaseModel):
    error_metrics: Dict

class ModelTestRequest(BaseModel):
    meter_id: str
    meter_type: str  # 'electricity' or 'water'
    test_size: float = 0.2  # Proportion of data to use for testing (default 20%)

class ModelTestResponse(BaseModel):
    test_results: Dict 