from fastapi import APIRouter, HTTPException
from src.schemas.models import LocationRequest
from src.services.location_service import set_global_location, get_global_location

router = APIRouter(prefix="/location", tags=["Location"])

@router.post("/", status_code=201)
def set_location_endpoint(request: LocationRequest):
    """
    Set the global location for temperature forecasting.
    This location will be used for all meters.
    """
    success = set_global_location(
        request.latitude,
        request.longitude,
        request.city
    )
    
    if success:
        return {"message": "Global location set successfully"}
    else:
        raise HTTPException(status_code=500, detail="Failed to set global location")

@router.get("/")
def get_location_endpoint():
    """
    Get the global location used for temperature forecasting
    """
    location = get_global_location()
    
    if "error" in location:
        raise HTTPException(status_code=500, detail=location["error"])
        
    return location 