from fastapi import APIRouter, HTTPException, Query
from src.schemas.models import ModelTestRequest, ModelTestResponse
from src.services.model_service import test_model, test_all_models
from src.logger_config import get_logger

router = APIRouter(prefix="/test_model", tags=["Model Testing"])
logger = get_logger("energy_forecasting.api.testing")

@router.post("/", response_model=ModelTestResponse)
def test_model_accuracy(request: ModelTestRequest):
    """
    Test the accuracy of a trained model using historical data.
    
    Splits historical data into training and test sets, then evaluates
    the model's accuracy on the test set.
    """
    meter_id = request.meter_id
    meter_type = request.meter_type.lower()
    test_size = request.test_size
    
    # Validate meter type
    if meter_type not in ["electricity", "water"]:
        raise HTTPException(status_code=400, detail="meter_type must be 'electricity' or 'water'")
    
    # Validate test size
    if test_size <= 0 or test_size >= 1:
        raise HTTPException(status_code=400, detail="test_size must be between 0 and 1")
    
    try:
        # Test model using model service
        test_results, error = test_model(
            meter_id, 
            meter_type, 
            test_size, 
            request.latitude, 
            request.longitude, 
            request.city
        )
        
        if error:
            raise HTTPException(status_code=404, detail=error)
        
        return ModelTestResponse(test_results=test_results)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error testing model: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to test model: {str(e)}")

@router.post("/all_models/")
def test_all_available_models(
    test_size: float = 0.2,
    latitude: float = Query(..., description="Location latitude (required for temperature data)"),
    longitude: float = Query(..., description="Location longitude (required for temperature data)"),
    city: str = Query(..., description="Location city name (required for temperature data)")
):
    """
    Test all available trained models and return their accuracy metrics.
    
    """
    # Validate test size
    if test_size <= 0 or test_size >= 1:
        raise HTTPException(status_code=400, detail="test_size must be between 0 and 1")
    
    try:
        # Test all models using model service
        results = test_all_models(test_size, latitude, longitude, city)
        
        if "error" in results:
            raise HTTPException(status_code=500, detail=results["error"])
        
        return results
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error testing all models: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to test all models: {str(e)}") 