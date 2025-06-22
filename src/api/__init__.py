"""
API routes for the Energy Forecasting system
"""
from fastapi import APIRouter

# Create a main API router
api_router = APIRouter()

# Import and include all routers
try:
    from src.api.routes.health import router as health_router
    api_router.include_router(health_router)
except Exception as e:
    print(f"Error importing health router: {e}")

try:
    from src.api.routes.location import router as location_router
    api_router.include_router(location_router)
except Exception as e:
    print(f"Error importing location router: {e}")

try:
    from src.api.routes.training import router as training_router
    api_router.include_router(training_router)
except Exception as e:
    print(f"Error importing training router: {e}")

try:
    from src.api.routes.forecast import router as forecast_router
    api_router.include_router(forecast_router)
except Exception as e:
    print(f"Error importing forecast router: {e}")

try:
    from src.api.routes.testing import router as testing_router
    api_router.include_router(testing_router)
except Exception as e:
    print(f"Error importing testing router: {e}")

try:
    from src.api.routes.temperature import router as temperature_router
    api_router.include_router(temperature_router)
except Exception as e:
    print(f"Error importing temperature router: {e}") 