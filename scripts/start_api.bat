@echo off
echo Starting Energy Forecasting API with Dramatiq...
echo.
echo This will start the FastAPI server with the new Dramatiq-based background task system
echo The server will be available at http://localhost:8000
echo.
echo Press Ctrl+C to stop the server
echo.

:: Set Redis URL environment variable
set REDIS_URL=redis://localhost:6379/0

:: Check if Redis is available using Python
echo Checking Redis connection...
python -c "import redis; r=redis.Redis(host='localhost',port=6379,db=0); r.ping(); print('Redis OK')" >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo WARNING: Redis is not accessible!
    echo Please make sure Redis is running with: docker run -d -p 6379:6379 --name redis redis
    echo Or check if the container is running: docker ps
    echo.
    echo The API will still start, but background tasks will not work until Redis is available.
    echo.
    timeout /t 3 >nul
) else (
    echo ✅ Redis connection verified - background tasks will work properly
)

:: Check if we're in debug mode
if /i "%DEBUG%"=="true" (
    echo Debug mode enabled - enhanced logging and auto-reload
    set DEBUG=true
) else (
    echo Production mode - set DEBUG=true for development with enhanced logging
    set DEBUG=false
)

echo.
echo Starting FastAPI server with Dramatiq...
echo API Documentation: http://localhost:8000/docs
echo Health Check: http://localhost:8000/health/
echo Worker Health: http://localhost:8000/health/worker
echo.
echo Remember to start workers separately with: start_dramatiq_worker.bat
echo.

:: Start the FastAPI server with Uvicorn (from src directory)
cd /d "%~dp0\.."
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000 