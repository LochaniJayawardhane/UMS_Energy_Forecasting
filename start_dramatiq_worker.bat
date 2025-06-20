@echo off
echo Starting Dramatiq Worker for Energy Forecasting System...
echo.
echo This will start background workers to process model training tasks
echo Press Ctrl+C to stop the worker
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
    pause
    exit /b 1
) else (
    echo ✅ Redis connection verified - ready to process tasks
)

:: Check if we're in debug mode
if /i "%DEBUG%"=="true" (
    echo Debug mode enabled - worker will auto-reload on code changes
    set DEBUG=true
) else (
    echo Production mode - set DEBUG=true for development
    set DEBUG=false
)

echo.
echo Starting Dramatiq worker...
echo Worker will process tasks in the background
echo Logs will be displayed here and saved to the logs/ directory
echo.

:: Start the Dramatiq worker
python dramatiq_worker.py 