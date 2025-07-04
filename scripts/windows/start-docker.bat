@echo off
echo Starting Energy Forecasting with Docker...
echo.

echo Checking Docker installation...
docker --version > nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Error: Docker is not installed or not in your PATH.
    echo Please install Docker Desktop from https://www.docker.com/products/docker-desktop/
    echo.
    pause
    exit /b 1
)

echo Checking Docker Compose...
docker-compose --version > nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Error: Docker Compose is not installed or not in your PATH.
    echo It should be included with Docker Desktop.
    echo.
    pause
    exit /b 1
)

echo Checking .env file...
if not exist .env (
    echo .env file not found.
    echo Creating from template...
    copy env.example .env
    echo Please edit the .env file with your actual values.
    echo.
    pause
)

echo Building and starting Docker containers...
docker-compose up -d --build

echo.
echo Containers are starting up. This may take a minute.
echo.
echo You can access the API at: http://localhost:8000
echo API documentation is available at: http://localhost:8000/docs
echo.
echo Use these scripts to interact with the system:
echo - scripts\docker-train-model.bat [model_type] [meter_id]
echo - scripts\docker-get-forecast.bat [model_type] [meter_id] [days]
echo.
echo To view logs:
echo docker-compose logs -f api
echo.
echo Press any key to exit...
pause > nul 