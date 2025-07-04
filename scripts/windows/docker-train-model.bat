@echo off
REM Script for training models using Docker containers

REM Parse command line arguments
set MODEL_TYPE=%1
set METER_ID=%2

REM Set defaults if not provided
if "%MODEL_TYPE%"=="" set MODEL_TYPE=electricity
if "%METER_ID%"=="" set METER_ID=meter_123

REM Display information about the operation
echo Training %MODEL_TYPE% model for meter %METER_ID%...
echo Using the dockerized Energy Forecasting service
echo.

REM Call the API to start model training
curl -s -X POST "http://localhost:8000/api/v1/training/%MODEL_TYPE%/%METER_ID%" ^
  -H "Content-Type: application/json" ^
  -d "{}"

echo.
echo.
echo To check training status, you can use:
echo curl http://localhost:8000/api/v1/training/status/YOUR_TASK_ID
echo.
echo When training is complete, you can get a forecast using:
echo scripts\docker-get-forecast.bat %MODEL_TYPE% %METER_ID%
echo.

pause 