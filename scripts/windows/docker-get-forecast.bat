@echo off
REM Script for getting forecasts from Docker containers

REM Parse command line arguments
set MODEL_TYPE=%1
set METER_ID=%2
set DAYS=%3

REM Set defaults if not provided
if "%MODEL_TYPE%"=="" set MODEL_TYPE=electricity
if "%METER_ID%"=="" set METER_ID=meter_123
if "%DAYS%"=="" set DAYS=7

REM Display information about the operation
echo Getting %DAYS% day forecast for %MODEL_TYPE% meter %METER_ID%...
echo Using the dockerized Energy Forecasting service
echo.

REM Call the API to get the forecast
curl -s -X GET "http://localhost:8000/api/v1/forecast/%MODEL_TYPE%/%METER_ID%?days=%DAYS%" ^
  -H "Content-Type: application/json"

REM Add helpful message
echo.
echo.
echo Request complete
echo For different forecast periods, specify days as the third argument:
echo scripts\docker-get-forecast.bat %MODEL_TYPE% %METER_ID% [days]
echo.

pause 