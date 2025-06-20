# Weather API Integration - Visual Crossing Weather API

This document explains the weather API integration for the Energy Forecasting system using **Visual Crossing Weather API**.

## Overview

The energy forecasting system integrates weather temperature data to improve prediction accuracy. The system uses Visual Crossing Weather API which provides:

- **Historical Weather Data**: Actual historical temperature data for past dates
- **Statistical Forecasts**: Future temperature predictions based on historical patterns
- **Global Coverage**: Weather data for any location worldwide
- **Free Tier**: 1000 records per day completely free (no credit card required)

## API Provider: Visual Crossing Weather API

**Why Visual Crossing?**
- ✅ **True Historical Data**: Access to actual historical weather records (not approximations)
- ✅ **No Credit Card Required**: Free tier with 1000 records/day
- ✅ **Single API**: Handles both historical and forecast data seamlessly
- ✅ **Reliable**: Enterprise-grade weather data provider since 2003
- ✅ **Comprehensive**: 50+ years of historical data available

**API Documentation**: https://www.visualcrossing.com/resources/documentation/weather-api/timeline-weather-api/

## Configuration

### 1. API Key Setup

**Option A: Environment Variable (Recommended)**
```bash
set VISUAL_CROSSING_API_KEY=your_actual_api_key_here
```

**Option B: Configuration File**
Update `config/weather_config.json`:
```json
{
  "visual_crossing": {
    "api_key": "your_actual_api_key_here",
    "units": "metric",
    "base_url": "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline"
  },
  "location": {
    "lat": 40.7128,
    "lon": -74.0060,
    "city": "New York"
  }
}
```

### 2. Get Your Free API Key

1. Visit: https://www.visualcrossing.com/weather-api
2. Sign up for a free account
3. Get your API key from the account dashboard
4. Free tier includes 1000 records per day

## Architecture

### File Structure
```
weather_utils/
├── __init__.py
├── weather.py              # Visual Crossing API integration
└── location_manager.py     # Location management

config/
└── weather_config.json     # API configuration
```

### Key Components

#### 1. **Historical Temperature Retrieval**
- **Past Dates**: Uses actual historical weather data from Visual Crossing
- **Future Dates**: Calculates statistical averages from 5 years of historical data
- **Fallback**: Climate normals when API fails

#### 2. **Smart Date Handling**
- **Training Data Boundary**: Only forecasts beyond last recorded training date
- **Mixed Requests**: Handles date ranges spanning past and future
- **Optimization**: Groups consecutive dates for efficient API usage

#### 3. **Error Handling**
- **Network Errors**: Graceful fallback to climate normals
- **API Limits**: Proper handling of rate limits and quotas
- **Invalid Keys**: Clear error messages for configuration issues

## API Endpoints

### 1. Set Location
```http
POST /location/
Content-Type: application/json

{
  "lat": 40.7128,
  "lon": -74.0060,
  "city": "New York"
}
```

### 2. Get Location
```http
GET /location/
```

### 3. Validate Temperature Forecasting
```http
GET /temperature/validate/
```

### 4. Energy Forecast with Weather
```http
POST /forecast/
Content-Type: application/json

{
  "meter_type": "electricity",
  "start_date": "2024-12-15",
  "end_date": "2025-01-15"
}
```

## Response Format

### Forecast Response
```json
{
  "forecast_data": [
    {
      "datetime": "2024-12-15",
      "value": 125.5,
      "type": "historical"
    },
    {
      "datetime": "2025-01-15",
      "value": 135.7,
      "type": "forecast",
      "temperature": 22.8
    }
  ]
}
```

**Key Points:**
- **Historical dates**: No temperature (uses actual consumption data)
- **Future dates**: Includes temperature for forecasting
- **Type field**: Distinguishes between "historical" and "forecast" data

## Visual Crossing API Features

### 1. **Timeline API Format**
```
https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/{location}/{date1}/{date2}?key={api_key}
```

### 2. **Supported Parameters**
- **location**: lat,lon or address
- **date1/date2**: Date range (YYYY-MM-DD)
- **unitGroup**: metric, us, uk, base
- **include**: days, hours, current, alerts
- **elements**: temp, tempmax, tempmin, etc.

### 3. **Response Structure**
```json
{
  "latitude": 40.7128,
  "longitude": -74.0060,
  "timezone": "America/New_York",
  "days": [
    {
      "datetime": "2024-12-15",
      "temp": 22.8,
      "tempmax": 28.5,
      "tempmin": 17.2
    }
  ]
}
```

## Implementation Details

### 1. **Temperature Forecasting Logic**

```python
def get_historical_temperature_average(date, location=None):
    """
    - Past dates: Direct API call for historical data
    - Future dates: Average of same calendar date from previous 5 years
    - Fallback: Climate normal calculation
    """
```

### 2. **Optimization Strategies**

**Date Range Grouping**
- Groups consecutive dates into single API calls
- Reduces API usage from N calls to ~1-3 calls per request
- Handles mixed historical/forecast ranges intelligently

**Caching Strategy**
- Results cached per request session
- Avoids duplicate API calls for same dates
- Memory-efficient temporary storage

### 3. **Error Handling Hierarchy**

1. **Primary**: Visual Crossing API call
2. **Secondary**: Historical average calculation
3. **Tertiary**: Climate normal fallback
4. **Final**: Graceful failure with error message

## Free Tier Limits

### Visual Crossing Free Plan
- **Records per day**: 1000
- **Rate limiting**: Reasonable limits for API calls
- **Features**: Full access to historical and forecast data
- **Restrictions**: No credit card required

### Usage Optimization
- **Efficient grouping**: Minimize API calls through date range requests
- **Smart caching**: Avoid duplicate requests within same session
- **Fallback systems**: Reduce API dependency for edge cases

## Testing

### 1. **Validate API Integration**
```bash
curl "http://localhost:8000/temperature/validate/"
```

### 2. **Test Location Management**
```bash
# Set location
curl -X POST "http://localhost:8000/location/" \
  -H "Content-Type: application/json" \
  -d '{"lat": 40.7128, "lon": -74.0060, "city": "New York"}'

# Get location
curl "http://localhost:8000/location/"
```

### 3. **Test Forecast with Weather**
```bash
curl -X POST "http://localhost:8000/forecast/" \
  -H "Content-Type: application/json" \
  -d '{"meter_type": "electricity", "start_date": "2024-12-15", "end_date": "2025-01-15"}'
```

## Troubleshooting

### Common Issues

1. **"API key not configured"**
   - Set VISUAL_CROSSING_API_KEY environment variable
   - Or update config/weather_config.json

2. **"Invalid API key"**
   - Verify API key from Visual Crossing dashboard
   - Check for extra spaces or characters

3. **"Rate limit exceeded"**
   - Free tier: 1000 records/day
   - Wait for reset or upgrade plan

4. **"No weather data available"**
   - Check location coordinates
   - Verify date format (YYYY-MM-DD)
   - Check internet connectivity

### Debug Mode
Enable detailed logging by setting environment variable:
```bash
set WEATHER_DEBUG=true
```

## Migration from OpenWeatherMap

### Key Changes
1. **API Provider**: OpenWeatherMap → Visual Crossing
2. **Historical Data**: Seasonal approximations → Actual historical records
3. **API Structure**: Current weather + adjustments → Timeline API
4. **Configuration**: Updated config file structure

### Benefits of Migration
- ✅ **Real Historical Data**: No more seasonal approximations
- ✅ **Better Accuracy**: Actual weather records for temperature forecasting
- ✅ **Simpler Integration**: Single API for all weather data needs
- ✅ **No Credit Card**: Free tier without payment method requirement

## Support

### Visual Crossing Support
- **Documentation**: https://www.visualcrossing.com/resources/documentation/
- **Support Forum**: https://www.visualcrossing.com/support
- **API Status**: Check service status on their website

### System Support
- Check logs for detailed error messages
- Verify configuration files
- Test API connectivity with validation endpoint
- Review environment variable settings

## Future Enhancements

### Potential Improvements
1. **Advanced Caching**: Redis/database caching for frequently requested data
2. **Multiple Locations**: Support for meter-specific locations
3. **Weather Variables**: Expand beyond temperature (humidity, precipitation, etc.)
4. **Forecast Models**: Integration with multiple weather forecast models
5. **Historical Analysis**: Weather impact analysis on energy consumption patterns

### Scalability Considerations
- **API Quotas**: Monitor usage and implement quota management
- **Caching Strategy**: Implement persistent caching for historical data
- **Load Balancing**: Distribute API calls across multiple keys if needed
- **Data Storage**: Consider storing frequently accessed weather data locally 