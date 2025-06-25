#!/usr/bin/env python
"""
Debug script to test each component individually
"""
import os
import sys
from pathlib import Path

# Add the project root directory to the Python path
project_root = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv, find_dotenv

def test_env_loading():
    """Test environment variable loading"""
    print("1. Testing environment variable loading...")
    try:
        # Try to locate and load the .env file
        dotenv_path = find_dotenv()
        if not dotenv_path:
            print(f"   ⚠️ No .env file found")
        else:
            print(f"   ✅ .env file found at {dotenv_path}")
            # Specify encoding to avoid UTF-8 issues
            load_dotenv(dotenv_path, encoding="latin-1")
        
        # Check for required environment variables
        influx_vars = ['INFLUXDB_URL', 'INFLUXDB_TOKEN', 'INFLUXDB_ORG', 'INFLUXDB_BUCKET']
        weather_vars = ['VISUAL_CROSSING_API_KEY', 'VISUAL_CROSSING_UNITS', 'VISUAL_CROSSING_BASE_URL']
        location_vars = ['LOCATION_LAT', 'LOCATION_LON', 'LOCATION_CITY']
        
        missing = []
        for var in influx_vars + weather_vars + location_vars:
            if not os.getenv(var):
                missing.append(var)
        
        if missing:
            print(f"   ⚠️ Missing environment variables: {', '.join(missing)}")
        else:
            print(f"   ✅ All required environment variables found")
            
        return True
    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_config_loading():
    """Test the configuration loading"""
    print("\n2. Testing configuration loading...")
    try:
        from config.influxdb_config import get_influxdb_config, load_influxdb_config
        from config.weather_config import load_weather_config
        
        # Test influxdb dict version
        config = load_influxdb_config()
        print(f"   ✅ load_influxdb_config(): {list(config.keys())}")
        
        # Test influxdb tuple version  
        result = get_influxdb_config()
        print(f"   ✅ get_influxdb_config(): {len(result)} items")
        print(f"   Values: {result}")
        
        # Test weather config
        weather_config = load_weather_config()
        print(f"   ✅ load_weather_config(): {list(weather_config.keys())}")
        print(f"   Visual Crossing API key: {weather_config['visual_crossing']['api_key'][:5]}...")
        
        return True
    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_influx_client():
    """Test InfluxClient initialization"""
    print("\n3. Testing InfluxClient initialization...")
    try:
        from src.influx_client import InfluxClient
        client = InfluxClient()
        print(f"   ✅ InfluxClient created: {client.url}")
        client.close()
        return True
    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_data_retrieval():
    """Test data retrieval"""
    print("\n4. Testing data retrieval...")
    try:
        from src.influx_client import InfluxClient
        client = InfluxClient()
        
        data = client.get_meter_data("meter_test_001", "electricity", limit=1)
        print(f"   ✅ Data retrieved: {len(data)} rows")
        
        client.close()
        return True
    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_weather_api():
    """Test weather API functions"""
    print("\n5. Testing weather API functions...")
    try:
        from weather_utils.weather import get_temperature_series
        from datetime import datetime, timedelta
        
        # Test with a small date range
        start_date = datetime.now() - timedelta(days=5)
        end_date = datetime.now() - timedelta(days=3)
        
        temperature_series = get_temperature_series(start_date, end_date)
        print(f"   ✅ Temperature series retrieved: {len(temperature_series)} data points")
        
        return True
    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_working_directory():
    """Test working directory and file access"""
    print("\n6. Testing working directory and files...")
    try:
        import os
        print(f"   Working directory: {os.getcwd()}")
        
        # Check if .env file exists
        env_file = ".env"
        print(f"   .env file exists: {os.path.exists(env_file)}")
        
        # Check if models directory exists
        models_dir = "models"
        print(f"   Models directory exists: {os.path.exists(models_dir)}")
        
        return True
    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🔧 COMPONENT DEBUG TEST")
    print("=" * 50)
    
    tests = [
        test_working_directory,
        test_env_loading,
        test_config_loading,
        test_influx_client,
        test_data_retrieval,
        test_weather_api
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"   💥 UNEXPECTED ERROR in {test.__name__}: {e}")
            results.append(False)
    
    print("\n" + "=" * 50)
    print("SUMMARY:")
    print("=" * 50)
    
    passed = sum(results)
    total = len(results)
    
    print(f"Tests passed: {passed}/{total}")
    
    if all(results):
        print("✅ All components working")
    else:
        print("❌ Found component issues that need to be addressed") 