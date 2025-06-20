#!/usr/bin/env python
"""
Debug script to test each component individually
"""

def test_config_loading():
    """Test the configuration loading"""
    print("1. Testing configuration loading...")
    try:
        from config.influxdb_config import get_influxdb_config, load_influxdb_config
        
        # Test dict version
        config = load_influxdb_config()
        print(f"   ✅ load_influxdb_config(): {list(config.keys())}")
        
        # Test tuple version  
        result = get_influxdb_config()
        print(f"   ✅ get_influxdb_config(): {len(result)} items")
        print(f"   Values: {result}")
        
        return True
    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_influx_client():
    """Test InfluxClient initialization"""
    print("\n2. Testing InfluxClient initialization...")
    try:
        from influx_client import InfluxClient
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
    print("\n3. Testing data retrieval...")
    try:
        from influx_client import InfluxClient
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

def test_training_functions():
    """Test training function imports"""
    print("\n4. Testing training function imports...")
    try:
        from utils import train_electricity_model, train_water_model
        print(f"   ✅ Training functions imported successfully")
        return True
    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_celery_imports():
    """Test Celery-related imports"""
    print("\n5. Testing Celery imports...")
    try:
        from celery_app import celery_app
        print(f"   ✅ celery_app imported: {celery_app}")
        
        from tasks import train_model_task
        print(f"   ✅ train_model_task imported: {train_model_task}")
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
        
        # Check if config file exists
        config_file = "config/influxdb_config.json"
        print(f"   Config file exists: {os.path.exists(config_file)}")
        
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
        test_config_loading, 
        test_influx_client,
        test_data_retrieval,
        test_training_functions,
        test_celery_imports
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
        print("✅ All components working - issue might be in Celery execution context")
    else:
        print("❌ Found component issues - these might be causing the Celery error")
        
    print("\nIf all tests pass, the issue is likely:")
    print("1. Working directory mismatch in Celery worker")
    print("2. Import path issues in Celery context") 
    print("3. Environment differences between direct and Celery execution") 