#!/usr/bin/env python
"""
Test script for the new Dramatiq-based Energy Forecasting System

This script tests the complete workflow from task submission to completion.
"""
import requests
import json
import time
import sys
from datetime import datetime

def test_dramatiq_system(meter_id="meter_test_001", meter_type="electricity", base_url="http://localhost:8000"):
    """
    Comprehensive test of the Dramatiq-based training system
    """
    print("\n" + "="*80)
    print("🚀 Testing Dramatiq-based Energy Forecasting System")
    print("="*80)
    
    # Test 1: Health checks
    print("\n1. 🏥 Testing system health...")
    
    try:
        # General health check
        health_response = requests.get(f"{base_url}/health/")
        if health_response.status_code == 200:
            health_data = health_response.json()
            print(f"   ✅ API Health: {health_data['status']}")
            print(f"   📊 Broker: {health_data.get('broker', 'unknown')}")
            print(f"   🗄️  InfluxDB: {health_data.get('influxdb', 'unknown')}")
        else:
            print(f"   ❌ API Health check failed: {health_response.status_code}")
            return False
            
        # Worker health check
        worker_response = requests.get(f"{base_url}/health/worker")
        if worker_response.status_code == 200:
            worker_data = worker_response.json()
            print(f"   ✅ Worker Health: {worker_data['status']}")
            print(f"   🔧 Broker Type: {worker_data.get('broker_type', 'unknown')}")
        else:
            print(f"   ⚠️  Worker health check failed: {worker_response.status_code}")
            
    except Exception as e:
        print(f"   ❌ Health check failed: {str(e)}")
        return False
    
    # Test 2: Start training task
    print(f"\n2. 🎯 Starting training task for {meter_type} meter {meter_id}...")
    
    try:
        payload = {
            "meter_id": meter_id,
            "meter_type": meter_type
        }
        
        start_time = datetime.now()
        response = requests.post(f"{base_url}/trainmodel/", json=payload)
        
        if response.status_code != 200:
            print(f"   ❌ Failed to start training: {response.status_code}")
            print(f"   📄 Response: {response.text}")
            return False
        
        result = response.json()
        task_id = result.get("task_id")
        
        if not task_id:
            print(f"   ❌ No task_id returned: {result}")
            return False
            
        print(f"   ✅ Training task started successfully!")
        print(f"   🆔 Task ID: {task_id}")
        print(f"   📊 Initial Status: {result.get('status')}")
        print(f"   📈 Initial Progress: {result.get('progress', 0)}%")
        
    except Exception as e:
        print(f"   ❌ Failed to start training: {str(e)}")
        return False
    
    # Test 3: Monitor task progress
    print(f"\n3. 📊 Monitoring task progress...")
    print("   (This may take several minutes for model training)")
    
    max_attempts = 300  # 10 minutes timeout
    attempt = 0
    last_status = None
    last_progress = -1
    
    try:
        while attempt < max_attempts:
            attempt += 1
            
            # Check status
            status_response = requests.get(f"{base_url}/trainmodel/status/{task_id}")
            
            if status_response.status_code != 200:
                print(f"   ❌ Status check failed: {status_response.status_code}")
                break
            
            status_data = status_response.json()
            current_status = status_data.get("status")
            current_progress = status_data.get("progress", 0)
            current_message = status_data.get("message", "")
            
            # Only print updates when status or significant progress changes
            if current_status != last_status or current_progress >= last_progress + 5:
                elapsed_time = datetime.now() - start_time
                print(f"   📊 [{elapsed_time}] Status: {current_status} | Progress: {current_progress}% | {current_message}")
                
                # Show estimated completion if available
                if status_data.get("estimated_completion"):
                    est_completion = status_data["estimated_completion"]
                    print(f"      ⏰ Estimated completion: {est_completion}")
                
                last_status = current_status
                last_progress = current_progress
            
            # Check if task completed
            if current_status == "SUCCESS":
                elapsed_time = datetime.now() - start_time
                print(f"\n   🎉 Task completed successfully! (Total time: {elapsed_time})")
                
                # Show detailed results
                if status_data.get("result"):
                    result_details = status_data["result"]["details"]
                    print(f"   📈 Training Results:")
                    print(f"      📊 Data Points: {result_details.get('data_points', 'N/A')}")
                    print(f"      🎯 Model Path: {result_details.get('model_path', 'N/A')}")
                    print(f"      ⏱️  Data Fetch Time: {result_details.get('data_fetch_time_seconds', 'N/A')}s")
                    print(f"      🧠 Training Time: {result_details.get('training_time_seconds', 'N/A')}s")
                    print(f"      📅 Completed At: {result_details.get('training_completed', 'N/A')}")
                
                break
                
            elif current_status == "FAILED":
                print(f"\n   ❌ Task failed!")
                if status_data.get("error"):
                    print(f"      🚨 Error: {status_data['error']}")
                if status_data.get("error_details"):
                    print(f"      🔍 Error Type: {status_data['error_details'].get('type', 'unknown')}")
                return False
            
            # Wait before next check
            time.sleep(2)
        
        if attempt >= max_attempts:
            print(f"\n   ⏰ Timeout after {max_attempts} attempts. Task may still be running.")
            print(f"      You can continue checking with: GET {base_url}/trainmodel/status/{task_id}")
            return False
            
    except Exception as e:
        print(f"   ❌ Error monitoring task: {str(e)}")
        return False
    
    # Test 4: Verify model file exists (if training succeeded)
    if last_status == "SUCCESS":
        print(f"\n4. 📁 Verifying model file...")
        
        try:
            import os
            model_path = f"models/{meter_type}/{meter_id}.h5"
            
            if os.path.exists(model_path):
                file_size = os.path.getsize(model_path)
                print(f"   ✅ Model file created: {model_path}")
                print(f"   📊 File size: {file_size:,} bytes")
            else:
                print(f"   ⚠️  Model file not found: {model_path}")
                return False
                
        except Exception as e:
            print(f"   ❌ Error checking model file: {str(e)}")
    
    print(f"\n" + "="*80)
    print("🎉 Dramatiq System Test Completed Successfully!")
    print("="*80)
    print("\n📝 Summary:")
    print(f"   • Task ID: {task_id}")
    print(f"   • Meter: {meter_type} - {meter_id}")
    print(f"   • Final Status: {last_status}")
    print(f"   • Total Time: {datetime.now() - start_time}")
    print("\n🔗 Useful URLs:")
    print(f"   • API Docs: {base_url}/docs")
    print(f"   • Health Check: {base_url}/health/")
    print(f"   • Worker Health: {base_url}/health/worker")
    print(f"   • Task Status: {base_url}/trainmodel/status/{task_id}")
    
    return True

def main():
    """Main test function"""
    if len(sys.argv) > 1:
        meter_id = sys.argv[1]
    else:
        meter_id = "meter_test_001"
    
    if len(sys.argv) > 2:
        meter_type = sys.argv[2]
    else:
        meter_type = "electricity"
        
    if len(sys.argv) > 3:
        base_url = sys.argv[3]
    else:
        base_url = "http://localhost:8000"
    
    print(f"Testing with:")
    print(f"  Meter ID: {meter_id}")
    print(f"  Meter Type: {meter_type}")
    print(f"  API URL: {base_url}")
    
    success = test_dramatiq_system(meter_id, meter_type, base_url)
    
    if success:
        print("\n✅ All tests passed! The Dramatiq system is working correctly.")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed. Please check the logs and configuration.")
        sys.exit(1)

if __name__ == "__main__":
    main() 