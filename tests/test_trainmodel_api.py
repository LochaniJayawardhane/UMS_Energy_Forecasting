#!/usr/bin/env python
import requests
import json
import time
import sys

def test_train_model_api(meter_id, meter_type):
    """
    Test the /trainmodel/ API endpoint
    
    Args:
        meter_id: ID of the meter
        meter_type: Type of meter (electricity/water)
    """
    base_url = "http://localhost:8000"
    
    # 1. Start the training process
    print(f"\n1. Starting training process for {meter_type} meter {meter_id}...")
    
    payload = {
        "meter_id": meter_id,
        "meter_type": meter_type
    }
    
    response = requests.post(
        f"{base_url}/trainmodel/",
        json=payload
    )
    
    if response.status_code != 200:
        print(f"Error: {response.status_code}")
        print(response.text)
        sys.exit(1)
    
    result = response.json()
    print(f"Response: {json.dumps(result, indent=2)}")
    
    task_id = result.get("task_id")
    if not task_id:
        print("Error: No task_id returned")
        sys.exit(1)
    
    # 2. Check the task status until completion or error
    print(f"\n2. Checking task status for task {task_id}...")
    
    # Increased from 30 to 450 to accommodate 15 minutes of training time (450 * 2 seconds = 900 seconds = 15 minutes)
    max_attempts = 450
    attempt = 0
    
    while attempt < max_attempts:
        attempt += 1
        
        response = requests.get(f"{base_url}/trainmodel/status/{task_id}")
        
        if response.status_code != 200:
            print(f"Error checking status: {response.status_code}")
            print(response.text)
            break
        
        status_result = response.json()
        status = status_result.get("status")
        progress = status_result.get("progress", 0)
        
        print(f"Status: {status}, Progress: {progress}%")
        
        if status == "SUCCESS":
            print("\nTask completed successfully!")
            print(f"Result: {json.dumps(status_result.get('result', {}), indent=2)}")
            break
        
        elif status == "FAILED":
            print("\nTask failed!")
            print(f"Error: {status_result.get('error', 'Unknown error')}")
            break
        
        # Wait before checking again
        time.sleep(2)
    
    if attempt >= max_attempts:
        print(f"\nGave up after {max_attempts} attempts. Task may still be running.")

if __name__ == "__main__":
    # You can change these values to test different meters
    test_meter_id = "meter_test_001"
    test_meter_type = "electricity"
    
    test_train_model_api(test_meter_id, test_meter_type) 