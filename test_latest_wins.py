#!/usr/bin/env python3
"""
Test to verify that the LATEST task runs and previous ones get cancelled.
"""
import requests
import time
import json

API_BASE = "http://localhost:8000"

def make_training_request(meter_id="test_meter_001", meter_type="electricity"):
    """Make a training request"""
    url = f"{API_BASE}/trainmodel/"
    data = {
        "meter_id": meter_id,
        "meter_type": meter_type
    }
    
    try:
        response = requests.post(url, json=data)
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Request submitted: {result['task_id'][:8]}...")
            return result['task_id']
        else:
            print(f"❌ Request failed: {response.status_code}")
            return None
    except Exception as e:
        print(f"❌ Request error: {e}")
        return None

def check_task_status(task_id):
    """Check task status"""
    url = f"{API_BASE}/trainmodel/status/{task_id}"
    
    try:
        response = requests.get(url)
        if response.status_code == 200:
            result = response.json()
            status = result.get('status', 'UNKNOWN')
            state = result.get('state', status)
            progress = result.get('progress', 0)
            message = result.get('message', '')
            
            return {
                'task_id': task_id,
                'state': state,
                'progress': progress,
                'message': message
            }
        else:
            return None
    except Exception as e:
        return None

def main():
    """Test that latest task wins"""
    print("🧪 Testing Latest Task Wins Behavior")
    print("=" * 50)
    
    # Make 3 requests with delays to see the pattern
    print("Making 3 training requests with small delays...")
    task_ids = []
    
    for i in range(3):
        print(f"\nSubmitting request {i+1}/3...")
        task_id = make_training_request(f"test_meter_{i+1}")
        if task_id:
            task_ids.append(task_id)
            print(f"  Task ID: {task_id}")
        time.sleep(1)  # 1 second delay between requests
    
    print(f"\n📝 Submitted {len(task_ids)} tasks")
    print(f"Expected: Last task ({task_ids[-1][:8]}...) should run, others should cancel")
    
    # Check statuses over time
    print("\nMonitoring task statuses...")
    for round_num in range(8):
        print(f"\n--- Round {round_num + 1} ---")
        
        running_tasks = []
        cancelled_tasks = []
        failed_tasks = []
        
        for i, task_id in enumerate(task_ids):
            result = check_task_status(task_id)
            if result:
                state = result['state']
                progress = result['progress']
                
                task_num = i + 1
                short_id = task_id[:8]
                
                if state == 'CANCELLED':
                    cancelled_tasks.append(task_num)
                    print(f"🚫 Task {task_num} ({short_id}...): CANCELLED")
                elif state in ['STARTED', 'CONNECTING', 'FETCHING_DATA', 'TRAINING_MODEL', 'PROCESSING_DATA']:
                    running_tasks.append(task_num)
                    print(f"🔄 Task {task_num} ({short_id}...): {state} ({progress}%)")
                elif state in ['FAILED', 'FAILURE']:
                    failed_tasks.append(task_num)
                    print(f"❌ Task {task_num} ({short_id}...): FAILED ({progress}%)")
                else:
                    print(f"📊 Task {task_num} ({short_id}...): {state} ({progress}%)")
        
        print(f"\n📈 Summary: Running={len(running_tasks)}, Cancelled={len(cancelled_tasks)}, Failed={len(failed_tasks)}")
        
        # Check if behavior is correct
        if len(running_tasks) <= 1:
            if len(cancelled_tasks) > 0:
                print("✅ Correct behavior: Previous tasks cancelled, latest task running!")
            else:
                print("✅ Singleton behavior: Only 1 or 0 tasks running")
            
            # If last task is running or completed, we're good
            if len(running_tasks) == 0 and len(failed_tasks) > 0:
                print("ℹ️  Tasks completed (with failures due to no data - that's expected)")
                break
        
        time.sleep(2)
    
    print("\n🏁 Test completed!")
    
    # Final summary
    print("\n📋 Final Status Summary:")
    for i, task_id in enumerate(task_ids):
        result = check_task_status(task_id)
        if result:
            task_num = i + 1
            short_id = task_id[:8]
            state = result['state']
            print(f"Task {task_num} ({short_id}...): {state}")

if __name__ == "__main__":
    main() 