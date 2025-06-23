"""
Streaming service for Server-Sent Events (SSE) logic.
"""
import os
import json
import asyncio
from typing import Dict, AsyncGenerator, Optional
from datetime import datetime

from src.task_system import task_tracker, TaskState
from src.logger_config import get_logger

logger = get_logger("energy_forecasting.services.streaming")

async def generate_task_updates(task_id: str) -> AsyncGenerator[str, None]:
    """
    Generate SSE updates for a task using Redis Pub/Sub.
    
    """
    try:
        # Import Redis
        import redis
        
        # Connect to Redis
        redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
        redis_client = redis.from_url(redis_url)
        
        # Create a new connection for pub/sub to avoid blocking
        pubsub = redis_client.pubsub()
        
        # Subscribe to the task's update channel
        channel = f"task_updates:{task_id}"
        pubsub.subscribe(channel)
        
        logger.info("SSE stream opened", task_id=task_id, channel=channel)
        
        # Send the current status immediately
        current_status = task_tracker.get_progress(task_id)
        if current_status:
            # Format as SSE event
            yield f"data: {json.dumps(current_status)}\n\n"
        else:
            # Task not found
            yield f"data: {json.dumps({'task_id': task_id, 'status': 'NOT_FOUND', 'error': 'Task not found'})}\n\n"
            return
        
        # Keep connection open for updates
        try:
            # Set a timeout for checking messages
            while True:
                # Check for new messages with timeout
                message = pubsub.get_message(timeout=1.0)
                
                if message and message['type'] == 'message':
                    data = message['data'].decode('utf-8')
                    # Send as SSE event
                    yield f"data: {data}\n\n"
                    
                    # Parse the message to check for terminal states
                    try:
                        status_data = json.loads(data)
                        status = status_data.get('status')
                        
                        # Close stream if task reached terminal state
                        if status in ['SUCCESS', 'FAILED', 'CANCELLED']:
                            logger.info("Task reached terminal state, closing SSE stream", 
                                      task_id=task_id, status=status)
                            break
                    except json.JSONDecodeError:
                        pass
                
                # Short sleep to avoid CPU spinning
                await asyncio.sleep(0.1)
                
        except Exception as e:
            logger.error("Error in SSE stream", task_id=task_id, error=str(e))
            # Send error as SSE event
            yield f"data: {json.dumps({'task_id': task_id, 'error': str(e)})}\n\n"
        finally:
            # Clean up pub/sub connection
            try:
                pubsub.unsubscribe(channel)
                pubsub.close()
            except:
                pass
            
            logger.info("SSE stream closed", task_id=task_id)
            
    except Exception as e:
        logger.error("Failed to create SSE stream", task_id=task_id, error=str(e))
        # Send error as SSE event
        yield f"data: {json.dumps({'task_id': task_id, 'error': str(e)})}\n\n"

def get_task_status(task_id: str) -> Dict:
    """
    Get the status of a task.
    
    """
    try:
        # Get task status from TaskTracker
        result_data = task_tracker.get_progress(task_id)
        
        if not result_data:
            # Task not found in progress tracking
            return {
                "task_id": task_id,
                "status": "NOT_FOUND",
                "progress": 0,
                "error": "Task not found or expired"
            }
        
        # Return task status
        return {
            "task_id": task_id,
            "status": result_data.get("state", "UNKNOWN"),
            "progress": result_data.get("progress", 0),
            "result": result_data.get("details", {}),
            "error": result_data.get("error"),
            "error_details": {"error_type": result_data.get("error_type")} if result_data.get("error_type") else None
        }
        
    except Exception as e:
        logger.error("Failed to get task status", task_id=task_id, error=str(e))
        return {
            "task_id": task_id,
            "status": "ERROR",
            "error": f"Failed to get task status: {str(e)}"
        }

def get_active_task() -> Dict:
    """
    Get the currently active training task, if any.
    
    """
    try:
        import redis
        
        redis_client = redis.from_url(os.getenv('REDIS_URL', 'redis://localhost:6379/0'))
        active_task_key = "active_training_task"
        
        # Get active task ID
        active_task = redis_client.get(active_task_key)
        
        if active_task:
            task_id = active_task.decode()
            
            # Get task details
            task_details = task_tracker.get_progress(task_id)
            
            if task_details:
                return {
                    "active_task_id": task_id,
                    "status": task_details.get("state", "UNKNOWN"),
                    "progress": task_details.get("progress", 0),
                    "details": task_details.get("details", {})
                }
            else:
                return {
                    "active_task_id": task_id,
                    "status": "UNKNOWN",
                    "message": "Task ID found but no details available"
                }
        else:
            return {
                "active_task_id": None,
                "message": "No active training task"
            }
            
    except Exception as e:
        logger.error("Failed to get active task", error=str(e))
        return {
            "active_task_id": None,
            "error": f"Failed to get active task: {str(e)}"
        }

def cancel_task(task_id: str) -> Dict:
    """
    Cancel a running task.
    
    """
    try:
        import redis
        
        # Update task status to CANCELLED
        task_tracker.update_progress(
            task_id, 
            TaskState.CANCELLED, 
            0, 
            "Task cancelled by user", 
            {"cancelled_by": "user", "cancelled_at": datetime.now().isoformat()},
            error="Cancelled by user", 
            error_type="CANCELLED"
        )
        
        # Broadcast cancellation to SSE streams
        redis_client = redis.from_url(os.getenv('REDIS_URL', 'redis://localhost:6379/0'))
        cancel_message = {
            "task_id": task_id,
            "status": "CANCELLED", 
            "progress": 0,
            "message": "Task cancelled by user",
            "error": "Cancelled by user",
            "error_type": "CANCELLED",
            "updated_at": datetime.now().isoformat()
        }
        redis_client.publish(f"task_updates:{task_id}", json.dumps(cancel_message))
        
        # If this is the active task, clear it
        active_task_key = "active_training_task"
        active_task = redis_client.get(active_task_key)
        if active_task and active_task.decode() == task_id:
            redis_client.delete(active_task_key)
            
        logger.info("Task cancelled", task_id=task_id)
        return {
            "task_id": task_id,
            "status": "CANCELLED",
            "message": "Task cancelled successfully"
        }
            
    except Exception as e:
        logger.error("Failed to cancel task", task_id=task_id, error=str(e))
        return {
            "task_id": task_id,
            "status": "ERROR",
            "error": f"Failed to cancel task: {str(e)}"
        } 