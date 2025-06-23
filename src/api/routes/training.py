from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
import asyncio
import json
import os
import redis
from datetime import datetime

from src.schemas.models import TrainModelRequest, TaskStatus
from src.task_system import train_model_task, task_tracker, TaskState
from src.services.streaming_service import generate_task_updates, get_task_status, get_active_task, cancel_task
from src.logger_config import get_logger

router = APIRouter(prefix="/trainmodel", tags=["Model Training"])
logger = get_logger("energy_forecasting.api.training")

@router.post("/", response_model=TaskStatus)
def train_model(request: TrainModelRequest):
    """
    Train a model for a specific meter by fetching all historical data from InfluxDB.
    Returns a task ID that can be used to check the status.

    """
    meter_id = request.meter_id
    meter_type = request.meter_type.lower()
    
    # Validate meter type
    if meter_type not in ["electricity", "water"]:
        raise HTTPException(status_code=400, detail="meter_type must be 'electricity' or 'water'")
    
    import redis
    from src.task_system import task_tracker, TaskState
    
    try:
        logger.info("Training model request received", meter_id=meter_id, meter_type=meter_type)
        
        redis_client = redis.from_url(os.getenv('REDIS_URL', 'redis://localhost:6379/0'))
        active_task_key = "active_training_task"
        
        # STEP 1: IMMEDIATELY CANCEL ALL EXISTING TASKS
        try:
            old_active_task = redis_client.get(active_task_key)
            if old_active_task:
                old_task_id = old_active_task.decode()
                logger.info("CANCELLING previous task immediately", old_task_id=old_task_id)
                
                # Cancel the old task using the streaming service
                cancel_result = cancel_task(old_task_id)
                logger.info("Previous task cancelled", old_task_id=old_task_id, result=cancel_result)
                
            # Clear any stale active task marker
            redis_client.delete(active_task_key)
            
        except Exception as e:
            logger.warning("Could not cancel previous tasks", error=str(e))
        
        # STEP 2: Submit the new task
        message = train_model_task.send(meter_id, meter_type)
        task_id = message.message_id
        
        # STEP 3: Set this as the ONLY active task
        try:
            redis_client.setex(active_task_key, 3600, task_id)  # 1 hour expiration
            logger.info("New task set as ONLY active training task", 
                       task_id=task_id, meter_id=meter_id, meter_type=meter_type)
                
        except Exception as e:
            logger.warning("Could not set new active task", task_id=task_id, error=str(e))
        
        # STEP 4: Initialize progress tracking for new task
        task_tracker.update_progress(task_id, 
            TaskState.PENDING, 0, 
            "Task submitted - waiting to start", 
            {"meter_id": meter_id, "meter_type": meter_type})
        
        logger.info("NEW training task submitted and active", task_id=task_id, meter_id=meter_id, meter_type=meter_type)
        
        return TaskStatus(
            task_id=task_id,
            status="PENDING",
            progress=0
        )
        
    except Exception as e:
        logger.error("Failed to submit training task", error=str(e), meter_id=meter_id, meter_type=meter_type)
        raise HTTPException(status_code=500, detail=f"Failed to submit training task: {str(e)}")

@router.get("/status/{task_id}", response_model=TaskStatus)
def get_task_status_endpoint(task_id: str):
    """
    Get the status of a training task.

    """
    try:
        # Get task status from streaming service
        result_data = get_task_status(task_id)
        
        if result_data.get("status") == "NOT_FOUND":
            raise HTTPException(status_code=404, detail="Task not found or expired")
        
        if "error" in result_data and result_data.get("status") == "ERROR":
            raise HTTPException(status_code=500, detail=result_data["error"])
        
        # Return task status
        return TaskStatus(
            task_id=task_id,
            status=result_data.get("status", "UNKNOWN"),
            progress=result_data.get("progress", 0),
            result=result_data.get("result", {}),
            error=result_data.get("error"),
            error_details=result_data.get("error_details")
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get task status", task_id=task_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get task status: {str(e)}")

@router.get("/stream/{task_id}")
async def stream_task_status(task_id: str):
    """
    Stream task status updates using Server-Sent Events (SSE).
    
    Frontend can consume this with EventSource API:
    ```javascript
    const eventSource = new EventSource(`/trainmodel/stream/${taskId}`);
    eventSource.onmessage = (event) => {
        const data = JSON.parse(event.data);
        // Update UI with data.progress, data.status, etc.
        if (data.status === 'SUCCESS' || data.status === 'FAILED') {
            eventSource.close();
        }
    };
    ```
    """
    return StreamingResponse(
        generate_task_updates(task_id),
        media_type="text/event-stream"
    )

@router.get("/active_task")
def get_active_task_endpoint():
    """
    Get the currently active training task.
    """
    try:
        # Get active task from streaming service
        active_task = get_active_task()
        
        if "error" in active_task:
            raise HTTPException(status_code=500, detail=active_task["error"])
            
        return active_task
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get active task", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get active task: {str(e)}")

@router.post("/cancel/{task_id}")
def cancel_task_endpoint(task_id: str):
    """
    Cancel a running task.
    """
    try:
        # Cancel task using streaming service
        result = cancel_task(task_id)
        
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
            
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to cancel task", task_id=task_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to cancel task: {str(e)}") 