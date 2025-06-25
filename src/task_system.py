import dramatiq
import json
import traceback
import time
import os
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
dotenv_path = Path(__file__).parent.parent / '.env'
if dotenv_path.exists():
    load_dotenv(dotenv_path=dotenv_path, encoding="latin-1")

import pandas as pd
from dramatiq.results import Results
from dramatiq.results.backends import RedisBackend

from src.dramatiq_broker import broker, get_broker
from src.influx_client import InfluxClient
from src.utils import train_electricity_model, train_water_model
from src.logger_config import TaskLogger, get_logger

logger = get_logger("task_system")

class TaskState(Enum):
    """Task states for better tracking"""
    PENDING = "PENDING"
    STARTED = "STARTED"
    CONNECTING = "CONNECTING"
    FETCHING_DATA = "FETCHING_DATA"
    VALIDATING_DATA = "VALIDATING_DATA"
    PROCESSING_DATA = "PROCESSING_DATA"
    TRAINING_MODEL = "TRAINING_MODEL"
    SAVING_MODEL = "SAVING_MODEL"
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"

@dataclass
class TaskProgress:
    """Enhanced task progress tracking"""
    task_id: str
    state: TaskState
    progress: int  # 0-100
    message: str
    details: Dict[str, Any]
    started_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    estimated_completion: Optional[datetime] = None
    error: Optional[str] = None
    error_type: Optional[str] = None
    retry_count: int = 0
    
    def to_dict(self):
        """Convert to dictionary for JSON serialization"""
        result = asdict(self)
        result['state'] = self.state.value
        # Convert datetime objects to ISO strings
        for key in ['started_at', 'updated_at', 'estimated_completion']:
            if result[key] is not None:
                result[key] = result[key].isoformat()
        return result

class TaskTracker:
    """Enhanced task tracking with Redis backend"""
    
    def __init__(self):
        self.logger = get_logger("task_tracker")
        # Initialize Redis client from broker
        try:
            broker = get_broker()
            for middleware in broker.middleware:
                if isinstance(middleware, Results):
                    self.redis_client = middleware.backend.client
                    break
            else:
                # Fallback: connect directly to Redis
                import redis
                redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
                self.redis_client = redis.from_url(redis_url)
        except Exception as e:
            self.logger.error("Failed to initialize TaskTracker", error=str(e))
            raise
    
    def update_progress(self, task_id: str, state: TaskState, progress: int, 
                       message: str, details: Dict[str, Any] = None,
                       error: str = None, error_type: str = None):
        """Update task progress with comprehensive tracking"""
        now = datetime.now()
        
        # Get existing progress or create new
        existing = self.get_progress(task_id)
        if existing:
            task_progress = TaskProgress(
                task_id=task_id,
                state=state,
                progress=progress,
                message=message,
                details={**existing.get('details', {}), **(details or {})},
                started_at=datetime.fromisoformat(existing['started_at']) if existing.get('started_at') else now,
                updated_at=now,
                error=error,
                error_type=error_type,
                retry_count=existing.get('retry_count', 0)
            )
        else:
            task_progress = TaskProgress(
                task_id=task_id,
                state=state,
                progress=progress,
                message=message,
                details=details or {},
                started_at=now,
                updated_at=now,
                error=error,
                error_type=error_type
            )
        
        # Calculate estimated completion
        if state not in [TaskState.SUCCESS, TaskState.FAILED] and task_progress.started_at:
            elapsed = now - task_progress.started_at
            if progress > 0:
                total_estimated = elapsed * (100 / progress)
                task_progress.estimated_completion = task_progress.started_at + total_estimated
        
        # Store in Redis with 1 hour expiration
        key = f"task_progress:{task_id}"
        progress_data = task_progress.to_dict()
        
        try:
            self.redis_client.setex(key, 3600, json.dumps(progress_data))
            self.logger.debug("Progress updated", task_id=task_id, state=state.value, 
                            progress=progress, message=message)
            
            # Publish update to Redis Pub/Sub for real-time SSE streaming
            self._publish_update(task_id, progress_data)
            
        except Exception as e:
            self.logger.error("Failed to update progress", task_id=task_id, error=str(e))
    
    def _publish_update(self, task_id: str, progress_data: Dict[str, Any]):
        """
        Publish task progress update to Redis Pub/Sub channel for real-time streaming.
        
        This enables Server-Sent Events (SSE) to receive real-time updates
        without polling the status endpoint.
        """
        try:
            channel = f"task_updates:{task_id}"
            message = json.dumps(progress_data)
            
            # Publish to Redis channel
            published = self.redis_client.publish(channel, message)
            
            if published > 0:
                self.logger.debug("Update published to SSE channel", 
                                task_id=task_id, channel=channel, subscribers=published)
            else:
                self.logger.debug("No SSE subscribers for task", task_id=task_id, channel=channel)
                
        except Exception as e:
            # Don't fail the main operation if pub/sub fails
            self.logger.warning("Failed to publish SSE update", task_id=task_id, error=str(e))
    
    def get_progress(self, task_id: str) -> Optional[Dict]:
        """Get task progress from Redis"""
        key = f"task_progress:{task_id}"
        try:
            data = self.redis_client.get(key)
            if data:
                return json.loads(data)
            return None
        except Exception as e:
            self.logger.error("Failed to get progress", task_id=task_id, error=str(e))
            return None
    
    def mark_started(self, task_id: str):
        """Mark task as started"""
        self.update_progress(task_id, TaskState.STARTED, 0, "Task started", 
                           {"worker_pid": os.getpid()})

# Global task tracker
task_tracker = TaskTracker()

@dramatiq.actor(store_results=True, max_retries=0, time_limit=None)  # No retries for singleton behavior, no time limit
def train_model_task(meter_id: str, meter_type: str, task_options: Dict[str, Any] = None):
    """
    Enhanced model training task with SINGLETON behavior and comprehensive logging.
    
    SINGLETON BEHAVIOR: Only ONE training task can run at a time across the entire system.
    This task checks if it should continue running before starting heavy work.
    If another task was started after this one, this task will cancel itself.
    """
    # Get task ID from message context
    current_message = dramatiq.middleware.CurrentMessage.get_current_message()
    task_id = current_message.message_id
    
    # Setup task logger
    task_logger = TaskLogger("train_model", task_id).bind(
        meter_id=meter_id, 
        meter_type=meter_type
    )
    
    task_logger.info("Starting model training task")
    
    # SINGLETON CHECK: Set this as the active task and check for conflicts
    active_task_key = "active_training_task"
    
    def check_if_cancelled():
        """Check if this task should be cancelled due to singleton behavior"""
        try:
            current_active = task_tracker.redis_client.get(active_task_key)
            if current_active and current_active.decode() != task_id:
                return True
        except:
            pass
        return False
    
    def cancel_self(reason="Cancelled - newer training task started"):
        """Cancel this task gracefully"""
        task_logger.info("Task cancelled", reason=reason)
        task_tracker.update_progress(task_id, TaskState.FAILED, 0, reason, 
                                   error=reason, error_type="CANCELLED")
        return {'status': 'cancelled', 'reason': reason, 'task_id': task_id}
    
    try:
        # Check if this task is the current active task
        # (The API should have already set this task as active)
        existing_active = task_tracker.redis_client.get(active_task_key)
        if existing_active and existing_active.decode() != task_id:
            # Another task is already active, cancel this one immediately
            return cancel_self("Cancelled - newer task is already active")
        
        # If no active task set, set this one as active (fallback)
        if not existing_active:
            task_tracker.redis_client.setex(active_task_key, 3600, task_id)
            task_logger.info("Task set as active singleton (fallback)", active_task_id=task_id)
        else:
            task_logger.info("Task confirmed as active singleton", active_task_id=task_id)
        
        task_tracker.mark_started(task_id)
        
        # Validate inputs
        if meter_type not in ["electricity", "water"]:
            raise ValueError(f"Invalid meter_type: {meter_type}")
        
        # Check for cancellation before heavy work
        if check_if_cancelled():
            return cancel_self()
        
        # Step 1: Connect to InfluxDB (5%)
        task_tracker.update_progress(task_id, TaskState.CONNECTING, 5, 
                                   "Connecting to InfluxDB...", 
                                   {"step": "database_connection"})
        task_logger.info("Connecting to InfluxDB")
        
        influx_client = InfluxClient()
        task_logger.info("InfluxDB connection established")
        
        # Check for cancellation
        if check_if_cancelled():
            influx_client.close()
            return cancel_self()
        
        # Step 2: Fetch data (10-30%)
        task_tracker.update_progress(task_id, TaskState.FETCHING_DATA, 10, 
                                   "Fetching training data...", 
                                   {"step": "data_fetch", "meter_id": meter_id})
        task_logger.info("Fetching training data from InfluxDB")
        
        data_start_time = time.time()
        data = influx_client.get_meter_data(meter_id=meter_id, meter_type=meter_type)
        data_fetch_time = time.time() - data_start_time
        influx_client.close()
        
        task_logger.info("Data fetched successfully", 
                       records_count=len(data), 
                       fetch_time_seconds=round(data_fetch_time, 2))
        
        # Check for cancellation
        if check_if_cancelled():
            return cancel_self()
        
        # Step 3: Validate data (35%)
        task_tracker.update_progress(task_id, TaskState.VALIDATING_DATA, 35, 
                                   "Validating training data...", 
                                   {"step": "data_validation", "records_count": len(data)})
        
        if data.empty:
            error_msg = f"No training data found for {meter_type} meter {meter_id}"
            task_logger.error("Data validation failed - no data")
            task_tracker.update_progress(task_id, TaskState.FAILED, 35, 
                                       "Validation failed", 
                                       error=error_msg, error_type="NO_DATA_FOUND")
            # Clear active task on failure
            task_tracker.redis_client.delete(active_task_key)
            return {'status': 'error', 'message': error_msg, 'error_type': 'NO_DATA_FOUND'}
        
        if len(data) < 100:
            error_msg = f"Insufficient data for {meter_type} meter {meter_id}. Found {len(data)} records"
            task_logger.warning("Insufficient training data", records_found=len(data))
            task_tracker.update_progress(task_id, TaskState.FAILED, 35, 
                                       "Insufficient data", {"records_found": len(data)},
                                       error=error_msg, error_type="INSUFFICIENT_DATA")
            # Clear active task on failure
            task_tracker.redis_client.delete(active_task_key)
            return {'status': 'error', 'message': error_msg, 'error_type': 'INSUFFICIENT_DATA'}
        
        task_logger.info("Data validation passed", records_count=len(data))
        
        # Check for cancellation before training
        if check_if_cancelled():
            return cancel_self()
        
        # Step 4: Model training (50-85%)
        task_tracker.update_progress(task_id, TaskState.TRAINING_MODEL, 50, 
                                   "Training model...", 
                                   {"step": "model_training", "algorithm": "XGBoost"})
        task_logger.info("Starting model training")
        
        training_start_time = time.time()
        
        # Check for cancellation during training setup
        if check_if_cancelled():
            return cancel_self()
        
        if meter_type == "electricity":
            model = train_electricity_model(data)
        else:
            model = train_water_model(data)
            
        training_time = time.time() - training_start_time
        task_logger.info("Model training completed", training_time_seconds=round(training_time, 2))
        
        # Final check before saving
        if check_if_cancelled():
            return cancel_self()
        
        # Step 5: Save model (90%)
        task_tracker.update_progress(task_id, TaskState.SAVING_MODEL, 90, 
                                   "Saving trained model...", {"step": "model_saving"})
        task_logger.info("Saving trained model")
        
        # Get project root directory (parent of src)
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_dir = os.path.join(project_root, f"models/{meter_type}")
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, f"{meter_id}.h5")
        
        model.save_model(model_path)
        task_logger.info("Model saved successfully", model_path=model_path)
        
        # Task completed (100%)
        completion_time = datetime.now()
        success_details = {
            'meter_id': meter_id,
            'meter_type': meter_type,
            'data_points': len(data),
            'model_path': model_path,
            'training_completed': completion_time.isoformat(),
            'data_fetch_time_seconds': data_fetch_time,
            'training_time_seconds': training_time
        }
        
        task_tracker.update_progress(task_id, TaskState.SUCCESS, 100, 
                                   "Model training completed successfully!", 
                                   success_details)
        
        task_logger.info("Task completed successfully", **success_details)
        
        # Clear the active task marker on success
        task_tracker.redis_client.delete(active_task_key)
        
        return {
            'status': 'success',
            'message': f'Model trained and saved at {model_path}',
            'details': success_details
        }
        
    except Exception as e:
        error_message = f"Error during model training: {str(e)}"
        error_traceback = traceback.format_exc()
        
        task_logger.error("Task failed with exception", error=error_message, traceback=error_traceback)
        
        # Classify error type
        if "Connection" in str(e) or "connection" in str(e).lower():
            error_type = "CONNECTION_ERROR"
        elif "memory" in str(e).lower():
            error_type = "MEMORY_ERROR"
        elif "timeout" in str(e).lower():
            error_type = "TIMEOUT_ERROR"
        else:
            error_type = "TRAINING_ERROR"
        
        task_tracker.update_progress(task_id, TaskState.FAILED, 0, 
                                   "Task failed", {"error_traceback": error_traceback},
                                   error=error_message, error_type=error_type)
        
        # Clear the active task marker on failure
        try:
            task_tracker.redis_client.delete(active_task_key)
        except:
            pass
        
        return {
            'status': 'error',
            'message': error_message,
            'error_type': error_type,
            'task_id': task_id
        }
