"""
Health service for system status and monitoring.
"""
import os
import redis
import psutil
import time
from datetime import datetime
from typing import Dict

from src.logger_config import get_logger

logger = get_logger("energy_forecasting.services.health")

def get_system_health() -> Dict:
    """
    Get basic system health information.
    
    """
    try:
        # Get system info
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "version": "1.0.0",
            "system_info": {
                "cpu_percent": psutil.cpu_percent(interval=0.1),
                "memory_percent": memory.percent,
                "disk_percent": disk.percent,
                "uptime": time.time() - psutil.boot_time()
            }
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "timestamp": datetime.now().isoformat(),
            "error": str(e)
        }

def get_worker_health() -> Dict:
    """
    Check the health of the Dramatiq worker system.

    """
    try:
        # Check Redis connection
        redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
        redis_client = redis.from_url(redis_url)
        redis_info = redis_client.info()
        
        # Get active task if any
        active_task = redis_client.get("active_training_task")
        active_task_id = active_task.decode() if active_task else None
        
        # Check if broker is available
        from src.dramatiq_broker import broker
        broker_info = {
            "broker_type": broker.__class__.__name__,
            "middleware_count": len(broker.middleware),
        }
        
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "redis": {
                "connected": True,
                "version": redis_info.get("redis_version", "unknown"),
                "used_memory_human": redis_info.get("used_memory_human", "unknown"),
                "connected_clients": redis_info.get("connected_clients", 0)
            },
            "broker": broker_info,
            "active_task": active_task_id
        }
    except Exception as e:
        logger.error(f"Worker health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "timestamp": datetime.now().isoformat(),
            "error": str(e)
        } 