import dramatiq
import os
import redis
from dramatiq.brokers.redis import RedisBroker
from dramatiq.middleware import CurrentMessage, Retries, TimeLimit, Callbacks, Pipelines
from dramatiq.results import Results
from dramatiq.results.backends import RedisBackend
from logger_config import get_logger

logger = get_logger("dramatiq.broker")

def create_broker():
    """
    Create and configure Dramatiq broker with Redis backend
    """
    # Redis connection
    redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
    logger.info("Setting up Dramatiq broker", redis_url=redis_url)
    
    try:
        # Test Redis connection
        redis_client = redis.from_url(redis_url)
        redis_client.ping()
        logger.info("Redis connection successful")
    except Exception as e:
        logger.error("Failed to connect to Redis", error=str(e))
        raise
    
    # Create Redis broker
    broker = RedisBroker(url=redis_url)
    
    # Add middleware for better reliability and monitoring
    broker.add_middleware(CurrentMessage())
    broker.add_middleware(Retries(max_retries=3, min_backoff=1000, max_backoff=30000))
    broker.add_middleware(TimeLimit(time_limit=3600000))  # 1 hour timeout
    broker.add_middleware(Callbacks())
    broker.add_middleware(Pipelines())
    
    # Add results backend for progress tracking
    result_backend = RedisBackend(url=redis_url)
    broker.add_middleware(Results(backend=result_backend))
    
    # Set as global broker
    dramatiq.set_broker(broker)
    
    logger.info("Dramatiq broker configured successfully", 
                middleware_count=len(broker.middleware),
                has_results_backend=True)
    
    return broker

def get_broker():
    """Get the current broker instance"""
    try:
        return dramatiq.get_broker()
    except RuntimeError:
        logger.warning("No broker set, creating new one")
        return create_broker()

# Create the broker instance
try:
    broker = create_broker()
    logger.info("Dramatiq broker initialized")
except Exception as e:
    logger.error("Failed to initialize Dramatiq broker", error=str(e))
    raise 