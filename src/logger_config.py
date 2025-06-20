import logging
import structlog
import sys
import os
from datetime import datetime
from pathlib import Path

def setup_logging(debug: bool = False, log_file: str = None):
    """
    Setup comprehensive logging for the energy forecasting system.
    
    Args:
        debug: Enable debug level logging
        log_file: Optional log file path
    """
    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Determine log level
    log_level = logging.DEBUG if debug else logging.INFO
    
    # Configure structlog
    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="ISO"),
            structlog.processors.add_log_level,
            structlog.dev.ConsoleRenderer(colors=True)
        ],
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        logger_factory=structlog.WriteLoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Console handler with colors
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    
    # Console formatter with colors
    console_format = "%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s"
    
    try:
        import colorlog
        console_formatter = colorlog.ColoredFormatter(
            "%(log_color)s%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            log_colors={
                'DEBUG': 'cyan',
                'INFO': 'green',
                'WARNING': 'yellow',
                'ERROR': 'red',
                'CRITICAL': 'red,bg_white',
            }
        )
        console_handler.setFormatter(console_formatter)
    except ImportError:
        console_formatter = logging.Formatter(console_format, datefmt="%Y-%m-%d %H:%M:%S")
        console_handler.setFormatter(console_formatter)
    
    root_logger.addHandler(console_handler)
    
    # File handler (always enabled)
    if log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"energy_forecasting_{timestamp}.log"
    
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)  # Always debug level for files
    
    file_format = "%(asctime)s | %(levelname)-8s | %(name)-20s | %(funcName)-15s | %(lineno)-4d | %(message)s"
    file_formatter = logging.Formatter(file_format, datefmt="%Y-%m-%d %H:%M:%S")
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)
    
    # Configure specific loggers
    configure_specific_loggers(debug)
    
    logger = structlog.get_logger("energy_forecasting.setup")
    logger.info("Logging system initialized", 
                debug_mode=debug, 
                log_file=str(log_file),
                log_level=logging.getLevelName(log_level))
    
    return logger

def configure_specific_loggers(debug: bool):
    """Configure logging levels for specific modules"""
    
    # Dramatiq loggers
    logging.getLogger("dramatiq").setLevel(logging.INFO)
    logging.getLogger("dramatiq.Worker").setLevel(logging.INFO)
    logging.getLogger("dramatiq.middleware").setLevel(logging.INFO if not debug else logging.DEBUG)
    
    # Redis logger (can be noisy)
    logging.getLogger("redis").setLevel(logging.WARNING)
    
    # InfluxDB logger
    logging.getLogger("influxdb_client").setLevel(logging.INFO if not debug else logging.DEBUG)
    
    # HTTP libraries (can be very noisy)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    
    # FastAPI/Uvicorn
    logging.getLogger("uvicorn").setLevel(logging.INFO)
    logging.getLogger("fastapi").setLevel(logging.INFO if not debug else logging.DEBUG)
    
    # XGBoost (can be verbose)
    logging.getLogger("xgboost").setLevel(logging.WARNING)
    
    # Optuna (can be verbose during hyperparameter optimization)
    logging.getLogger("optuna").setLevel(logging.INFO)

def get_logger(name: str):
    """Get a structured logger for a specific module"""
    return structlog.get_logger(name)

class TaskLogger:
    """Logger specifically for background tasks with context"""
    
    def __init__(self, task_name: str, task_id: str = None):
        self.logger = structlog.get_logger(f"task.{task_name}")
        self.task_id = task_id
        self.context = {"task_name": task_name}
        if task_id:
            self.context["task_id"] = task_id
    
    def bind(self, **kwargs):
        """Add context to logger"""
        self.context.update(kwargs)
        return self
    
    def debug(self, message, **kwargs):
        self.logger.debug(message, **{**self.context, **kwargs})
    
    def info(self, message, **kwargs):
        self.logger.info(message, **{**self.context, **kwargs})
    
    def warning(self, message, **kwargs):
        self.logger.warning(message, **{**self.context, **kwargs})
    
    def error(self, message, **kwargs):
        self.logger.error(message, **{**self.context, **kwargs})
    
    def critical(self, message, **kwargs):
        self.logger.critical(message, **{**self.context, **kwargs})

# Convenience functions
def debug_mode_from_env() -> bool:
    """Check if debug mode should be enabled from environment"""
    return os.getenv("DEBUG", "false").lower() in ("true", "1", "yes", "on") 