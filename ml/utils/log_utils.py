import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime

class CustomFormatter(logging.Formatter):
    """Custom formatter with different colors for different log levels"""
    
    grey = "\x1b[38;20m"
    blue = "\x1b[34;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    
    format_template = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    FORMATS = {
        logging.DEBUG: grey + format_template + reset,
        logging.INFO: blue + format_template + reset,
        logging.WARNING: yellow + format_template + reset,
        logging.ERROR: red + format_template + reset,
        logging.CRITICAL: bold_red + format_template + reset
    }
    
    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)

class Logger:
    """Centralized logging utility for the application"""
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Logger, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not Logger._initialized:
            self.logger = logging.getLogger("text_classification")
            self.logger.setLevel(logging.INFO)
            self._setup_handlers()
            Logger._initialized = True
    
    def _setup_handlers(self):
        """Set up console and file handlers"""
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(CustomFormatter())
        self.logger.addHandler(console_handler)
        
        # File handler
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"text_classification_{timestamp}.log"
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
        self.logger.addHandler(file_handler)
    
    @classmethod
    def get_logger(cls, name: Optional[str] = None) -> logging.Logger:
        """
        Get a logger instance with the specified name.
        
        Args:
            name (Optional[str]): Name for the logger. If None, returns the main logger.
            
        Returns:
            logging.Logger: Logger instance
        """
        instance = cls()
        return instance.logger

# Create a default logger instance
logger = Logger.get_logger()