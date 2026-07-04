import logging
import os
import json
from datetime import datetime

class StructuredLogger:
    _configured = False
    _log_file = "gesture_verse.log"

    @classmethod
    def configure(cls, log_file="gesture_verse.log"):
        if cls._configured:
            return
        
        # Resolve path relative to project root if not absolute
        if not os.path.isabs(log_file):
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            log_file = os.path.join(base_dir, log_file)
            
        cls._log_file = log_file
        
        log_dir = os.path.dirname(cls._log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            
        logger = logging.getLogger("GestureVerse")
        logger.setLevel(logging.INFO)
        logger.handlers.clear()

        # Console Handler with structured text format
        c_handler = logging.StreamHandler()
        c_formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        c_handler.setFormatter(c_formatter)
        logger.addHandler(c_handler)

        # File Handler
        f_handler = logging.FileHandler(cls._log_file, encoding='utf-8')
        
        # JSON-structured log format for files
        class JSONFormatter(logging.Formatter):
            def format(self, record):
                log_data = {
                    "timestamp": datetime.now().isoformat(),
                    "level": record.levelname,
                    "component": record.name,
                    "message": record.getMessage()
                }
                if hasattr(record, "extra_data"):
                    log_data.update(record.extra_data)
                return json.dumps(log_data)

        f_handler.setFormatter(JSONFormatter())
        logger.addHandler(f_handler)
        
        cls._configured = True
        print(f"[StructuredLogger] Configured. Log file: {cls._log_file}")

    @classmethod
    def get_logger(cls, name):
        cls.configure()
        return logging.getLogger(f"GestureVerse.{name}")
