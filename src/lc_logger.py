import logging 
import logging.handlers
from pathlib import Path 

class lcurve_logging:
    def __init__(self, stage_name: str, log_file: str): 
        self.stage_name = stage_name
        self.log_file = log_file

    def setup_logger(self) -> logging.Logger:
        Path(self.log_file).parent.mkdir(parents=True, exist_ok=True)

        logger = logging.getLogger(self.stage_name)
        logger.setLevel(logging.INFO)

        if logger.handlers:
            return logger

        formatter = logging.Formatter(
            "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler = logging.handlers.RotatingFileHandler(
            self.log_file, maxBytes=10 * 1024 * 1024, backupCount=5
            )
        file_handler.setFormatter(formatter)
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        return logger