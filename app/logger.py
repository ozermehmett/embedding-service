import logging
import os
from dotenv import load_dotenv

# .env dosyasını yükle
load_dotenv()

def setup_logger(name: str = None) -> logging.Logger:
    """Logger ayarları"""
    
    log_level = os.getenv('LOG_LEVEL', 'INFO').upper()
    log_format = os.getenv('LOG_FORMAT', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    service_name = os.getenv('SERVICE_NAME', 'embedding-service')
    
    logger_name = f"{service_name}.{name}" if name else service_name
    logger = logging.getLogger(logger_name)
    
    logger.setLevel(getattr(logging, log_level))
    
    if not logger.handlers:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, log_level))
        
        formatter = logging.Formatter(log_format)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(console_handler)
        
        logger.propagate = False
    
    return logger
