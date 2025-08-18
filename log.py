import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from datetime import datetime

def setup_log(name = 'main_logger', dir = 'logs', 
              console_level = logging.INFO, file_level = logging.INFO, 
              max_bytes = 10 * 1024 * 1024, backup_count = 5):
    os.makedir(dir, exist_ok = True)

    log_filename = datetime.now().strftime("%Y-%m-%d_%H-%M-%S.log")
    log_filepath = os.path.join(dir, log_filename)

    logger = logging.getLogger(name)
    logger.setLevel(min(console_level, file_level))
    if logger.handlers:
        return logger

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s')

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    console_handler.setFormatter(formatter)

    file_handler = RotatingFileHandler(log_filepath, maxBytes = max_bytes, 
                                       backupCount = backup_count, encoding = 'utf-8', )
    file_handler.setLevel(file_level)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger

logger = setup_log()