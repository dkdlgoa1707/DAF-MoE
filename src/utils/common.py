import os
import logging
from datetime import datetime
import random
import numpy as np
import torch

def seed_everything(seed):
    """
    Fixes random seeds for reproducibility across runs.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) 
    
    # Deterministic operations ensure identical results on the same hardware
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False 

def get_logger(log_dir, log_name=None):
    """
    Configures a logger to output to both file and console.
    """
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{log_name}_{timestamp}.log" if log_name else f"train_{timestamp}.log"
    log_path = os.path.join(log_dir, filename)

    logger = logging.getLogger(log_path)
    logger.setLevel(logging.INFO)
    logger.propagate = False 

    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter('[%(asctime)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger