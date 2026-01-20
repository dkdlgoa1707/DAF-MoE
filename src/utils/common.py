import os
import random
import torch
import numpy as np
import logging
from datetime import datetime

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_logger(log_dir):
    """
    파일과 콘솔에 동시에 로그를 남기는 Logger 생성
    """
    os.makedirs(log_dir, exist_ok=True)
    
    # 로그 파일 이름: train_20260120_123000.log
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"train_{timestamp}.log")

    # Logger 설정
    logger = logging.getLogger("DAF-MoE")
    logger.setLevel(logging.INFO)
    logger.propagate = False # 중복 출력 방지

    # Formatter
    formatter = logging.Formatter('[%(asctime)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    # 1. File Handler (파일에 쓰기)
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # 2. Stream Handler (터미널에 출력)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger