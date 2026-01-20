import torch.nn as nn
from .daf_moe_loss import DAFLoss

def create_criterion(config, device):
    """
    모델 이름에 따라 적절한 Loss Function을 반환합니다.
    """
    model_name = config.model_name.lower()
    
    # 1. DAF-MoE: 전용 커스텀 로스 사용
    if model_name == 'daf_moe':
        return DAFLoss(config).to(device)
        
    # 2. 그 외 모델: 일반적인 Loss 사용
    else:
        if config.task_type == 'classification':
            if config.out_dim == 1:
                return nn.BCEWithLogitsLoss().to(device)
            else:
                return nn.CrossEntropyLoss().to(device)
        else: # Regression
            return nn.MSELoss().to(device)