import torch.nn as nn
from .daf_moe_loss import DAFLoss

def create_criterion(config, device):
    """
    Loss Function Factory.
    
    Returns:
        DAFLoss: If model is 'daf_moe', returns the custom joint objective (Eq. 15).
        nn.Module: Standard task loss (MSE, BCE, CE) for baseline models.
    """
    model_name = config.model_name.lower()
    
    # 1. DAF-MoE: Use Custom Joint Objective
    if model_name.startswith('daf_moe'):
        return DAFLoss(config).to(device)
        
    # 2. Baselines: Use Standard Task Loss
    else:
        if config.task_type == 'classification':
            if config.out_dim == 1:
                return nn.BCEWithLogitsLoss().to(device)
            else:
                return nn.CrossEntropyLoss().to(device)
        else: # Regression
            return nn.MSELoss().to(device)