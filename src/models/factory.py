import torch
import torch.nn as nn

# Models Import
from .daf_moe.daf_moe_transformer import DAFMoETransformer
from .daf_moe_v15.daf_moe_transformer import DAFMoETransformerV15
from .daf_moe_v2.daf_moe_transformer import DAFMoETransformerV2
from .baselines.ft_transformer import FTTransformerWrapper
from .baselines.mlp import MLP
from .baselines.resnet import TabularResNet
from .baselines.tabm import TabMWrapper

def create_model(config):
    """
    Model Factory
    =============
    Instantiates the appropriate model based on the configuration.
    
    Args:
        config (DAFConfig): Configuration object containing model hyperparameters.
        
    Returns:
        nn.Module: The initialized PyTorch model.
    """
    model_name = config.model_name.lower()

    if model_name.startswith('daf_moe_v2'):
        return DAFMoETransformerV2(config)

    if model_name.startswith('daf_moe_v15'):
        return DAFMoETransformerV15(config)
    
    # 1. Proposed Model (DAF-MoE)
    if model_name.startswith('daf_moe'):
        return DAFMoETransformer(config)
    
    # 2. Deep Learning Baselines
    elif model_name == 'ft_transformer':
        return FTTransformerWrapper(config)
    
    elif model_name == 'mlp':
        return MLP(config)
        
    elif model_name == 'resnet':
        return TabularResNet(config)

    elif model_name == 'tabm':
        return TabMWrapper(config)

    # 3. GBDT Models (Handled separately in runners/run_trees.py)
    elif model_name in ['xgboost', 'catboost']:
        raise ValueError(
            "GBDT models (XGBoost, CatBoost) are not PyTorch modules. "
            "Please use `runners/run_trees.py` for training and evaluation."
        )
        
    else:
        raise ValueError(f"Unknown model architecture: {model_name}")