import torch
import torch.nn as nn

# Models Import
from .daf_moe.daf_moe_transformer import DAFMoETransformer
from .daf_moe_v15.daf_moe_transformer import DAFMoETransformerV15
from .daf_moe_v2.daf_moe_transformer import DAFMoETransformerV2
from .baselines.ft_transformer import FTTransformerWrapper
from .baselines.mlp import MLP
from .baselines.resnet import TabularResNet
from .baselines.tabm import TabMWrapper, TabMPLEWrapper
from .baselines.tabr import TabRWrapper
from .baselines.modernnca import ModernNCAWrapper
from .baselines.config_validation import validate_model_config


def create_model(config):
    """
    Model Factory
    =============
    Instantiates the appropriate model based on the configuration.

    Args:
        config (DAFConfig): Configuration object containing hyperparameters.

    Returns:
        nn.Module: The initialized PyTorch model.
    """
    model_name = config.model_name.lower()
    validate_model_config(config)

    if model_name.startswith('daf_moe_v2'):
        return DAFMoETransformerV2(config)

    if model_name.startswith('daf_moe_v15'):
        return DAFMoETransformerV15(config)

    if model_name.startswith('daf_moe'):
        return DAFMoETransformer(config)

    if model_name == 'ft_transformer':
        return FTTransformerWrapper(config)

    if model_name == 'mlp':
        return MLP(config)

    if model_name == 'resnet':
        return TabularResNet(config)

    if model_name == 'tabm':
        return TabMWrapper(config)

    if model_name == 'tabm_ple':
        return TabMPLEWrapper(config)

    if model_name == 'tabr':
        return TabRWrapper(config)

    if model_name == 'modernnca':
        return ModernNCAWrapper(config)

    if model_name in {'xgboost', 'catboost', 'realmlp', 'tabicl'}:
        raise ValueError(
            f"{model_name} is an official/native Phase 2 estimator, not a PyTorch "
            "module. Use `runners/run_phase2_native.py`."
        )

    raise ValueError(f"Unknown model architecture: {model_name}")
