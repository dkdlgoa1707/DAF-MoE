import torch
import torch.nn as nn

# 내 모델 Import
from .daf_moe_transformer import DAFMoETransformer

# 추후 추가될 베이스라인 모델들 (주석 처리)
# from .baselines.tabnet import TabNetWrapper

def create_model(config):
    """
    Config의 model_name에 따라 적절한 모델 객체를 반환합니다.
    """
    model_name = config.model_name.lower()
    print(f"🏭 Model Factory: Building '{model_name}'...")

    if model_name == 'daf_moe':
        return DAFMoETransformer(config)
    
    elif model_name == 'tabnet':
        # 나중에 TabNet 구현 후 주석 해제
        # return TabNetWrapper(config)
        raise NotImplementedError("TabNet is not implemented yet.")
        
    elif model_name == 'ft_transformer':
        raise NotImplementedError("FT-Transformer is not implemented yet.")
        
    else:
        raise ValueError(f"🚨 Unknown model name: {model_name}")