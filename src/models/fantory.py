import torch
import torch.nn as nn
from .daf_moe_transformer import DAFMoETransformer

# [Tip] Baseline 모델들은 src/models/baselines/ 폴더에 넣어두고 여기서 import 합니다.
# from .baselines.tabnet import TabNet
# from .baselines.ft_transformer import FTTransformer

def create_model(config):
    """
    Config의 model_name에 따라 모델 객체를 생성하여 반환하는 Factory 함수
    """
    model_name = config.model_name.lower()
    
    print(f"🏭 Model Factory: Building '{model_name}'...")

    # 1. DAF-MoE (제안 모델)
    if model_name == 'daf_moe':
        return DAFMoETransformer(config)

    # 2. TabNet (비교 모델 예시)
    elif model_name == 'tabnet':
        try:
            from .baselines.tabnet import TabNet
        except ImportError:
            raise ImportError("🚨 TabNet module not found. Please implement it in src/models/baselines/")
            
        return TabNet(
            input_dim=config.n_features,
            output_dim=config.out_dim,
            n_d=config.n_d,      # Config에서 TabNet 전용 변수 매핑
            n_a=config.n_a,
            n_steps=config.n_steps,
            gamma=config.gamma,
            lambda_sparse=1e-3
        )

    # 3. FT-Transformer (비교 모델 예시)
    elif model_name == 'ft_transformer':
        try:
            from .baselines.ft_transformer import FTTransformer
        except ImportError:
             raise ImportError("🚨 FT-Transformer module not found.")

        return FTTransformer(
            categories=config.total_cats,
            num_continuous=config.n_numerical,
            dim=config.d_emb,
            depth=config.n_layers,
            heads=config.n_heads,
            attn_dropout=config.dropout,
            ff_dropout=config.dropout
        )

    else:
        raise ValueError(f"🚨 Unknown model name: '{model_name}'. Available: ['daf_moe', 'tabnet', 'ft_transformer']")