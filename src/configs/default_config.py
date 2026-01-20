from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class DAFConfig:
    """
    [Superset Configuration]
    모든 모델의 설정을 여기서 통합 관리합니다.
    """

    # ==========================================
    # 1. Experiment Meta
    # ==========================================
    model_name: str = "daf_moe"  # [핵심] 'daf_moe', 'tabnet', 'xgboost' 등
    data_config_path: str = "" 
    dataset_name: str = "default"
    gpu_ids: str = "0"
    
    # [Data Constraints] (로더에서 자동 채움)
    n_numerical: int = 0
    n_categorical: int = 0
    n_features: int = 0
    total_cats: int = 0

    # ==========================================
    # 2. Common Hyperparameters
    # ==========================================
    task_type: str = 'classification' # 'classification' or 'regression'
    out_dim: int = 1
    
    batch_size: int = 256
    epochs: int = 50
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    
    # Transformer Common
    d_emb: int = 64
    n_layers: int = 4
    n_heads: int = 4
    dropout: float = 0.1

    # ==========================================
    # 3. [Model: DAF-MoE]
    # ==========================================
    n_experts: int = 4
    top_k: int = 2
    d_ff: int = 128
    router_noise_std: float = 0.1
    mu_init_strategy: str = 'uniform'
    
    # Loss Weights
    lambda_spec: float = 0.1
    lambda_repel: float = 0.01
    lambda_bal: float = 0.001

    # ==========================================
    # 4. [Model: Baseline Placeholders]
    # ==========================================
    # TabNet 등 다른 모델을 위한 자리를 미리 만들어둡니다.
    n_steps: int = 3
    gamma: float = 1.3
    
    # ==========================================
    # 5. Preprocessing
    # ==========================================
    n_quantiles: int = 1000
    subsample: int = 100000